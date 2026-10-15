# Plan: Vast.ai GPU Orchestration — IndicF5 narration + LTX-Video 2B 0.9.8 distilled (HFIE)

## Goal
- Narration → IndicF5 on Vast GPU with local CPU fallback (remove Indic Parler-TTS completely).
- GPU → RTX 4090 only.
- Video → LTX-Video 2B 0.9.8 distilled (HFIE) single-stage, 12 fps + RIFE interpolation to 24 fps, I2V from stills, clip reuse.
- Time target → ~2-4 s generation per 1 s of finished video (generate FEWER/faster seconds).
- Quality target → raise quality at the SAME budget (move work out of the diffusion steps).
- Provisioning target → cold start ~1-3 min (baked image + model volume) instead of ~30-80 min.

## Decisions
- **Narration** → IndicF5 on Vast (GPU) + local CPU subprocess fallback; run **float32** (fp16 previously produced NaN → DC/silent WAVs).
- **Image gen** → RTX 4090, FLUX.1-schnell. Single GPU type.
- **Image fallback** → local SSD-1B (`IMAGE_MODEL` default) when the GPU is unavailable.
- **Video generation** → LTX-Video 2B 0.9.8 distilled (HFIE), single-stage, 512x768 or 576x1024, 12 fps → RIFE 24 fps; no CPU offload; reuse a clip pool; 2D (Ken Burns/parallax) on stills for B-roll.
- **Provisioning** → prebaked Docker image (env only) + Vast network volume for weights + hf_transfer.
- **Instance reuse** → any existing 4090; stop (never destroy when reusable). Assets saved locally; stop after assets.
- **Indic Parler** → remove completely.
- **ComfyUI / LTX-2.5** → removed entirely from scope.
- **Staging** → narration/GPU (Phases 1-6) → provisioning (7) → video (8) → tests/docs (9).

## Verified facts / constraints
- IndicF5 fp16 NaN bug is why Indic Parler exists → float32 + audio guards + cache version bump.
- `--onstart-cmd` is baked at creation → onstart changes need destroy + re-rent.
- LTX constraints: `num_frames % 8 == 1`; W/H `% 32 == 0`. For a 9:16 1080 final, generate **576x1024**.
- Cold start today ≈ 15–40 min build + model load; dominated by net bandwidth.

## Key files
- `scripts/vast_manager.py`, `scripts/vast_server.py`, `scripts/tts.py`, `scripts/new_video.py`,
  `scripts/generate_images.py`, `scripts/generate_videos.py`, `scripts/batch.py`,
  `examples/indicf5/indicf5_batch.py`, `src/types.ts`, `README.md`, `.env.example`

## Phase 0 — Spike gate (do FIRST; blocks video validation)
0.1 Rent one 4090; time ONE LTX-Video 2B 0.9.8 distilled (HFIE) 576x1024 single-stage clip.
0.2 Time 12 fps generation + RIFE interpolation to 24 fps; record peak VRAM + wall time.
Gate: if 2B is not ~<15 s/clip, re-plan.

## Phase 1 — Add IndicF5 (server + client), KEEP Parler until verified
1.1 `vast_manager.py`: py3.11 venv (`/workspace/venv-indicf5`) with CUDA torch, float32.
1.2 `vast_manager.py`: venv dep list + probe (MUST import the lazy module `f5_tts.infer.utils_infer` + `vocos` via the venv interpreter) + separate readiness marker.
1.3 `vast_manager.py`: prefetch `ai4bharat/IndicF5` (gated → HF_TOKEN; env/mirror override).
1.4 `vast_server.py`: `_INDICF5_VENV` + subprocess runner (`indicf5_batch.py` protocol); IndicF5 `handle_audio_synthesize`/`_batch` (voice = reference-clip stem); `/health` += `indicf5_ready`/`indicf5_error`.
1.5 `tts.py`: add the MISSING remote IndicF5 client — route `indicf5_*` to `VastManager.synthesize_tts(voice=<reference>)` when `VAST_SERVER_URL` is set, else local subprocess. Leave Parler intact.

## Phase 2 — Verify live (gate before deleting Parler)
2.1 Destroy + re-rent. 2.2 `/health` `indicf5_ready: true`. 2.3 non-NaN audio; no -32768 DC WAVs.

## Phase 3 — Remove Indic Parler-TTS
3.1 `tts.py`: remove `INDIC_PARLER_*`, `indic_parler_model_candidates()`, `IndicParlerEngine`, `synthesize_indic_parler()`, `_pace_word()`; remove `indic_parler_*` aliases.
3.2 `vast_server.py`: remove `_INDIC_PARLER_ENGINE`, `load_indic_parler_engine()`, `_resolve_indic_description()`, `_pace_word()`, `tts_import_error()`; `/health` → IndicF5; update `_release_other_models` tuple.
3.3 `vast_manager.py`: drop `parler-tts` dep + `parler_tts` probe; `DEFAULT_PREFETCH_MODELS` → IndicF5; align `repair_dependencies()`.
3.4 `new_video.py`: remove `indic_parler`/`parler` from `--audio-model` + `voice_requests_parler`/`indic_engine_selected`; auto-rent keyed to `indicf5`; `vast_prefetch = ["ai4bharat/IndicF5"]`.
3.5 Bump `AUDIO_PIPELINE_VERSION`; purge cache; drop `'indic_parler'` from `src/types.ts`.

## Phase 4 — Always RTX 4090
4.1 `vast_manager.py`: remove the `"RTX 3090"` budget; delete `search_rtx_3090()`; remove `conflicting_instances()`.
4.2 `new_video.py`: `expected_gpu = "RTX 4090"` always; drop the conflict block; `require_tts = (audio_model == "indicf5")`.
4.3 `generate_images.py`: `provider_title` → "Vast.ai Cloud RTX 4090 GPU".

## Phase 5 — Fallback: video → local SSD-1B images
5.1 `new_video.py`: when `start_vast()` fails and `visual_mode == "video"`, switch to `image` with local SSD-1B + local IndicF5 narration instead of `return 1`.
5.2 Implicit fallback for video mode.

## Phase 6 — require_gpu() + onstart
6.1 `vast_manager.py`: `require_gpu(require_tts=)` checks the IndicF5 venv readiness via `/health`.
6.2 `server_onstart_command()` rewrite: uv → in parallel {main deps, py3.11 IndicF5 venv, HF prefetch} → clone/refresh → start `vast_server.py`.
6.3 Note: onstart changes need destroy + re-rent.

## Phase 7 — Fast provisioning (cold start ~1-3 min)
7.1 Bake a custom Docker image (`VAST_SERVER_IMAGE`): main deps + IndicF5 py3.11 venv, ENV ONLY (no weights). Push to Docker Hub / GHCR → removes apt/uv/venv/install from onstart.
7.2 Model volume: mount a Vast network volume, set `HF_HOME` to it → weights download once and are reused by every instance (FLUX, IndicF5, LTX-Video 2B HFIE).
7.3 Add `hf_transfer` + `HF_HUB_ENABLE_HF_TRANSFER=1` to the image (multi-GB pulls 2-5× faster).
7.4 Use `snapshot_download(allow_patterns=[...])` to fetch ONLY needed files.
7.5 Replace deadsnakes with `uv python install 3.11`; pin the CUDA torch index.
7.6 Make the py3.11 venv install a third parallel job.
7.7 Shrink downloads: FLUX fp8 (~12 GB vs ~24 GB); LTX-Video weights.
7.8 Prefer fast hosts (`inet_down ≥ 1000`, `disk_bw ≥ 500`) + blacklist slow ones.
7.9 Target budget: cold start ~1-3 min (image pull) + model load ~1 min.

## Phase 8 — Video pipeline: LTX-Video 2B 0.9.8 distilled (HFIE)
8.1 `generate_videos.py`: integrate LTX-Video 2B 0.9.8 distilled (HFIE) single-stage 576x1024; no CPU offload; conv VAE.
8.2 12 fps generation → RIFE interpolation to 24 fps.
8.3 Cache generated clips keyed by prompt; reuse a small pool across similar scenes.
8.4 Route B-roll to 2D motion (Ken Burns/parallax on stills) where appropriate.
8.5 Quality enhancements:
    - I2V from existing stills for composition/detail.
    - Structured prompts (action → gesture → appearance → environment → camera → lighting, ≤ ~200 words).
    - Fixed seed per scene for consistency.
    - Lanczos upscale to 1080; consistent LUT/grade; film grain + motion blur.

## Phase 9 — Tests & docs
9.1 `test_providers.py`: keep `test_vast_call_sites()` green; 4090-always routing. `test_audio.py`: Parler → IndicF5.
9.2 `README.md` + `.env.example`: audio model, GPU, video pipeline, quality checklist, provisioning, RTF.

## Verification
1. `vast_manager.py search` — RTX 4090 offers only.
2. Dry runs (image + video) — 4090 selected.
3. Image then video run — same 4090 reused.
4. Fresh `/health` `indicf5_ready`; WAV peak/RMS not DC (`-32768`).
5. `/v1/audio/synthesize` works with an IndicF5 reference voice.
6. Simulated GPU failure in video mode — local SSD-1B images + local IndicF5 (no abort).
7. Video: LTX-Video 2B HFIE single-stage 576x1024 12 fps clip < ~15 s; RIFE output is 24 fps and smooth.
8. Reuse/caching: a repeated prompt is served from cache.
9. Quality: I2V-from-still visibly cleaner than T2V at the same step count; seed reproducible.
10. Provisioning: baked image + volume → cold start ~1-3 min.
11. `npm test` exits 0; instance stops after asset gen.
