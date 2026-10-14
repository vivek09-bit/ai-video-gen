# Plan: Vast.ai GPU Orchestration — IndicF5 narration + fast/hero video paths

## Goal
- Narration → IndicF5 on Vast GPU with local CPU fallback (remove Indic Parler-TTS completely).
- GPU → RTX 4090 only.
- Video → FAST path (LTX-Video 2B, single-stage, 12 fps + RIFE, I2V from stills, clip reuse) for the
  bulk; HERO path (LTX-2.5 GGUF Q4_K_M via ComfyUI) for a few shots.
- Time target → ~2-4 s generation per 1 s of finished video (generate FEWER/faster seconds).
- Quality target → raise quality at the SAME budget (move work out of the diffusion steps).
- Provisioning target → cold start ~1-3 min (baked image + model volume) instead of ~30-80 min.

## Decisions
- **Narration** → IndicF5 on Vast (GPU) + local CPU subprocess fallback; run **float32**
  (fp16 previously produced NaN → DC/silent WAVs).
- **Image gen** → RTX 4090, FLUX.1-schnell. Single GPU type.
- **Image fallback** → local SSD-1B (`IMAGE_MODEL` default) when the GPU is unavailable.
- **Video FAST path** → LTX-Video 2B, single-stage, 512x768, 12 fps → RIFE 24 fps; no CPU offload;
  reuse a clip pool; 2D (Ken Burns/parallax) on stills for B-roll.
- **Video HERO path** → LTX-2.5 22B GGUF Q4_K_M (`realrebelai/LTX-2.5_GGUFs`) via a ComfyUI runtime,
  4090 + heavy offload, single-stage, audio off, hero shots only.
- **Provisioning** → prebaked Docker image (env only) + Vast network volume for weights + hf_transfer.
- **Instance reuse** → any existing 4090; stop (never destroy when reusable). Assets saved locally;
  stop after assets.
- **Indic Parler** → remove completely.
- **Staging** → narration/GPU (Phases 1-6) → provisioning (7) → video (8-9) → tests/docs (10).

## Verified facts / constraints
- IndicF5 fp16 NaN bug is why Indic Parler exists → float32 + audio guards + cache version bump.
- `--onstart-cmd` is baked at creation → onstart changes need destroy + re-rent.
- `Lightricks/LTX-2.5` is 22B dual-stream (video+audio), gated; needs Gemma4-12B TE + video VAE +
  audio VAE. The GGUF replaces ONLY the transformer.
- `realrebelai/LTX-2.5_GGUFs` is ComfyUI-format (city96 ComfyUI-GGUF); `LTX-2.5-Distilled-Q4_K_M.gguf`
  (~15 GB). Its `config` blob is a GGUF KV field → ComfyUI reads it; diffusers does NOT.
- Diffusers LTX-2.5 = git-main only; `LTX2ImageToVideoPipeline`; pack `Lightricks/LTX-2.5-Diffusers`.
- LTX constraints: `num_frames % 8 == 1`; W/H `% 32 == 0`. For a 9:16 1080 final, generate **576x1024**.
- 22B + 12B TE will not fit 24 GB without offload (offload is the main wall-clock killer).
- LTX-2.5 on a 4090 ≈ RTF 5-11. RTF 2-4 is NOT reachable on the 22B; needs a small model / low fps / 2D.
- Cold start today ≈ 15–40 min build + 10–40 min hero stack + model load; dominated by net bandwidth.

## Key files
- `scripts/vast_manager.py`, `scripts/vast_server.py`, `scripts/tts.py`, `scripts/new_video.py`,
  `scripts/generate_images.py`, `scripts/generate_videos.py`, `scripts/batch.py`,
  `examples/indicf5/indicf5_batch.py`, `src/types.ts`, `README.md`, `.env.example`

## Phase 0 — Spike gate (do FIRST; blocks Phase 9)
0.1 Rent one 4090; time ONE LTX-Video 2B 512x768 single-stage clip (fast-path baseline).
0.2 Time ONE LTX-2.5 Q4_K_M 121-frame 576x1024 clip; record peak VRAM + wall time.
0.3 Confirm ComfyUI runs on the current base image (torch 2.4 / CUDA 12.4) or identify the needed tag.
Gate: if 2B is not ~<15 s/clip or 2.5 not ~<6 min/clip, re-plan. If the 22B OOMs despite offload,
allow a 48 GB tier for hero shots only.

## Phase 1 — Add IndicF5 (server + client), KEEP Parler until verified
1.1 `vast_manager.py`: py3.11 venv (`/workspace/venv-indicf5`) with CUDA torch, float32.
1.2 `vast_manager.py`: venv dep list + probe (MUST import the lazy module
    `f5_tts.infer.utils_infer` + `vocos` via the venv interpreter) + separate readiness marker.
1.3 `vast_manager.py`: prefetch `ai4bharat/IndicF5` (gated → HF_TOKEN; env/mirror override).
1.4 `vast_server.py`: `_INDICF5_VENV` + subprocess runner (`indicf5_batch.py` protocol); IndicF5
    `handle_audio_synthesize`/`_batch` (voice = reference-clip stem); `/health` +=
    `indicf5_ready`/`indicf5_error`.
1.5 `tts.py`: add the MISSING remote IndicF5 client — route `indicf5_*` to
    `VastManager.synthesize_tts(voice=<reference>)` when `VAST_SERVER_URL` is set, else local subprocess.
    Leave Parler intact.

## Phase 2 — Verify live (gate before deleting Parler)
2.1 Destroy + re-rent. 2.2 `/health` `indicf5_ready: true`. 2.3 non-NaN audio; no -32768 DC WAVs.

## Phase 3 — Remove Indic Parler-TTS
3.1 `tts.py`: remove `INDIC_PARLER_*`, `indic_parler_model_candidates()`, `IndicParlerEngine`,
    `synthesize_indic_parler()`, `_pace_word()`; remove `indic_parler_*` aliases.
3.2 `vast_server.py`: remove `_INDIC_PARLER_ENGINE`, `load_indic_parler_engine()`,
    `_resolve_indic_description()`, `_pace_word()`, `tts_import_error()`; `/health` → IndicF5; update
    `_release_other_models` tuple.
3.3 `vast_manager.py`: drop `parler-tts` dep + `parler_tts` probe; `DEFAULT_PREFETCH_MODELS` → IndicF5;
    align `repair_dependencies()`.
3.4 `new_video.py`: remove `indic_parler`/`parler` from `--audio-model` + `voice_requests_parler`/
    `indic_engine_selected`; auto-rent keyed to `indicf5`; `vast_prefetch = ["ai4bharat/IndicF5"]`.
3.5 Bump `AUDIO_PIPELINE_VERSION`; purge cache; drop `'indic_parler'` from `src/types.ts`.

## Phase 4 — Always RTX 4090
4.1 `vast_manager.py`: remove the `"RTX 3090"` budget; delete `search_rtx_3090()`; remove
    `conflicting_instances()`.
4.2 `new_video.py`: `expected_gpu = "RTX 4090"` always; drop the conflict block;
    `require_tts = (audio_model == "indicf5")`.
4.3 `generate_images.py`: `provider_title` → "Vast.ai Cloud RTX 4090 GPU".

## Phase 5 — Fallback: video → local SSD-1B images
5.1 `new_video.py`: when `start_vast()` fails and `visual_mode == "video"`, switch to `image` with local
    SSD-1B + local IndicF5 narration instead of `return 1`.
5.2 Decide implicit vs gated behind `--allow-local-fallback` (recommend implicit for video mode).

## Phase 6 — require_gpu() + onstart
6.1 `vast_manager.py`: `require_gpu(require_tts=)` checks the IndicF5 venv readiness via `/health`.
6.2 `server_onstart_command()` rewrite: uv → in parallel {main deps, py3.11 IndicF5 venv, HF prefetch}
    → clone/refresh → start `vast_server.py`.
6.3 Note: onstart changes need destroy + re-rent.

## Phase 7 — Fast provisioning (cold start ~1-3 min)
7.1 Bake a custom Docker image (`VAST_SERVER_IMAGE`): main deps + IndicF5 py3.11 venv + ComfyUI stack,
    ENV ONLY (no weights). Push to Docker Hub / GHCR → removes apt/uv/venv/install from onstart.
7.2 Model volume: mount a Vast network volume, set `HF_HOME` to it → weights download once and are
    reused by every instance (FLUX, IndicF5, LTX-2.5 Q4 + TE + VAEs).
7.3 Add `hf_transfer` + `HF_HUB_ENABLE_HF_TRANSFER=1` to the image (multi-GB pulls 2-5× faster).
7.4 Use `snapshot_download(allow_patterns=[...])` to fetch ONLY needed files (one `.gguf`, TE, VAEs).
7.5 Replace deadsnakes with `uv python install 3.11`; pin the CUDA torch index (no resolver search).
7.6 Make the py3.11 venv install a THIRD parallel job (not serialized after apt).
7.7 Shrink downloads: FLUX fp8 (~12 GB vs ~24 GB); LTX-2.5 int8 TE; skip audio VAE/upscalers when unused.
7.8 Prefer fast hosts (`inet_down ≥ 1000`, `disk_bw ≥ 500`) + blacklist slow ones (existing logic).
7.9 Target budget: cold start ~1-3 min (image pull) + model load ~1-3 min (hero stack).

## Phase 8 — FAST video path (the RTF target)
8.1 `generate_videos.py`: default LTX-Video 2B single-stage 512x768; no CPU offload; conv VAE;
    prompt-enhancer off.
8.2 12 fps generation → RIFE interpolation to 24 fps (new pipeline step).
8.3 Cache generated clips keyed by prompt; reuse a small pool across similar scenes.
8.4 `new_video.py`: `--video-quality {fast,hero}`; route B-roll to 2D motion (Ken Burns/parallax on
    stills); reserve LTX for shots needing real motion.
8.5 Document the cost model: RTF ~1-3 for 2B; whole-video RTF < 1 via 2D + reuse.

### Phase 8 QUALITY (same time budget — zero added steps)
8.6  I2V from existing stills (biggest free win): still carries composition/detail; model adds motion.
8.7  Prompt discipline: LTX structured prompt (action → gesture → appearance → environment → camera →
     lighting, ≤ ~200 words); negative prompt; per-scene template; LLM enhancement done once and cached.
8.8  Consistency: fixed seed per scene; style LoRA (interchangeable across 2.3/2.5); reference image /
     multishot for character + palette + lighting continuity.
8.9  Spend the same seconds better: A/B the resolution↔steps sweet spot; Real-ESRGAN upscale instead of
     extra steps; diffusion VAE only where detail matters; `fp8-cast` to free memory → spend on RESOLUTION.
8.10 RIFE smooths temporal flicker (cheaper AND cleaner).
8.11 Render-side polish (≈0 GPU): Lanczos to 1080; consistent LUT/grade; film grain + motion blur;
     temporal denoise; frame blending at cuts.
8.12 Keep Q4 (not Q2/Q3) for hero; CFG 1.0 (distilled weights bake guidance in).

## Phase 9 — HERO video path: LTX-2.5 GGUF Q4_K_M via ComfyUI
9.1 `vast_manager.py`: optional heavy stage `install_comfyui_stack` (isolated py3.12 venv; ComfyUI +
    `city96/ComfyUI-GGUF` + LTX-2.5 nodes; `--lowvram` on :8188). Main env untouched; prefer baked in image.
9.2 Prefetch (Phase 7 volume): `LTX-2.5-Distilled-Q4_K_M.gguf` + Gemma4-12B-with-proj TE + video VAE +
    audio VAE (gated → HF_TOKEN).
9.3 `tools/comfyui/ltx25_q4km_workflow.json` (API): Unet Loader (GGUF) → Q4_K_M, distilled schedule,
    CFG 1.0, TE + VAE. SINGLE-STAGE (skip 2x upsampler; upscale to 1080 outside LTX). Audio OFF.
    Enforce `%8==1` / `%32==0`.
9.4 `vast_server.py`: ComfyUI supervisor + rewrite `/v1/videos/generate` (inject prompt/seed/W/H/frames
    → POST `/prompt` → poll `/history` → fetch mp4). `/health` += `comfyui_ready`. Diffusers path behind
    `VAST_LTX_BACKEND=diffusers|comfyui`.
9.5 `generate_videos.py` / `new_video.py --video-model` / `batch.py` / `README.md`: route hero shots to
    the ComfyUI backend (3-call-site sweep).
9.6 Memory: sequential TE → free → DiT load to avoid per-step offload where possible; `--reserve-vram`.
9.7 Quality: Q4_K_M (not lower); CFG 1.0; strong prompt + negative; single-stage + ESGAN upscale.
9.8 Revert safety: `VAST_LTX_BACKEND` + `VAST_VIDEO_MODEL` toggles; keep 2B as default until verified.

## Phase 10 — Tests & docs
10.1 `test_providers.py`: keep `test_vast_call_sites()` green; 4090-always routing. `test_audio.py`:
     Parler → IndicF5.
10.2 `README.md` + `.env.example`: audio model, GPU, fast/hero video, quality checklist, provisioning, RTF.

## Verification
1. `vast_manager.py search` — RTX 4090 offers only.
2. Dry runs (image + video) — 4090 selected.
3. Image then video run — same 4090 reused.
4. Fresh `/health` `indicf5_ready`; WAV peak/RMS not DC (`-32768`).
5. `/v1/audio/synthesize` works with an IndicF5 reference voice.
6. Simulated GPU failure in video mode — local SSD-1B images + local IndicF5 (no abort).
7. FAST: 2B single-stage 512x768 12 fps clip < ~15 s; RIFE output is 24 fps and smooth.
8. HERO: `/health` `comfyui_ready`; one LTX-2.5 clip plays, no OOM; time measured.
9. Reuse/caching: a repeated prompt is served from cache.
10. Quality: I2V-from-still visibly cleaner than T2V at the same step count; seed + negative reproducible.
11. Provisioning: baked image + volume → cold start ~1-3 min (vs ~15-40 min baseline).
12. `npm test` exits 0; instance stops after asset gen.

## Further Considerations
1. RTF 2-4 is NOT achievable on the 22B/4090 — the fast path (2B + 12 fps + RIFE + 2D + reuse) meets the
   budget. Confirm 2B fast-path quality vs 2.5 hero shots.
2. ComfyUI torch/CUDA vs base image — confirm in Phase 0; prefer baking into a custom image.
3. Video→image fallback trigger: implicit vs `--allow-local-fallback`.
4. HF gating for IndicF5 / LTX-2.5: ship `HF_TOKEN` vs mirrors.
5. Hero shots on a 4090 may need a 48 GB tier if OOM persists.
6. I2V quality depends on still quality → generate the FLUX/SSD-1B stills at good resolution.
7. Baked-image maintenance: rebuild on dependency changes; keep weights OUT of the image (volume instead).