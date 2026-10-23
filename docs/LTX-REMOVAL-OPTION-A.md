# LTX-Video removal — Option A (surgical)

Status: **partially applied**. Read the file before each edit; line numbers have moved.

## Confirmed applied in `scripts/vast_server.py`

These returned success:

| Change | Where |
| --- | --- |
| `_LTX_PIPE` / `_LTX_MODEL_ID` globals deleted (comment left in their place) | globals block (~line 57) |
| `_release_other_models()` — `"ltx"` branch removed, map is now `flux`/`indic_parler`/`indicf5` | `_release_other_models()` |
| `/health` payload `ltx_loaded` → hardcoded `False` | `do_GET` health block |

## Unverified — check before assuming

The docstring edits returned ambiguous results, so verify all of these:

```bash
grep -n "LTX\|ltx" scripts/vast_server.py
```

Every line it prints is still to do. Expected remnants: the module docstring title, the
`/v1/videos/generate` endpoint line, `load_ltx_pipeline()`, and the handler body.

## Remaining in `scripts/vast_server.py`

1. **Docstring** — the module docstring still advertises LTX in the title and the endpoint
   list. Rewrite to:

   ```python
   """Vast.ai GPU Inference Server for FLUX images & Indic narration.

   Serves, on one rented GPU:
     1. Image generation (FLUX.2-klein-4B in audio/image mode, FLUX.1-schnell otherwise)
     2. AI4Bharat IndicF5 reference-conditioned Hindi TTS

   LTX-Video has been REMOVED. Its checkpoint pinned diffusers==0.32.2, which is mutually
   exclusive with the diffusers>=0.37 that Flux2KleinPipeline requires, so the two could
   never share one environment. /v1/videos/generate is retained as an explicit 501 so an
   old client gets a readable reason instead of a 404 that looks like a network fault.

   Endpoints:
     GET  /health                    -> liveness, device and per-pipeline readiness
     POST /v1/images/generate        -> FLUX image generation
     POST /v1/audio/synthesize       -> IndicF5 / Indic Parler-TTS audio (WAV + duration)
     POST /v1/audio/synthesize_batch -> many Indic audio segments in one round trip
     POST /v1/videos/generate        -> 501 GONE (LTX-Video removed)
   """
   ```

2. **Delete `load_ltx_pipeline()`** — the whole function. It imports `LTXPipeline` from
   diffusers, which no longer exists at the pinned `diffusers>=0.37`.

3. **Replace the video handler body** with an explicit 501 that keeps the route alive:

   ```python
   def _handle_video_generate(self, req: dict) -> None:
       """LTX-Video was removed; answer with an actionable 501 instead of a 404.

       A 404 looks like a routing or network fault to a client. The whole reason this
       route still exists is so an old caller is told exactly what replaced it and can
       be fixed with one flag change.
       """
       self._send_json(
           {
               "error": (
                   "LTX-Video generation has been removed from this server. "
                   "Use POST /v1/images/generate with --visual-mode image; FLUX.2-klein-4B "
                   "stills + IndicF5 narration is the supported pipeline."
               ),
               "code": "ltx_removed",
               "replacement": "/v1/images/generate",
           },
           status=501,
       )
   ```

   Keep the `elif self.path == "/v1/videos/generate":` dispatch line pointing at it, so the
   route is reachable and returns a body rather than an HTML 404.

4. **Delete every remaining LTX mention** in comments (dimension-validation notes,
   `LTXPipeline takes the conditioning frame as image`, the `ltx-video-` tempdir prefix,
   the `[Vast Server Error] LTX-Video generation failed` handler) — all of that lives inside
   the handler you are replacing in step 3, so it goes away with it.

## Remaining in `scripts/vast_manager.py`

Option A keeps the two-mode flag machinery deliberately; overrides become inert instead of
being deleted. That is what "low risk" bought you: `reuse_contract_problem()` and
`require_gpu()` keep working untouched.

1. **`DEFAULT_PREFETCH_MODELS`** — drop the LTX repo:

   ```python
   DEFAULT_PREFETCH_MODELS: tuple[str, ...] = (
       "ai4bharat/IndicF5",
       "black-forest-labs/FLUX.1-schnell",
   )
   ```

2. **`SERVER_DEPENDENCIES`** — promote the Klein pins into the base set and empty the
   override table. This is the single-pin-set change without deleting the plumbing:

   ```python
   "diffusers>=0.37,<0.42",     # was diffusers==0.32.2 (LTX)
   ...
   "transformers>=4.51,<6",     # was transformers==4.49.0 (LTX)
   ```

   ```python
   SERVER_DEPENDENCY_OVERRIDES: dict[str, str] = {}
   ```

   Leave `KLEIN_DIFFUSERS_PIN` / `KLEIN_TRANSFORMERS_PIN` defined (tests import them) but
   the override dict no longer needs to reference them.

3. **`SERVER_DEPENDENCY_PROBE`** — Klein becomes the only mode, so import it unconditionally
   and drop `FluxPipeline, LTXPipeline`:

   ```python
   "from diffusers import Flux2KleinPipeline; "
   "from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL; "
   "from transformers import Qwen3ForCausalLM"
   ```

4. **`server_dependency_probe()`** — collapse the `if not audio_image_only(): return
   SERVER_DEPENDENCY_PROBE` branch; both modes now return the same probe.

5. **`server_onstart_command()`** — `verify_klein` is now always on. Remove the
   `if audio_image_only() else ""` and keep the Klein import verification.

6. **`preflight_audio_image()`** — delete the two now-vacuous assertions:
   - `if any("LTX" in model for model in models)`
   - `if "LTXPipeline" in onstart`

   Keep the Klein ones. They still guard the real contract.

7. **Delete `generate_video()`** (the client method, ~line 2870) — nothing should be able to
   call a removed endpoint through a typed helper.

8. **`describe_mode()`** — the `"normal (FLUX.1 + LTX video)"` label is now wrong; normal
   mode means FLUX.1 only.

## Remaining in `scripts/new_video.py`

1. **Hard-fail video mode right after parsing** (near the other `parser.error` calls around
   line 1667). Keep `"video"` in `choices` so old scripts get this message instead of an
   argparse "invalid choice":

   ```python
   if args.visual_mode == "video":
       parser.error(
           "--visual-mode video (LTX-Video) has been removed. Use --visual-mode image: "
           "FLUX.2-klein-4B stills + IndicF5 narration is the supported pipeline."
       )
   ```

2. **Delete the `defer_video` block** (`if defer_video: ... from generate_videos import
   generate_videos_for_script`) — unreachable once video mode is rejected.

3. **`vast_prefetch`** — drop `if args.visual_mode == "video": vast_prefetch.append(args.video_model)`.
   The remaining branches are IndicF5/Parler + Klein.

4. **Delete `--video-model` / `--video-seconds` / `--video-steps`** and the
   `--video-gen` shortcut. Removing them is cleaner than leaving accepted-but-ignored flags;
   the `parser.error` above already tells anyone who used `--video-gen` what happened.

5. **`needs_vast` / `vast_required_early`** — the `args.visual_mode == "video"` disjuncts
   become dead. Leave or remove; they are now unreachable. Removing them makes the logic
   honest.

6. **`defer_video = args.visual_mode == "video"`** (~line 2347) — delete.

## Remaining elsewhere

| File | Change |
| --- | --- |
| `scripts/generate_videos.py` | Delete the file, OR reduce `generate_videos_for_script` to a function that raises the same "LTX removed" message (better: keeps the import in tests meaningful until they are updated) |
| `scripts/test_providers.py` | Delete `test_video_generation_mode()`; drop `import generate_videos` (line 18); invert the LTX prefetch assertions at ~735-741; add a test that `--visual-mode video` exits with the removal message |
| `.env` / `.env.example` | Remove `FLUX/LTX` wording (`.env:41`, `.env.example:64,76`) |
| `Dockerfile.vast` | Phase 4 comment — replace the "LTX-Video and FLUX are deliberately left out" paragraph with the Klein-only truth |
| `AUDIO_IMAGE_ONLY_PLAN.md`, `VAST_IMPLEMENTATION_PLAN.md` | Mark historical; do not edit |

## Post-refactor checks

```bash
# 1. Local contract
python scripts/vast_manager.py preflight --mode audio-image

# 2. Video mode must fail with the new message, not an argparse error
python scripts/new_video.py --visual-mode video --no-vast  # expect the removal error

# 3. No surviving LTX references
grep -rn "LTX" scripts/ Dockerfile.vast

# 4. Tests
python scripts/test_providers.py
```

## What Option A deliberately does NOT do

- It does **not** remove `VAST_AUDIO_IMAGE_ONLY`. That flag drives `container_env_spec()`,
  `reuse_contract_problem()`, `require_gpu()` and the `_aiimg` deps marker. Deleting it is
  Option B and requires replacement guards for each of those.
- It does **not** delete `SERVER_DEPENDENCY_OVERRIDES` — it empties it, so the code path
  stays but resolves to the base pins.
- It does **not** touch the Image-side (`Fluid`) naming: `FLUX1_MODEL_ID` /
  `IMAGE_MODEL_ID` still exist so normal mode keeps working until you drop it too.
