# Audio/image-only Vast mode (FLUX.2-klein-4B)

Contract key: `VAST_AUDIO_IMAGE_ONLY=1`. Derived in `new_video.py` from
`--visual-mode image` + image provider `vast`, exported BEFORE any `VastManager()`.

## Chokepoints (change all of them together)
- `vast_manager.audio_image_only()` — the single reader for everything below.
- `SERVER_DEPENDENCY_OVERRIDES` REPLACES pins by name (`_package_name`). Never append:
  two `diffusers` lines = uv silently takes the last.
- `deps_marker_path()` → `.tube2dscript_deps_ready_aiimg`. Both onstart branches AND
  `repair_dependencies` must use it, or a warm normal-mode instance skips the install.
- `server_dependency_probe()` — must name the classes the handlers construct lazily
  (`Flux2KleinPipeline`, `Qwen3ForCausalLM`), not `LTXPipeline`.
- `prefetch_models()` — mode-first, `VAST_SERVER_PREFETCH` only in normal mode.
- `PREFETCH_IGNORE` drops Klein's 7.75GB monolithic duplicate (23.7GB → ~15.9GB).

## Server side (`vast_server.py`)
- `IMAGE_MODEL_ID` / `DEFAULT_GUIDANCE` read once at import. Klein wants guidance 1.0,
  schnell 0.0 — passing 0.0 to Klein washes out silently.
- `load_image_pipeline()` keeps the `_FLUX_PIPE` slot; one image model resident.
- THREE call sites: `handle_image_generate`, `handle_batch` (pins guidance inline — the
  one that 500s every scene), `handle_video_generate` (409, not 400).
- `/health` keeps `flux_loaded` for back-compat, adds `audio_image_only`, `image_model`,
  `image_ready`/`image_error`, `klein_ready`.

## Verification gates
Steps 0.1/0.2 of `plans/AUDIO_IMAGE_ONLY_IMPLEMENTATION_STEPS.md` are unrun — they need a
throwaway Vast instance. `diffusers>=0.37` + `transformers>=4.51` must import
`Flux2KleinPipeline` + `Qwen3ForCausalLM` before trusting any of this on real hardware.

## Gotcha
`harness: no terminal tool in this session`, so changes were verified by inspection only.
Run `python scripts/test_providers.py` on a machine with a terminal before renting.

## GPU picker (short rentals)
- `rank_offers()` sorts by fully-loaded 1h cost, NOT `dph_total`:
  `rate*1h + 25GB*inet_down_cost + storage_cost*disk_gb*1h`. Ties break on download ETA.
- A 100 Mbps host with $0.01/GB egress loses to a $0.06/hr dearer 900 Mbps free-egress
  host. That is the intended result — setup dominates a 1h rental.
- `prompt_pick_with_timeout()`: single keypress 1-5 (msvcrt on Windows, cbreak POSIX),
  `q` cancels, expiry auto-picks #1. Non-tty stdin returns the default immediately.
- Knobs: `VAST_RENTAL_HOURS`, `VAST_RENTAL_GB`, `VAST_PICK_TIMEOUT`;
  flags `--hours`, `--download-gb`, `--timeout` on `vast_manager.py choose`,
  `--choosegpu-timeout` on `new_video.py`.

