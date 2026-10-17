# Audio & Image-Only Mode — Step-by-Step Executable Plan

Companion to `AUDIO_IMAGE_ONLY_PLAN.md` (revision 2). That document is the *design*;
this one is the *execution checklist*, ordered so that every step leaves the repo working
and the two steps that can invalidate the rest come first.

Legend: `[gated]` = stop and verify on a throwaway Vast instance before continuing.
`[code]` = repository edit, testable without renting.

## Implementation status

| Phase | Status |
|---|---|
| 0 — preflight gates | **0.0 local done** (`vast_manager.py preflight`); 0.1-0.3 **not run** (need a throwaway rental) |
| 1 — `vast_manager.py` | done |
| 2 — `vast_server.py` | done |
| 3 — `new_video.py` | done |
| 4 — `generate_images.py` | done (mode-derived banner **and** mode-derived default model) |
| 5 — `test_providers.py` | done (`test_audio_image_only_mode`, `test_vast_preflight`) |
| 6 — end-to-end | **not run** (needs a paid rental) |

Only the hardware-gated checks remain. Run the local half first, then 0.1-0.3 on a
throwaway box:

```bat
.venv\Scripts\python.exe scripts\vast_manager.py preflight --mode audio-image
```

The command is offline and needs no API key. It prints the effective contract (marker,
prefetch set, dependency overrides, probe, disk default) and fails loudly if the mode
stops wiring together. It deliberately reports 0.1-0.3 as UNVERIFIED rather than
implying a green rental.


---

## Phase 0 — Preflight (gated, no repo edits)

### Step 0.0 — Local wiring check (no GPU, no network)

```bat
.venv\Scripts\python.exe scripts\vast_manager.py preflight --mode audio-image
```

Asserts the parts this repo controls: the container is told the mode, the port mapping
survives, Klein is prefetched and LTX is not, the diffusers/transformers pins are
*replaced* rather than appended, and the onstart marker is `deps_ready_aiimg` with a
`Flux2KleinPipeline` probe. Pass -> continue to 0.1. Fail -> fix the wiring before
spending anything; a rental cannot tell you which of these is wrong.

These next checks decide whether the rest of the plan is even real. Do them on a
throwaway instance (or a local Linux container with CUDA), not on a production rental.

### Step 0.1 — Verify the dependency pins import

```bash
# on the throwaway box, base image pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime
uv pip install --system "diffusers>=0.37,<0.42" "transformers>=4.51,<6" accelerate sentencepiece
python -c "from diffusers import Flux2KleinPipeline; from transformers import Qwen3ForCausalLM; print('ok')"
```

- Pass → record the exact resolved versions, continue.
- Fail → the plan's override becomes
  `diffusers @ git+https://github.com/huggingface/diffusers` and this must be re-verified
  before any code change. Do not proceed on the released-pin assumption.

### Step 0.2 — Verify IndicF5 survives the new transformers

```bash
uv pip install --system \
  "f5_tts @ git+https://github.com/AI4Bharat/IndicF5.git@13f7c4d627cc10111aea8fe9c0039462cacacdc7" \
  vocos soundfile "numpy==1.26.4"
python -c "from f5_tts.infer.utils_infer import infer_process; print('ok')"
```

Then synthesise one short Hindi sentence with the cloned reference clip and confirm
non-silent audio. This is the narration path for the whole pipeline.

### Step 0.3 — Measure Klein on 24 GB

One 1024×1024 image, `steps=4`, `guidance_scale=1.0`, `enable_model_cpu_offload()`.
Record wall time and peak VRAM. Expected working set ~15.8 GB BF16 weights.

**Gate:** if 0.1 or 0.2 fail, stop. Everything after this is speculative.

---

## Phase 1 — `scripts/vast_manager.py`

Four independent edits, all `[code]`, all testable with `test_providers.py`.

### Step 1.1 — Add the mode helper + dependency overrides

- Add `audio_image_only()` module-level function
  (`VAST_AUDIO_IMAGE_ONLY == "1"`).
- Keep `SERVER_DEPENDENCIES` **unchanged** (still pins `diffusers==0.32.2`,
  `transformers==4.49.0`) so the LTX path is untouched.
- Add `KLEIN_DIFFUSERS_PIN`, `KLEIN_TRANSFORMERS_PIN`, `SERVER_DEPENDENCY_OVERRIDES`.
- Rewrite `server_dependency_packages()` to **replace** matching entries by package name
  (never append — two `diffusers` lines in one `uv pip install` is a silent coin flip).

### Step 1.2 — Mode-specific dependency marker

- Add a `_deps_marker_suffix()` helper returning `"_aiimg"` or `""`.
- Interpolate `$HOME/.tube2dscript_deps_ready{suffix}` in **both** branches of
  `server_onstart_command` (the `VAST_IMAGE_HAS_DEPS` branch and the cold-start branch).
- Same substitution in `repair_dependencies()`.

Rationale: a warm-reused normal-mode instance must not short-circuit the Klein install.

### Step 1.3 — Mode-aware probe

- Add `server_dependency_probe()` returning the audio/image probe
  (`Flux2KleinPipeline` + `Qwen3ForCausalLM`, no `LTXPipeline`) or `SERVER_DEPENDENCY_PROBE`.
- `SERVER_DEPENDENCY_PROBE` stays as the default constant.
- `dependency_probe_command()` uses the mode-aware selector.
- `server_onstart_command`'s local `verify_imports` must use the same selector so the
  marker certifies exactly what the probe certifies.

### Step 1.4 — Prefetch set, ignore patterns, env, disk floor

- Add `AUDIO_IMAGE_PREFETCH_MODELS = ("ai4bharat/IndicF5", "black-forest-labs/FLUX.2-klein-4B")`.
- Add `PREFETCH_IGNORE` for Klein (`flux-2-klein-4b.safetensors`, `*.jpg`, `*.png`).
- Add `prefetch_models()` — mode default when the caller passes `None`, `VAST_SERVER_PREFETCH`
  otherwise. `server_onstart_command` calls it.
- `prefetch_cmd` gains `ignore_patterns=[...]` when the repo has an entry.
- `container_env_spec()` appends `-e VAST_AUDIO_IMAGE_ONLY=1`; fix the "~42 GB" docstring.
- Document the 60 GB audio/image disk floor default in the module docstring (the value
  itself lives in `new_video.py`).

### Step 1.5 — Update docstring knob list

Add `VAST_AUDIO_IMAGE_ONLY` to the tuning-knobs block at the top of the module.

---

## Phase 2 — `scripts/vast_server.py`

### Step 2.1 — Mode constants

At import time, read `AUDIO_IMAGE_ONLY` **once**. Define `KLEIN_MODEL_ID`,
`FLUX1_MODEL_ID`, `IMAGE_MODEL_ID`, `DEFAULT_GUIDANCE` (1.0 Klein / 0.0 schnell).

### Step 2.2 — `image_pipeline_import_error()`

Mirror `indicf5_import_error()`: lazy probe of `Flux2KleinPipeline` +
`Qwen3ForCausalLM`, latch success, never cache failure, return the reason string.

### Step 2.3 — `load_image_pipeline()`

Replace `load_flux_pipeline()` with `load_image_pipeline(model_id=None)`:
same `_FLUX_PIPE` slot, same `_release_other_models("flux")`, class chosen by mode,
`enable_model_cpu_offload()` mandatory in audio/image mode.

Keep a thin `load_flux_pipeline` alias so nothing external breaks.

### Step 2.4 — Three call sites

| Handler | Change |
|---|---|
| `handle_image_generate` | `load_image_pipeline()`; `guidance` default → `DEFAULT_GUIDANCE` |
| `handle_video_generate` | return **409** early when `AUDIO_IMAGE_ONLY` |
| `handle_batch` | `load_image_pipeline()`; `guidance_scale=DEFAULT_GUIDANCE` |

`handle_batch` is the one that would 500 on every scene if missed.

### Step 2.5 — `/health`

Add `audio_image_only`, `image_model`, `klein_ready`. Keep `flux_loaded` (compat) and
force `ltx_loaded: False` in this mode.

### Step 2.6 — `handle_audio_synthesize_batch` `started` bug

Verify `started` resolves before use; if not, hoist `started = time.time()` above the loop.

---

## Phase 3 — `scripts/new_video.py`

### Step 3.1 — Derive the mode before any `VastManager()`

Immediately after `detected_image_provider` is computed (~line 2060):

```python
audio_image_only_mode = args.visual_mode == "image" and detected_image_provider == "vast"
```

Set/clear `os.environ["VAST_AUDIO_IMAGE_ONLY"]` there, **before** `--choosegpu` runs
(`--choosegpu` constructs a `VastManager`).

### Step 3.2 — Prefetch follow the mode

`vast_prefetch` gains `black-forest-labs/FLUX.2-klein-4B` in audio/image mode and
never the LTX model.

### Step 3.3 — Disk floor

`vast_disk_gb` default becomes `60` in audio/image mode, `100` otherwise.
`--choosegpu`'s `_choose_disk` uses the same expression.

### Step 3.4 — Contradictory-mode guard

`parser.error(...)` when `VAST_AUDIO_IMAGE_ONLY=1` and `--visual-mode video`.

---

## Phase 4 — `scripts/generate_images.py`

### Step 4.1 — Mode-derived banner

Replace the hardcoded `"Vast.ai Cloud RTX 3090 GPU (FLUX.1-schnell, ...)"` with a
mode-derived label (`FLUX.2-klein-4B` when `VAST_AUDIO_IMAGE_ONLY=1`).

No other change: steps stay 4, guidance comes from the server default, seeds already
deterministic.

---

## Phase 5 — `scripts/test_providers.py`

### Step 5.1 — Add `test_audio_image_only_mode()`

Assert, inside `VAST_AUDIO_IMAGE_ONLY=1`:
1. `container_env_spec()` has `-p 8000:8000` and `-e VAST_AUDIO_IMAGE_ONLY=1`.
2. `prefetch_models()` has Klein and no LTX.
3. `server_dependency_packages()` has `diffusers>=0.37`, no `diffusers==0.32.2`,
   `transformers>=4.51`.
4. `server_onstart_command()` mentions `deps_ready_aiimg` + `Flux2KleinPipeline`,
   not `LTXPipeline`.

Outside the `try/finally`, assert defaults unchanged (`diffusers==0.32.2`,
no `deps_ready_aiimg`).

### Step 5.2 — Register it in `__main__` and run the suite.

---

## Phase 6 — End-to-end (gated, one paid rental)

```bat
.venv\Scripts\python.exe scripts\new_video.py ^
  --topic "aadha din aadhi raat" --minutes 10 ^
  --visual-mode image --vast --imageprovider vast ^
  --audio-model indicf5 --no-render --scenes 1
```

Confirm:
- no LTX entry in the prefetch list / onstart log;
- `git` clone + deps install writes `deps_ready_aiimg`;
- `server.log` shows `Flux2KleinPipeline` loaded;
- `/health` reports `audio_image_only: true`, `image_model: black-forest-labs/FLUX.2-klein-4B`;
- narration audio cache populated;
- render left to local.

Then render locally from `script.json` with Remotion, unchanged.

---

## Rollback

Every edit is additive and keyed on `VAST_AUDIO_IMAGE_ONLY`. Unsetting the variable
returns the process to the exact current behaviour: same pins, same prefetch, same
marker, same probe, FLUX.1-schnell, LTX allowed.
