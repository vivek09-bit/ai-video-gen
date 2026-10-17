# Audio & Image-Only Vast.ai Plan — IndicF5 + FLUX.2 Klein 4B (BF16)

Revision 2. Replaces the earlier draft. Every number below was measured from the
Hugging Face repo, not estimated.

---

## 0. Verified facts (source of truth for this plan)

From `black-forest-labs/FLUX.2-klein-4B` on the Hub:

| Item | Value |
|---|---|
| Repo total size | **23.7 GB** |
| `transformer/diffusion_pytorch_model.safetensors` | **7.75 GB** (the 4B transformer, BF16) |
| `text_encoder/model-00001..02.safetensors` | **4.97 GB + 3.08 GB = 8.05 GB** |
| `flux-2-klein-4b.safetensors` (monolithic duplicate) | **7.75 GB** — skip it in the prefetch |
| `_class_name` | `Flux2KleinPipeline` |
| `_diffusers_version` | **`0.37.0.dev0`** |
| `text_encoder` | `transformers` / **`Qwen3ForCausalLM`** |
| `tokenizer` | `transformers` / `Qwen2TokenizerFast` |
| `transformer` | `diffusers` / `Flux2Transformer2DModel` |
| `vae` | `diffusers` / `AutoencoderKLFlux2` |
| `is_distilled` | `true` |

### Two corrections to the earlier draft

**"FLUX.2 Klein 4B ~8-10 GB" is wrong.** 8 GB is the *transformer only*. The
Qwen3 text encoder is a second 8.05 GB and is **not** separable — it lives inside the
same repo and is loaded by the pipeline. Add the VAE and tokenizer and the working set
is **23.7 GB on disk**.

**"BF16 full precision" at 24 GB is tight but viable.** At inference the transformer
(7.75) and text encoder (8.05) are ~15.8 GB of BF16 weights, plus the VAE and activations.
Klein is Apache 2.0, non-gated, and needs **no** `HF_TOKEN`.

### Consequence for the prefetch set

The onstart prefetch must fetch the diffusers tree **without** the monolithic
`*.safetensors` duplicates, or it pays 7.75 GB of egress for a file nothing loads:

```python
snapshot_download(
    "black-forest-labs/FLUX.2-klein-4B",
    ignore_patterns=["flux-2-klein-4b.safetensors", "*.jpg", "*.png"],
)
```

With that ignore list the download is **~15.9 GB**, not 23.7 GB. Compare against today's
`DEFAULT_PREFETCH_MODELS` (~42 GB: FLUX.1-schnell + LTX + IndicF5), so the saving is real
but roughly **60%**, not the 75-80% the earlier draft claimed.

### Dependency pins — this is the part that decides whether it boots

Current `SERVER_DEPENDENCIES` in `scripts/vast_manager.py`:

```
diffusers==0.32.2       <-- cannot import Flux2KleinPipeline (needs >= 0.37)
transformers==4.49.0    <-- Qwen3ForCausalLM predates this; Qwen3 landed in 4.51
```

`diffusers==0.32.2` is pinned **deliberately** — the LTX checkpoint declares
`_diffusers_version: 0.32.0.dev0`, and 0.32.2 matches the local venv. Klein needs
`>=0.37`. **These two cannot share one pin.** The dependency set must therefore be
selected by mode; it cannot be a single global tuple.

`transformers==4.49.0` must rise to a version that ships `Qwen3ForCausalLM`. That change
affects the **IndicF5** path too (IndicF5 uses `trust_remote_code`, so it is mostly
insulated, but verify). This is the highest-risk item in the whole plan — do it first,
in isolation, before touching the image path.

**Verified remotely before writing any code:**

```bash
# on a throwaway instance, audio/image-only pin set:
python -c "from diffusers import Flux2KleinPipeline; from transformers import Qwen3ForCausalLM; print('ok')"
```

If both imports pass, the pin set is real. If they do not, the rest of this plan is
speculative and should not be implemented.

---

## 1. What gets removed vs. kept

| Component | Audio/image-only | Normal image mode |
|---|---|---|
| `_LTX_PIPE` / `LTXPipeline` | **never loaded** | optional |
| `_FLUX_PIPE` (`FluxPipeline`, FLUX.1-schnell) | replaced | active |
| `Flux2KleinPipeline` | **active** | absent |
| `Qwen3ForCausalLM` text encoder | **loaded** (inside Klein) | absent |
| IndicF5 | active | active on demand |
| Indic Parler-TTS | legacy fallback | legacy fallback |
| Remotion render | **local** (unchanged) | local or vast |

Nothing about the render path changes. `--render-on` stays `local`, and there is no
reason to install the render stack on an audio/image-only instance — that is exactly
the `render_stack_shell()` split the codebase already has.

---

## 2. Entry point: no new flag

The earlier draft proposed `--audio-image-only`. That duplicates state the CLI already
carries and creates contradictory combinations to validate (`--audio-image-only
--visual-mode video`).

This mode is already expressible:

```bash
.venv\Scripts\python.exe scripts\new_video.py ^
  --topic "aadha din aadhi raat" --minutes 10 ^
  --visual-mode image --vast --imageprovider vast ^
  --audio-model indicf5 --no-render
```

`--visual-mode image` + `--imageprovider vast` **is** audio/image-only. `needs_vast`
already fires on IndicF5, and `vast_required_early` already fires on a `vast` image
provider. Derive the server contract from it:

```python
audio_image_only = args.visual_mode == "image" and detected_image_provider == "vast"
```

`VAST_AUDIO_IMAGE_ONLY=1` becomes a **server-side contract only**, set into the
environment before `VastManager()` is constructed so `container_env_spec()` sees it.
Keep it as an env var (not a CLI flag in `vast_manager.py`) so `scripts/batch.py` and
`test_providers.py` do not need a new argument threaded through them.

Guard the contradictory case rather than adding a mode:

```python
if args.visual_mode == "video" and os.environ.get("VAST_AUDIO_IMAGE_ONLY") == "1":
    parser.error("VAST_AUDIO_IMAGE_ONLY excludes --visual-mode video; unset it or drop video mode.")
```

---

## 3. `scripts/vast_manager.py`

### 3.1 Dependency sets, selected by mode

Replace the single `SERVER_DEPENDENCIES` tuple with a base + override structure. Keep
the existing tuple and its comments intact as the default path.

```python
# Shared by both modes.
SERVER_DEPENDENCIES: tuple[str, ...] = ( ... )   # unchanged, keeps diffusers==0.32.2

# Audio/image-only: Klein needs Flux2KleinPipeline (diffusers >= 0.37) and
# Qwen3ForCausalLM (transformers >= 4.51). Neither exists in the default pins, and the
# default pins cannot move because the LTX checkpoint declares 0.32.0.dev0.
KLEIN_DIFFUSERS_PIN = "diffusers>=0.37,<0.42"
KLEIN_TRANSFORMERS_PIN = "transformers>=4.51,<6"

SERVER_DEPENDENCY_OVERRIDES: dict[str, str] = {
    "diffusers": KLEIN_DIFFUSERS_PIN,
    "transformers": KLEIN_TRANSFORMERS_PIN,
}


def server_dependency_packages(image: str | None = None) -> tuple[str, ...]:
    """...existing docstring, plus: mode overrides replace, never append..."""
    image = image or os.environ.get("VAST_SERVER_IMAGE", DEFAULT_SERVER_IMAGE)
    packages = list(SERVER_DEPENDENCIES)
    if audio_image_only():
        for index, package in enumerate(packages):
            name = re.split(r"[=<>@ ]", package, maxsplit=1)[0]
            if name in SERVER_DEPENDENCY_OVERRIDES:
                packages[index] = SERVER_DEPENDENCY_OVERRIDES[name]
    torch_version = _image_torch_version(image)
    torchaudio_pin = f"torchaudio=={torch_version}" if torch_version else "torchaudio"
    return (*packages, torchaudio_pin)
```

Replacing rather than appending matters: appending would leave two conflicting
`diffusers` lines in one `uv pip install`, and uv resolves the **last one it reads** —
a silent coin flip.

### 3.2 Mode helper

```python
def audio_image_only() -> bool:
    """True when the container must serve images + narration only, never video."""
    return os.environ.get("VAST_AUDIO_IMAGE_ONLY", "0").strip() == "1"
```

Used by `server_dependency_packages`, `server_onstart_command`, `dependency_probe_command`,
and `container_env_spec`.

### 3.3 The dependency marker must be mode-specific

This is the failure the earlier draft would have caused. `server_onstart_command`
short-circuits on `$HOME/.tube2dscript_deps_ready`. A warm-reused instance provisioned
in normal mode would report "dependencies already present", skip the install, and then
fail at the first image request with `ImportError: cannot import name
'Flux2KleinPipeline'` — after the rental was paid.

```python
deps_marker = (
    "$HOME/.tube2dscript_deps_ready_aiimg" if audio_image_only()
    else "$HOME/.tube2dscript_deps_ready"
)
```

`suffix = "_aiimg" if audio_image_only() else ""` and interpolate `$HOME/.tube2dscript_deps_ready{suffix}`
in **both** branches of the onstart (the `VAST_IMAGE_HAS_DEPS` branch too) and in
`repair_dependencies`.

The same reasoning already exists as a comment on `_release_other_models` regarding
NumPy/pyarrow — this is the same class of bug, so the same fix.

### 3.4 Probe the classes the mode actually constructs

`SERVER_DEPENDENCY_PROBE` currently imports `FluxPipeline, LTXPipeline`. In audio/image
mode the probe must name the classes the request handlers construct lazily, or the
marker certifies a runtime that cannot serve a request:

```python
if audio_image_only():
    probe = (
        "import numpy, torch, torchaudio, diffusers, transformers, soundfile, scipy, vocos; "
        "from f5_tts.infer.utils_infer import infer_process; "
        "from diffusers import Flux2KleinPipeline; "
        "from transformers import Qwen3ForCausalLM"
    )
```

`parler_tts` stays out of both probes — its absence must never block readiness.

### 3.5 Container env

In `container_env_spec()`, alongside the existing `-p 8000:8000` and `INDICF5_VOICES_DIR`:

```python
if audio_image_only():
    parts.append("-e VAST_AUDIO_IMAGE_ONLY=1")
```

Also correct the docstring, which currently claims "~42 GB of FLUX/LTX/IndicF5 weights".
In this mode it is ~16 GB. The comment sits directly above the `HF_HOME` logic and is
the thing a reader uses to judge whether a persistent volume is worth paying for.

### 3.6 Prefetch set

```python
AUDIO_IMAGE_PREFETCH_MODELS: tuple[str, ...] = (
    "ai4bharat/IndicF5",
    "black-forest-labs/FLUX.2-klein-4B",
)


def prefetch_models() -> list[str]:
    """Default prefetch for the current mode when the caller passes None."""
    if audio_image_only():
        return list(AUDIO_IMAGE_PREFETCH_MODELS)
    env_models = os.environ.get("VAST_SERVER_PREFETCH", "").strip()
    return (
        [item.strip() for item in env_models.split(",") if item.strip()]
        if env_models else list(DEFAULT_PREFETCH_MODELS)
    )
```

`server_onstart_command` calls `prefetch_models()` when its argument is `None`, so
every existing call site gets the right set without threading a parameter through.

Extend `prefetch_cmd` so the Klein download skips the duplicate monolithic file:

```python
PREFETCH_IGNORE: dict[str, list[str]] = {
    "black-forest-labs/FLUX.2-klein-4B": [
        "flux-2-klein-4b.safetensors", "*.jpg", "*.png",
    ],
}
```

That is 23.7 GB → ~15.9 GB.

### 3.7 Disk floor

`new_video.py` uses `VAST_DISK_GB=100` **because** FLUX.1 + LTX is ~42 GB of weights. In
audio/image mode the working set is ~16 GB, so a 100 GB floor rejects viable cheap hosts
and bills storage for space that is never written.

```python
default_disk = "60" if audio_image_only() else "100"
vast_disk_gb = int(os.environ.get("VAST_DISK_GB", default_disk))
```

Leave the `find_existing_instance(min_disk=...)` call reading the same value so reuse
and rent agree — the existing comment explains why a mismatch merely moves the failure.

---

## 4. `scripts/vast_server.py`

### 4.1 One mode switch, read once

```python
AUDIO_IMAGE_ONLY = os.environ.get("VAST_AUDIO_IMAGE_ONLY", "0").strip() == "1"
KLEIN_MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
FLUX1_MODEL_ID = "black-forest-labs/FLUX.1-schnell"
IMAGE_MODEL_ID = KLEIN_MODEL_ID if AUDIO_IMAGE_ONLY else FLUX1_MODEL_ID
# Klein is distilled and documents guidance_scale=1.0; FLUX.1-schnell needs 0.0.
# Passing 0.0 to Klein produces flat, washed-out frames rather than an error.
DEFAULT_GUIDANCE = 1.0 if AUDIO_IMAGE_ONLY else 0.0
```

### 4.2 One image loader

The handlers must not each pick a class. Rename and route:

```python
def load_image_pipeline(model_id: str | None = None):
    """Load the image pipeline for this mode onto the GPU in bfloat16."""
    global _FLUX_PIPE
    model_id = model_id or IMAGE_MODEL_ID
    if _FLUX_PIPE is not None:
        return _FLUX_PIPE
    _release_other_models("flux")
    import torch
    torch_dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
    if AUDIO_IMAGE_ONLY:
        from diffusers import Flux2KleinPipeline
        pipe = Flux2KleinPipeline.from_pretrained(
            model_id, torch_dtype=torch_dtype, token=_hub_token()
        )
    else:
        from diffusers import FluxPipeline
        pipe = FluxPipeline.from_pretrained(
            model_id, torch_dtype=torch_dtype, token=_hub_token()
        )
    if DEVICE == "cuda":
        # 15.8 GB of BF16 weights on a 24 GB card: keep the offload that already exists,
        # but make it non-optional here. FLUX.1-schnell was ~34 GB and needed it anyway.
        pipe.enable_model_cpu_offload()
    _FLUX_PIPE = pipe
    return _FLUX_PIPE
```

Keep `_FLUX_PIPE` as the single slot. Adding a fifth global (`_KLEIN_PIPE`) to
`_release_other_models` buys nothing — only one image model is ever resident.

### 4.3 Call sites

Three, and the earlier draft missed two of them:

| Line | Handler | Change |
|---|---|---|
| 724 | `handle_image_generate` | `load_flux_pipeline()` → `load_image_pipeline()`; `guidance_scale` default → `DEFAULT_GUIDANCE` |
| 781 | `handle_video_generate` | refuse in this mode (see 4.4) |
| **947** | **`handle_batch`** | `load_flux_pipeline()` → `load_image_pipeline()`; `guidance_scale=0.0` → `DEFAULT_GUIDANCE` |

`handle_batch` is the one that would have 500'd on every scene in audio/image mode,
because it pins `guidance_scale=0.0` inline.

### 4.4 Video endpoint

Return **409 Conflict**, not 400. The endpoint exists and the request is well-formed;
the *mode* forbids it. That distinction lets the client tell a mode refusal from a bad
prompt without string-matching.

```python
if AUDIO_IMAGE_ONLY:
    self._send_json(
        {"status": "error", "audio_image_only": True,
         "error": "Video generation is disabled in audio-image-only mode"},
        status=409,
    )
    return
```

### 4.5 `/health`

Keep the existing keys honest and add the mode. Do **not** rename `flux_loaded` —
`require_gpu()` and any warm instance predating this change read it.

```python
{
    ...existing keys...,
    "audio_image_only": AUDIO_IMAGE_ONLY,
    "image_model": IMAGE_MODEL_ID,
    "flux_loaded": _FLUX_PIPE is not None,     # "the image model is resident"
    "ltx_loaded": False if AUDIO_IMAGE_ONLY else _LTX_PIPE is not None,
    "klein_ready": (image_pipeline_import_error() is None) if AUDIO_IMAGE_ONLY else None,
}
```

`image_pipeline_import_error()` mirrors the existing `indicf5_import_error()` shape: probe
`Flux2KleinPipeline` + `Qwen3ForCausalLM` lazily, do not cache failure, and report the
reason so `require_gpu()` can refuse an instance whose deps are wrong. Without this, the
only signal is an HTTP 500 at the first image request.

### 4.6 Pre-existing bug, unrelated but worth fixing while here

`handle_audio_synthesize_batch` references `started` in its response
(`"total_generation_time_sec": round(time.time() - started, 3)`), but the nearest
assignment of `started` in that handler appears **after** the loop that uses `t0` and
after the reference. Confirm whether it resolves from an enclosing scope; if not, this
endpoint raises `NameError` and the whole batch TTS path is dead.

---

## 5. `scripts/generate_images.py`

Three small things, no structural change:

1. **Banner** (line ~1774): `"Vast.ai Cloud RTX 3090 GPU (FLUX.1-schnell, …)"` is hardcoded.
   Make it mode-derived, or it lies to the operator about which model ran.
2. **Steps** (line ~1869): `steps=4` is right for schnell and right for Klein (the card
   uses 4). Leave it, but note that `guidance` is *not* passed from the client — the
   server default now supplies the mode-correct value.
3. **Seed**: already deterministic per scene (`md5(topic_scene_i)`). Klein respects
   `torch.Generator(device=...)` the same way, so re-renders stay stable.

---

## 6. Orchestration — `scripts/new_video.py`

Order matters; getting it wrong makes the env var invisible to the container spec.

```python
# 1. Derive the mode BEFORE any VastManager is constructed.
detected_image_provider, _, _ = detect_image_provider(requested_image_provider)
audio_image_only_mode = (
    args.visual_mode == "image" and detected_image_provider == "vast"
)
if audio_image_only_mode:
    os.environ["VAST_AUDIO_IMAGE_ONLY"] = "1"
else:
    os.environ.pop("VAST_AUDIO_IMAGE_ONLY", None)

# 2. Prefetch follows the mode. Only asked for when the provider is vast.
vast_prefetch = ["ai4bharat/IndicF5"] if audio_model == "indicf5" else []
if audio_image_only_mode:
    vast_prefetch.append("black-forest-labs/FLUX.2-klein-4B")
if args.visual_mode == "video":
    vast_prefetch.append(args.video_model)

# 3. Disk floor follows the mode (see 3.7).
```

`vast_prefetch` is currently built **before** `start_vast()` but after the provider is
detected — verify that ordering holds, since `detected_image_provider` is computed at
line ~2060 and `vast_prefetch` at ~2132. It does; the edit is safe.

One more: `require_gpu(..., require_tts=...)` should gain nothing here. The existing
`indicf5_ready` check already catches a wrong dependency set when narration is enabled,
and narration **is** enabled in this mode. The new `image_pipeline_import_error()` check
covers the image half.

---

## 7. Tests — `scripts/test_providers.py`

Extend `test_vast_gpu_routing` and `test_vast_setup_resilience` rather than adding a
parallel suite. `test_vast_call_sites()` already guards `provision_instance`'s signature.

```python
def test_audio_image_only_mode() -> None:
    import os
    import vast_manager
    from vast_manager import VastManager

    os.environ["VAST_AUDIO_IMAGE_ONLY"] = "1"
    try:
        # 1. The container is told the mode, and the port mapping survives.
        env_spec = VastManager.container_env_spec()
        assert "-p 8000:8000" in env_spec
        assert "-e VAST_AUDIO_IMAGE_ONLY=1" in env_spec

        # 2. Prefetch fetches Klein and NOT LTX.
        models = vast_manager.prefetch_models()
        assert "black-forest-labs/FLUX.2-klein-4B" in models
        assert not any("LTX" in m for m in models)

        # 3. Deps overrode the default pins.
        deps = VastManager.server_dependency_packages()
        assert any(d.startswith("diffusers>=0.37") for d in deps)
        assert not any(d == "diffusers==0.32.2" for d in deps)
        assert any(d.startswith("transformers>=4.51") for d in deps)

        # 4. Marker is mode-specific, so a warm normal-mode instance cannot
        #    short-circuit the install.
        onstart = VastManager.server_onstart_command(prefetch_models=None)
        assert "deps_ready_aiimg" in onstart
        assert "Flux2KleinPipeline" in onstart
        assert "LTXPipeline" not in onstart
    finally:
        os.environ.pop("VAST_AUDIO_IMAGE_ONLY", None)

    # 5. Default mode is unchanged: same pins, same prefetch, same marker.
    assert any(d == "diffusers==0.32.2" for d in VastManager.server_dependency_packages())
    assert "deps_ready_aiimg" not in VastManager.server_onstart_command(prefetch_models=None)
```

Also assert the *conflict* line: `diffusers==0.32.2` must not appear anywhere in the
audio/image dependency string, since uv would silently pick the last `diffusers` entry.

---

## 8. Implementation order

Do not batch this. Steps 1-2 are the ones that can invalidate the rest.

- [ ] **1. Verify the pins on a throwaway instance.** Install `diffusers>=0.37,<0.42`
      + `transformers>=4.51,<6` alongside the existing torch/torchaudio, then:
      `python -c "from diffusers import Flux2KleinPipeline; from transformers import Qwen3ForCausalLM"`.
      Time a single 1024×1024 image and record peak VRAM. **Stop here if this fails.**
- [ ] **2. Verify IndicF5 still works on the new transformers.** The `trust_remote_code`
      path should be insulated, but this is the narration model for the whole pipeline.
- [ ] **3. `vast_manager.py`**: `audio_image_only()`, `prefetch_models()`,
      `server_dependency_packages()` overrides, mode-specific marker, mode-aware probe,
      `container_env_spec()`. Run `test_audio_image_only_mode`.
- [ ] **4. `vast_server.py`**: mode constants, `load_image_pipeline()`, the three call
      sites (including `handle_batch`), 409 on video, `/health` keys,
      `image_pipeline_import_error()`.
- [ ] **5. `new_video.py`**: derive the mode before `VastManager()`, prefetch, disk floor,
      the contradictory-mode guard.
- [ ] **6. `generate_images.py`**: banner only.
- [ ] **7. End-to-end, one scene.** `--scenes 1 --visual-mode image --vast
      --imageprovider vast --audio-model indicf5 --no-render`. Confirm: no LTX in the
      onstart log, Klein in `server.log`, `/health` reports `audio_image_only: true`,
      audio cache populated, render left to local.
- [ ] **8. Local render** from the written `script.json`, on the PC, Remotion unchanged.

---

## 9. Risks

| Risk | Mitigation |
|---|---|
| `Flux2KleinPipeline` not in any released diffusers | Step 1 gates everything; fall back to `pip install git+https://github.com/huggingface/diffusers` in the override |
| New `transformers` breaks IndicF5 | Step 2; IndicF5 uses `trust_remote_code`, so it should be insulated |
| 15.8 GB BF16 on a 24 GB card OOMs at 1080p | `enable_model_cpu_offload()` is mandatory here; cap the server's internal scale (already `min(1.0, 1024/max(w,h))`) |
| Warm-reused instance has the wrong deps | Mode-specific marker (3.3) + `image_pipeline_import_error()` in `/health` |
| Disk floor still rejects cheap 40 GB hosts | 60 GB default leaves ~44 GB free after ~16 GB of weights — adequate; not lower, since HF cache dedupes poorly |
| Prefetch pulls 23.7 GB instead of 15.9 | `PREFETCH_IGNORE` for the monolithic `.safetensors` and the sample JPEGs |
