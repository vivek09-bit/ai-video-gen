#!/usr/bin/env python3
"""Vast.ai GPU Inference Server for FLUX.1-schnell, LTX-Video & IndicF5 TTS.

Runs on one remote 24GB GPU instance, hosting one requested visual pipeline:
  1. FLUX.1-schnell image generation (RTX 3090 workflow)
  2. LTX-Video 2B distilled video generation (RTX 4090 workflow)
  2. AI4Bharat IndicF5 zero-shot flow-matching Hindi TTS

Endpoints:
  GET  /health              -> System, GPU VRAM & model status
  POST /v1/images/generate  -> FLUX.1 image generation
  POST /v1/videos/generate  -> LTX-Video MP4 generation
  POST /v1/audio/synthesize -> IndicF5 audio generation (WAV + duration)
  POST /v1/audio/synthesize_batch -> many IndicF5 segments in one round trip
  POST /v1/batch            -> images + audio for a scene list in one round trip
  POST /v1/shutdown         -> Gracefully shutdown server / instance
"""

from __future__ import annotations

import base64
import gc
import io
import json
import os
import sys
import tempfile
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PORT = int(os.environ.get("PORT", 8000))
DEVICE = "cuda" if os.environ.get("FORCE_CPU", "0") != "1" else "cpu"

_FLUX_PIPE = None
_LTX_PIPE = None
_LTX_MODEL_ID: str | None = None
_INDIC_PARLER_ENGINE = None

# Latched once the IndicF5 import path is known to work. A failure is deliberately NOT
# cached: an instance can be repaired over SSH while this process keeps running, and the
# next /health should reflect that without needing a restart.
_TTS_IMPORT_OK = False


def _audio_is_bad(wav) -> bool:
    """True when a waveform is empty, non-finite, silent or pure DC.

    IndicF5 on fp16 can return NaN; libsndfile then encodes it as a constant -32768
    (0x8000) DC WAV that is structurally valid but carries no speech. Rejecting it here
    turns a corrupt generation into an explicit error instead of a silent success.
    """
    import numpy as np

    x = np.asarray(wav, dtype=np.float64).reshape(-1)
    if x.size == 0 or not np.all(np.isfinite(x)):
        return True
    if np.max(np.abs(x)) < 1e-4:  # silence
        return True
    if np.std(x) < 1e-5:  # constant / pure DC
        return True
    return False


def tts_import_error() -> str | None:
    """Return why the Indic Parler-TTS import path is unusable, or None when it works.

    The synthesis handler imports ``parler_tts`` lazily, inside the request, so a broken
    wheel anywhere in that chain stays invisible until the first request, which the client
    then sees only as an opaque HTTP 500 after it has already paid to rent this GPU.
    Probing the same import up front turns that into a boot-time fact it can act on.
    """
    global _TTS_IMPORT_OK
    if _TTS_IMPORT_OK:
        return None
    try:
        import numpy  # noqa: F401 - importing it is the point, not using it
        import transformers  # noqa: F401
        from parler_tts import ParlerTTSForConditionalGeneration  # noqa: F401
    except Exception as err:  # noqa: BLE001 - any failure here makes synthesis impossible
        return f"{type(err).__name__}: {err}"
    _TTS_IMPORT_OK = True
    return None


def _hub_token() -> str | None:
    """Read an optional Hugging Face token from the server environment."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY") or None


def _release_other_models(active: str) -> None:
    """Keep only one large inference model resident on the 24GB GPU."""
    global _FLUX_PIPE, _LTX_PIPE, _LTX_MODEL_ID, _INDIC_PARLER_ENGINE
    for name in ("flux", "ltx", "indic_parler"):
        if name == active:
            continue
        value = {
            "flux": _FLUX_PIPE,
            "ltx": _LTX_PIPE,
            "indic_parler": _INDIC_PARLER_ENGINE,
        }[name]
        if value is None:
            continue
        try:
            if hasattr(value, "to"):
                value.to("cpu")
        except Exception as err:
            print(f"[Vast Server Warning] Could not move {name} model to CPU before release: {err}", file=sys.stderr)
        if name == "flux":
            _FLUX_PIPE = None
        elif name == "ltx":
            _LTX_PIPE = None
            _LTX_MODEL_ID = None
        else:
            _INDIC_PARLER_ENGINE = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _pace_word(speed: float) -> str:
    """Map a numeric speed multiplier onto the pace phrase Parler-TTS understands."""
    if speed <= 0.92:
        return "a slow, deliberate pace"
    if speed >= 1.12:
        return "a fast, urgent pace"
    return "a moderate pace"


def _resolve_indic_description(voice: str, speed: float = 1.0) -> str:
    """Map a voice id/alias onto the Parler-TTS speaker description.

    Parler-TTS is steered by a natural-language description rather than a reference clip,
    so this replaces the old reference-wav lookup. The shared catalog in ``tts.py`` is the
    source of truth; the literal below is only a fallback for a standalone deployment.
    """
    try:
        from tts import INDIC_PARLER_VOICES

        lowered = (voice or "").lower()
        base: str | None = None
        if lowered in INDIC_PARLER_VOICES:
            base = INDIC_PARLER_VOICES[lowered]["description"]
        else:
            for vk, vi in INDIC_PARLER_VOICES.items():
                if lowered == vk.lower() or lowered in [a.lower() for a in vi.get("aliases", [])]:
                    base = vi["description"]
                    break
        if base:
            return base.replace("a moderate pace", _pace_word(speed))
    except Exception:
        pass
    return (
        "A male speaker with a deep, resonant and authoritative voice speaks with a "
        f"{_pace_word(speed)} in a very clear, close-sounding recording with no background noise."
    )


def get_vram_info() -> dict[str, Any]:
    """Return current GPU VRAM utilization if CUDA is available."""
    try:
        import torch
        if torch.cuda.is_available():
            device_idx = torch.cuda.current_device()
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_idx)
            allocated_bytes = torch.cuda.memory_allocated(device_idx)
            return {
                "available": True,
                "device_name": torch.cuda.get_device_name(device_idx),
                "total_gb": round(total_bytes / (1024**3), 2),
                "free_gb": round(free_bytes / (1024**3), 2),
                "allocated_gb": round(allocated_bytes / (1024**3), 2),
            }
    except Exception:
        pass
    return {"available": False}


def load_flux_pipeline(model_id: str = "black-forest-labs/FLUX.1-schnell"):
    """Load FLUX pipeline onto GPU in bfloat16 / fp8."""
    global _FLUX_PIPE
    if _FLUX_PIPE is not None:
        return _FLUX_PIPE

    _release_other_models("flux")
    import torch
    from diffusers import FluxPipeline

    print(f"[Vast Server] Loading FLUX model: {model_id} on {DEVICE}...")
    torch_dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32

    try:
        pipe = FluxPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            token=_hub_token(),
        )
        if DEVICE == "cuda":
            # Increase timeout for first load
            pipe.enable_model_cpu_offload() if os.environ.get("OFFLOAD_FLUX", "1") == "1" else pipe.to("cuda")
        _FLUX_PIPE = pipe
        print("[Vast Server] FLUX model loaded successfully.")
    except Exception as e:
        print(f"[Vast Server Error] Failed to load FLUX: {e}", file=sys.stderr)
        raise
    return _FLUX_PIPE


def load_ltx_pipeline(model_id: str = "jbilcke-hf/LTX-Video-2b-0-9-8-distilled-HFIE"):
    """Load the distilled LTX video pipeline with CPU offload for 24GB cards."""
    global _LTX_PIPE, _LTX_MODEL_ID
    if _LTX_PIPE is not None and _LTX_MODEL_ID == model_id:
        return _LTX_PIPE
    if _LTX_PIPE is not None:
        _release_other_models("none")

    _release_other_models("ltx")
    import torch
    from diffusers import LTXPipeline

    print(f"[Vast Server] Loading LTX-Video model: {model_id} on {DEVICE}...")
    dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
    pipe = LTXPipeline.from_pretrained(model_id, torch_dtype=dtype, token=_hub_token())
    if DEVICE == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(DEVICE)
    _LTX_PIPE = pipe
    _LTX_MODEL_ID = model_id
    print("[Vast Server] LTX-Video model loaded successfully.")
    return _LTX_PIPE


def load_indic_parler_engine():
    """Load the AI4Bharat Indic Parler-TTS engine onto the GPU (float32)."""
    global _INDIC_PARLER_ENGINE
    if _INDIC_PARLER_ENGINE is not None:
        return _INDIC_PARLER_ENGINE

    _release_other_models("indic_parler")
    print(f"[Vast Server] Loading Indic Parler-TTS engine on {DEVICE}...")
    try:
        from tts import IndicParlerEngine
        _INDIC_PARLER_ENGINE = IndicParlerEngine()
    except Exception:
        # Standalone loading if tts.py is not importable from the server CWD.
        import numpy as np
        import torch
        import torchaudio
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        repo_id = os.environ.get("INDIC_PARLER_MODEL", "ai4bharat/indic-parler-tts")

        class SimpleIndicParler:
            def __init__(self):
                self.device = DEVICE
                self.model = (
                    ParlerTTSForConditionalGeneration.from_pretrained(
                        repo_id, torch_dtype=torch.float32
                    )
                    .to(DEVICE)
                    .eval()
                )
                self.description_tokenizer = AutoTokenizer.from_pretrained(
                    self.model.config.text_encoder._name_or_path
                )
                self.prompt_tokenizer = AutoTokenizer.from_pretrained(repo_id)
                self.sample_rate = int(getattr(self.model.config, "sampling_rate", 44100))
                self.resampler = torchaudio.transforms.Resample(self.sample_rate, 24000)

            def synthesize(self, text, description, speed=1.0):
                input_ids = self.description_tokenizer(
                    description.strip(), return_tensors="pt"
                ).input_ids.to(self.device)
                prompt_ids = self.prompt_tokenizer(text, return_tensors="pt").input_ids.to(self.device)
                with torch.no_grad():
                    generation = self.model.generate(input_ids=input_ids, prompt_input_ids=prompt_ids)
                wav = generation.cpu().numpy().squeeze().astype(np.float32)
                if wav.ndim > 1:
                    wav = wav.mean(axis=0)
                return self.resampler(torch.from_numpy(wav)).numpy().astype(np.float32)

        _INDIC_PARLER_ENGINE = SimpleIndicParler()

    print("[Vast Server] Indic Parler-TTS engine loaded successfully.")
    return _INDIC_PARLER_ENGINE


class VastHandler(BaseHTTPRequestHandler):
    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health" or self.path == "/":
            vram = get_vram_info()
            # status stays "healthy": this server is alive and the GPU is usable. The
            # `tts_ready` field is the separate question the client actually needs answered,
            # because narration cannot use a server whose IndicF5 import path is broken.
            tts_error = tts_import_error()
            self._send_json({
                "status": "healthy",
                "device": DEVICE,
                "vram": vram,
                "flux_loaded": _FLUX_PIPE is not None,
                "ltx_loaded": _LTX_PIPE is not None,
                "tts_loaded": _INDIC_PARLER_ENGINE is not None,
                "tts_ready": tts_error is None,
                "tts_error": tts_error,
            })
        else:
            self._send_json({"error": "Not Found"}, status=404)

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else "{}"
        try:
            req = json.loads(raw_body)
        except Exception:
            self._send_json({"error": "Invalid JSON"}, status=400)
            return

        # Never let a handler error drop the connection silently: the client
        # would only see RemoteDisconnected, with the real cause buried in the
        # server log. Always answer with a JSON error payload instead.
        try:
            if self.path == "/v1/images/generate":
                self.handle_image_generate(req)
            elif self.path == "/v1/videos/generate":
                self.handle_video_generate(req)
            elif self.path == "/v1/audio/synthesize":
                self.handle_audio_synthesize(req)
            elif self.path == "/v1/audio/synthesize_batch":
                self.handle_audio_synthesize_batch(req)
            elif self.path == "/v1/batch":
                self.handle_batch(req)
            elif self.path == "/v1/shutdown":
                self._send_json({"message": "Server shutting down..."})
                sys.exit(0)
            else:
                self._send_json({"error": "Unknown endpoint"}, status=404)
        except Exception as err:  # noqa: BLE001 - report any handler failure to the client
            import traceback
            traceback.print_exc()
            try:
                self._send_json(
                    {"status": "error", "error": f"{type(err).__name__}: {err}"},
                    status=500,
                )
            except Exception:
                pass

    def handle_image_generate(self, req: dict):
        prompt = req.get("prompt", "")
        if not prompt:
            self._send_json({"error": "Missing prompt"}, status=400)
            return

        width = int(req.get("width", 1024))
        height = int(req.get("height", 1024))
        scale = min(1.0, 1024 / max(width, height))
        width = max(256, int(width * scale) // 16 * 16)
        height = max(256, int(height * scale) // 16 * 16)
        steps = int(req.get("steps", 4))
        guidance = float(req.get("guidance_scale", 0.0))
        seed = req.get("seed")

        import torch
        pipe = load_flux_pipeline()

        generator = None
        if seed is not None:
            generator = torch.Generator(device=DEVICE).manual_seed(int(seed))

        t0 = time.time()
        with torch.inference_mode():
            image = pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
            ).images[0]
        gen_time = round(time.time() - t0, 3)

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        self._send_json({
            "status": "success",
            "image_base64": img_b64,
            "width": width,
            "height": height,
            "duration_sec": gen_time,
        })

    def handle_video_generate(self, req: dict):
        prompt = str(req.get("prompt", "")).strip()
        if not prompt:
            self._send_json({"error": "Missing prompt"}, status=400)
            return

        width = max(256, int(req.get("width", 768)))
        height = max(256, int(req.get("height", 512)))
        num_frames = int(req.get("num_frames", 97))
        fps = int(req.get("fps", 24))
        steps = int(req.get("steps", 8))
        model_id = str(req.get("model", "jbilcke-hf/LTX-Video-2b-0-9-8-distilled-HFIE"))
        seed = req.get("seed")
        if width % 32 or height % 32 or num_frames < 9 or (num_frames - 1) % 8:
            self._send_json({
                "error": "LTX dimensions must be multiples of 32 and num_frames must be 8n+1 (at least 9)."
            }, status=400)
            return

        try:
            import torch
            from diffusers.utils import export_to_video

            pipe = load_ltx_pipeline(model_id)
            generator = None
            if seed is not None:
                generator = torch.Generator(device=DEVICE).manual_seed(int(seed))

            started = time.time()
            with torch.inference_mode():
                result = pipe(
                    prompt=prompt,
                    width=width,
                    height=height,
                    num_frames=num_frames,
                    num_inference_steps=steps,
                    guidance_scale=1.0,
                    generator=generator,
                )
            frames = result.frames[0]
            with tempfile.TemporaryDirectory(prefix="ltx-video-") as temp_dir:
                video_path = Path(temp_dir) / "scene.mp4"
                export_to_video(frames, str(video_path), fps=fps)
                video_bytes = video_path.read_bytes()

            self._send_json({
                "status": "success",
                "video_base64": base64.b64encode(video_bytes).decode("ascii"),
                "width": width,
                "height": height,
                "num_frames": num_frames,
                "fps": fps,
                "duration_sec": round(time.time() - started, 3),
            })
        except Exception as err:
            print(f"[Vast Server Error] LTX-Video generation failed: {err}", file=sys.stderr)
            self._send_json({"error": f"LTX-Video generation failed: {err}"}, status=500)

    def handle_audio_synthesize(self, req: dict):
        text = req.get("text", "").strip()
        voice = req.get("voice", "indic_parler_aarav").lower()
        speed = float(req.get("speed", 1.0))

        if not text:
            self._send_json({"error": "Missing text"}, status=400)
            return

        import numpy as np
        import soundfile as sf
        engine = load_indic_parler_engine()
        description = _resolve_indic_description(voice, speed)

        t0 = time.time()
        audio_np = engine.synthesize(text, description, speed=speed)
        gen_time = round(time.time() - t0, 3)

        audio_np = np.asarray(audio_np, dtype=np.float32)
        if _audio_is_bad(audio_np):
            self._send_json(
                {
                    "status": "error",
                    "error": "Indic Parler-TTS produced non-finite/constant audio",
                },
                status=500,
            )
            return

        buf = io.BytesIO()
        sf.write(buf, audio_np, 24000, format="WAV", subtype="PCM_16")
        audio_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        audio_sec = round(len(audio_np) / 24000, 2)

        self._send_json({
            "status": "success",
            "audio_base64": audio_b64,
            "sample_rate": 24000,
            "audio_duration_sec": audio_sec,
            "generation_time_sec": gen_time,
        })

    def handle_audio_synthesize_batch(self, req: dict):
        """Synthesize many narration segments in one round trip.

        The Indic Parler-TTS engine stays resident on the GPU, so a long script only pays
        the HTTP round-trip cost once instead of once per sentence.
        """
        items = req.get("items") or []
        if not items:
            self._send_json({"error": "Missing items"}, status=400)
            return

        voice = str(req.get("voice", "indic_parler_aarav"))
        speed = float(req.get("speed", 1.0))

        import soundfile as sf

        engine = load_indic_parler_engine()
        description = _resolve_indic_description(voice, speed)
        results = []
        started = time.time()
        for index, item in enumerate(items):
            entry_id = item.get("id")
            text = str(item.get("text") or "").strip()
            if not text:
                results.append({"id": entry_id, "error": "empty text"})
                continue
            t0 = time.time()
            try:
                audio_np = engine.synthesize(text, description, speed=speed)
            except Exception as err:  # noqa: BLE001 - report per-item failure to the client
                results.append({"id": entry_id, "error": f"{type(err).__name__}: {err}"})
                continue
            if _audio_is_bad(audio_np):
                results.append(
                    {"id": entry_id, "error": "non-finite/constant audio"}
                )
                continue
            buf = io.BytesIO()
            sf.write(buf, audio_np, 24000, format="WAV", subtype="PCM_16")
            results.append({
                "id": entry_id,
                "audio_base64": base64.b64encode(buf.getvalue()).decode("ascii"),
                "sample_rate": 24000,
                "audio_duration_sec": round(len(audio_np) / 24000, 3),
                "generation_time_sec": round(time.time() - t0, 3),
            })
            print(
                f"[Vast Server] audio batch {index + 1}/{len(items)} id={entry_id} "
                f"({round(time.time() - t0, 2)}s)",
                file=sys.stderr,
            )

        self._send_json({
            "status": "success",
            "sample_rate": 24000,
            "total_generation_time_sec": round(time.time() - started, 3),
            "results": results,
        })

    def handle_batch(self, req: dict):
        """Process a scene list: generate images and audio in one round trip."""
        import torch
        import soundfile as sf

        scenes = req.get("scenes", [])
        voice = req.get("voice", "indic_parler_aarav")
        speed = float(req.get("speed", 1.0))
        width = int(req.get("width", 1080))
        height = int(req.get("height", 1920))

        description = _resolve_indic_description(voice, speed)

        results = []
        for i, sc in enumerate(scenes):
            scene_id = sc.get("id", f"scene-{i+1:03d}")
            prompt = sc.get("image_prompt")
            narration_text = sc.get("narration_text")

            res_entry = {"id": scene_id}

            if prompt:
                pipe = load_flux_pipeline()
                with torch.inference_mode():
                    img = pipe(prompt=prompt, width=width, height=height, num_inference_steps=4, guidance_scale=0.0).images[0]
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                res_entry["image_base64"] = base64.b64encode(buf.getvalue()).decode("ascii")

            if narration_text:
                engine = load_indic_parler_engine()
                audio_np = engine.synthesize(narration_text, description, speed=speed)
                buf_a = io.BytesIO()
                sf.write(buf_a, audio_np, 24000, format="WAV", subtype="PCM_16")
                res_entry["audio_base64"] = base64.b64encode(buf_a.getvalue()).decode("ascii")
                res_entry["audio_duration_sec"] = round(len(audio_np) / 24000, 2)

            results.append(res_entry)

        self._send_json({"status": "success", "results": results})


def main():
    server = HTTPServer(("0.0.0.0", PORT), VastHandler)
    print(f"[Vast Server] Running on http://0.0.0.0:{PORT} (Device: {DEVICE})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Vast Server] Stopping server...")
        server.server_close()


if __name__ == "__main__":
    main()
