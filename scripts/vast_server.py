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
  POST /v1/audio/synthesize -> IndicF5 audio generation + word timings
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
_INDICF5_ENGINE = None


def _hub_token() -> str | None:
    """Read an optional Hugging Face token from the server environment."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY") or None


def _release_other_models(active: str) -> None:
    """Keep only one large inference model resident on the 24GB GPU."""
    global _FLUX_PIPE, _LTX_PIPE, _LTX_MODEL_ID, _INDICF5_ENGINE
    for name in ("flux", "ltx", "indicf5"):
        if name == active:
            continue
        value = {
            "flux": _FLUX_PIPE,
            "ltx": _LTX_PIPE,
            "indicf5": _INDICF5_ENGINE,
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
            _INDICF5_ENGINE = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


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


def load_indicf5_engine():
    """Load AI4Bharat IndicF5 TTS engine onto GPU."""
    global _INDICF5_ENGINE
    if _INDICF5_ENGINE is not None:
        return _INDICF5_ENGINE

    _release_other_models("indicf5")
    print(f"[Vast Server] Loading AI4Bharat IndicF5 engine on {DEVICE}...")
    try:
        from tts import IndicF5Engine
        _INDICF5_ENGINE = IndicF5Engine()
    except Exception:
        # Standalone loading if tts.py is not in path
        from cached_path import cached_path
        from f5_tts.model import CFM, DiT
        from f5_tts.model.utils import get_tokenizer
        from safetensors.torch import load_file
        from vocos import Vocos
        import soundfile as sf
        import torch, torchaudio

        def _soundfile_load(filepath, *args, **kwargs):
            wav_np, sr = sf.read(str(filepath), dtype="float32")
            if wav_np.ndim == 1:
                t = torch.from_numpy(wav_np).unsqueeze(0)
            else:
                t = torch.from_numpy(wav_np.T)
            return t, sr

        torchaudio.load = _soundfile_load
        torch._dynamo.config.suppress_errors = True
        torch.compiler.disable()

        vocab_path = str(cached_path("hf://rsolanki1822/IndicF5-mirror/checkpoints/vocab.txt"))
        model_path = str(cached_path("hf://rsolanki1822/IndicF5-mirror/model.safetensors"))
        vocab_char_map, vocab_size = get_tokenizer(vocab_path, tokenizer="custom")

        transformer = DiT(
            dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512,
            conv_layers=4, text_num_embeds=vocab_size, mel_dim=100,
        )
        state_dict = load_file(model_path, device="cpu")
        clean_sd = {k.replace("ema_model._orig_mod.", "").replace("ema_model.", ""): v for k, v in state_dict.items() if "vocoder" not in k}

        cfm_model = CFM(
            transformer=transformer,
            mel_spec_kwargs=dict(
                target_sample_rate=24000, n_mel_channels=100, hop_length=256,
                win_length=1024, n_fft=1024, mel_spec_type="vocos",
            ),
            odeint_kwargs=dict(method="euler"),
            vocab_char_map=vocab_char_map,
        )
        cfm_model.load_state_dict(clean_sd, strict=False)
        cfm_model.to(device=DEVICE, dtype=torch.float32 if DEVICE == "cpu" else torch.float16).eval()

        vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
        voc_sd = {k.replace("vocoder._orig_mod.", "").replace("vocoder.", ""): v for k, v in state_dict.items() if "vocoder" in k}
        if voc_sd:
            vocos.load_state_dict(voc_sd, strict=True)
        vocos.to(device=DEVICE).eval()

        class SimpleIndicF5:
            def __init__(self, m, v):
                self.model = m
                self.vocos = v
                self.device = DEVICE

            def synthesize(self, text, ref_audio, ref_text, speed=1.0, nfe_step=32):
                import numpy as np
                from f5_tts.infer.utils_infer import infer_process
                with torch.no_grad():
                    wav, sr, _ = infer_process(
                        ref_audio=str(ref_audio),
                        ref_text=ref_text,
                        gen_text=text,
                        model_obj=self.model,
                        vocoder=self.vocos,
                        mel_spec_type="vocos",
                        target_rms=0.1,
                        cross_fade_duration=0.15,
                        nfe_step=nfe_step,
                        cfg_strength=2.0,
                        speed=speed,
                        device=self.device,
                    )
                if isinstance(wav, torch.Tensor):
                    wav = wav.squeeze().cpu().numpy()
                return np.asarray(wav, dtype=np.float32)

        _INDICF5_ENGINE = SimpleIndicF5(cfm_model, vocos)

    print("[Vast Server] IndicF5 TTS engine loaded successfully.")
    return _INDICF5_ENGINE


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
            self._send_json({
                "status": "healthy",
                "device": DEVICE,
                "vram": vram,
                "flux_loaded": _FLUX_PIPE is not None,
                "ltx_loaded": _LTX_PIPE is not None,
                "indicf5_loaded": _INDICF5_ENGINE is not None,
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
        voice = req.get("voice", "indicf5_aarav").lower()
        speed = float(req.get("speed", 1.0))

        if not text:
            self._send_json({"error": "Missing text"}, status=400)
            return

        import numpy as np
        import soundfile as sf
        engine = load_indicf5_engine()

        # Resolve prompt reference (absolute, so it works regardless of CWD)
        ref_dir = Path(__file__).resolve().parent.parent / "public/audio/prompts/indicf5"
        speaker = "aarav"
        if "priya" in voice:
            speaker = "priya"
        elif "kabir" in voice:
            speaker = "kabir"
        elif "rohan" in voice:
            speaker = "rohan"
        elif "ananya" in voice:
            speaker = "ananya"

        ref_wav = ref_dir / f"{speaker}.wav"
        ref_txt = ref_dir / f"{speaker}.txt"
        if not ref_wav.exists():
            self._send_json(
                {"status": "error", "error": f"Missing IndicF5 reference voice: {ref_wav}"},
                status=500,
            )
            return
        ref_text_content = ref_txt.read_text(encoding="utf-8").strip() if ref_txt.exists() else "मौर्यन एम्पायर की हर डेट सिर्फ़ रटो मत।"

        t0 = time.time()
        audio_np = engine.synthesize(text, ref_wav, ref_text_content, speed=speed, nfe_step=32)
        gen_time = round(time.time() - t0, 3)

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

    def handle_batch(self, req: dict):
        """Process batch of scenes: generate images and audio in one roundtrip."""
        scenes = req.get("scenes", [])
        voice = req.get("voice", "indicf5_aarav")
        speed = float(req.get("speed", 1.0))
        width = int(req.get("width", 1080))
        height = int(req.get("height", 1920))

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
                engine = load_indicf5_engine()
                ref_dir = Path(__file__).resolve().parent.parent / "public/audio/prompts/indicf5"
                ref_wav = ref_dir / "aarav.wav"
                ref_txt = ref_dir / "aarav.txt"
                ref_t = ref_txt.read_text(encoding="utf-8").strip() if ref_txt.exists() else "मौर्यन एम्पायर"
                audio_np = engine.synthesize(narration_text, ref_wav, ref_t, speed=speed, nfe_step=32)
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
