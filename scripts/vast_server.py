#!/usr/bin/env python3
"""Vast.ai GPU Inference Server for FLUX.1-schnell, LTX-Video & Indic Parler-TTS.

Runs on one remote 24GB GPU instance, hosting one requested visual pipeline:
  1. FLUX.1-schnell image generation (RTX 4090 workflow)
  2. LTX-Video 2B distilled video generation (RTX 4090 workflow)
  3. AI4Bharat IndicF5 reference-conditioned Hindi TTS
  4. Legacy Indic Parler-TTS compatibility

Endpoints:
  GET  /health              -> System, GPU VRAM & model status
  POST /v1/images/generate  -> FLUX.1 image generation
  POST /v1/videos/generate  -> LTX-Video MP4 generation
  POST /v1/audio/synthesize -> IndicF5 or legacy Indic Parler-TTS audio (WAV + duration)
  POST /v1/audio/synthesize_batch -> many Indic audio segments in one round trip
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

# Ensure UTF-8 output. Guarded because this module is imported by the test suite as well
# as run as the server: a captured or replaced stdout has no `reconfigure`, and an
# import-time AttributeError would take down the whole test process.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

PORT = int(os.environ.get("PORT", 8000))
DEVICE = "cuda" if os.environ.get("FORCE_CPU", "0") != "1" else "cpu"

_FLUX_PIPE = None
_LTX_PIPE = None
_LTX_MODEL_ID: str | None = None
_INDIC_PARLER_ENGINE = None
_INDICF5_ENGINE = None

# Latched once each engine's import path is known to work. A failure is deliberately NOT
# cached: an instance can be repaired over SSH while this process keeps running, and the
# next /health should reflect that without needing a restart.
_TTS_IMPORT_OK = False
_INDICF5_IMPORT_OK = False
INDICF5_REPOSITORY = "ai4bharat/IndicF5"
INDICF5_SAMPLE_RATE = 24000
INDICF5_AUDIO_EXTENSIONS = (".wav", ".flac", ".ogg", ".mp3")


def _audio_is_bad(wav) -> bool:
    """True when a waveform is empty, non-finite, silent or pure DC.

    Indic Parler-TTS can return NaN; libsndfile then encodes it as a constant -32768
    (0x8000) DC WAV that is structurally valid but carries no speech. Rejecting it here
    turns a corrupt generation into an explicit error instead of a silent success.
    """
    import numpy as np

    x = np.asarray(wav, dtype=np.float64).reshape(-1)
    if x.size == 0 or not np.all(np.isfinite(x)):
        return True
    if float(np.max(np.abs(x))) < 0.02:  # near-silence / blank output (~-34 dBFS peak)
        return True
    if float(np.sqrt(np.mean(x ** 2))) < 0.01:  # far too quiet to be speech
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


def indicf5_import_error() -> str | None:
    """Return why the remote IndicF5 runtime is unusable, or None when it imports."""
    global _INDICF5_IMPORT_OK
    if _INDICF5_IMPORT_OK:
        return None
    try:
        import numpy  # noqa: F401
        import soundfile  # noqa: F401
        import torch  # noqa: F401
        import torchaudio  # noqa: F401
        from transformers import AutoModel  # noqa: F401
        import vocos  # noqa: F401
        from f5_tts.infer import utils_infer  # noqa: F401
    except Exception as err:  # noqa: BLE001 - every failure makes inference impossible
        return f"{type(err).__name__}: {err}"
    _INDICF5_IMPORT_OK = True
    return None


def _hub_token() -> str | None:
    """Read an optional Hugging Face token from the server environment."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY") or None


def _release_other_models(active: str) -> None:
    """Keep only one large inference model resident on the 24GB GPU."""
    global _FLUX_PIPE, _LTX_PIPE, _LTX_MODEL_ID, _INDIC_PARLER_ENGINE, _INDICF5_ENGINE
    for name in ("flux", "ltx", "indic_parler", "indicf5"):
        if name == active:
            continue
        value = {
            "flux": _FLUX_PIPE,
            "ltx": _LTX_PIPE,
            "indic_parler": _INDIC_PARLER_ENGINE,
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
        elif name == "indic_parler":
            _INDIC_PARLER_ENGINE = None
        else:
            _INDICF5_ENGINE = None
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
        "A male speaker with a deep, resonant and authoritative voice speaks only in "
        f"pure Hindi with a clear native Indian accent at {_pace_word(speed)} in a very "
        "clear, close-sounding recording with no background noise."
    )


# How many fresh-seed draws to try before giving up on a segment. Parler-TTS can draw a
# degenerate (near-silent) sample for one seed, and because the seed is fixed for voice
# consistency, a plain retry reproduces the same silence -- so each attempt advances it.
INDIC_PARLER_ATTEMPTS = int(os.environ.get("INDIC_PARLER_ATTEMPTS", "5"))


def _synthesize_with_retries(engine, text: str, description: str, speed: float):
    """Generate audio, re-rolling the seed until the result is usable.

    Returns (audio_np, generation_time_sec), or (None, 0.0) when every draw was bad.
    """
    import numpy as np

    base = int(getattr(engine, "seed", 1234))
    for attempt in range(INDIC_PARLER_ATTEMPTS):
        t0 = time.time()
        try:
            audio = engine.synthesize(text, description, speed=speed, seed=base + attempt)
        except TypeError:  # an engine without a seed parameter
            audio = engine.synthesize(text, description, speed=speed)
        gen_time = round(time.time() - t0, 3)
        arr = np.asarray(audio, dtype=np.float32)
        if not _audio_is_bad(arr):
            return arr, gen_time
        peak = float(np.max(np.abs(arr))) if arr.size else 0.0
        print(
            f"[Vast Server] silent draw (seed={base + attempt}, peak={peak:.4f}); retrying",
            file=sys.stderr,
        )
    return None, 0.0


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

        repo_candidates: list[str]
        try:
            from tts import indic_parler_model_candidates
            repo_candidates = indic_parler_model_candidates()
        except Exception:
            repo_candidates = [os.environ.get("INDIC_PARLER_MODEL") or "RXD03/indic-parler-tts"]

        class SimpleIndicParler:
            def __init__(self):
                self.device = DEVICE
                model = None
                loaded_repo = ""
                last_err: Exception | None = None
                for repo_id in repo_candidates:
                    try:
                        model = ParlerTTSForConditionalGeneration.from_pretrained(
                            repo_id, torch_dtype=torch.float32
                        )
                        loaded_repo = repo_id
                        break
                    except Exception as err:  # noqa: BLE001
                        last_err = err
                        print(
                            f"[Vast Server] {repo_id} unavailable ({type(err).__name__}); "
                            "trying next candidate...",
                            file=sys.stderr,
                        )
                if model is None:
                    raise RuntimeError(f"no Indic Parler-TTS model could be loaded: {last_err}")
                self.model = model.to(DEVICE).eval()
                self.description_tokenizer = AutoTokenizer.from_pretrained(
                    self.model.config.text_encoder._name_or_path
                )
                self.prompt_tokenizer = AutoTokenizer.from_pretrained(loaded_repo)
                self.sample_rate = int(getattr(self.model.config, "sampling_rate", None) or 44100)
                self.resampler = torchaudio.transforms.Resample(self.sample_rate, 24000)
                self.seed = int(os.environ.get("INDIC_PARLER_SEED", "1234"))
                self.do_sample = os.environ.get("INDIC_PARLER_GREEDY", "0") != "1"

            def synthesize(self, text, description, speed=1.0, seed=None):
                input_ids = self.description_tokenizer(
                    description.strip(), return_tensors="pt"
                ).input_ids.to(self.device)
                prompt_ids = self.prompt_tokenizer(text, return_tensors="pt").input_ids.to(self.device)
                # One seed + greedy decoding for every segment keeps a single speaker.
                use_seed = self.seed if seed is None else int(seed)
                torch.manual_seed(use_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(use_seed)
                with torch.no_grad():
                    generation = self.model.generate(
                        input_ids=input_ids,
                        prompt_input_ids=prompt_ids,
                        do_sample=self.do_sample,
                        num_beams=1,
                    )
                wav = generation.cpu().numpy().squeeze().astype(np.float32)
                if wav.ndim > 1:
                    wav = wav.mean(axis=0)
                return self.resampler(torch.from_numpy(wav)).numpy().astype(np.float32)

        _INDIC_PARLER_ENGINE = SimpleIndicParler()

    print("[Vast Server] Indic Parler-TTS engine loaded successfully.")
    return _INDIC_PARLER_ENGINE


def _indicf5_voice_stem(voice: str) -> str:
    """Map public ``indicf5_aarav`` ids to their reference-file stem."""
    lowered = (voice or "indicf5_aarav").strip().lower()
    return lowered[len("indicf5_"):] if lowered.startswith("indicf5_") else lowered


def _voices_dir_has_clips(directory: Path) -> bool:
    """True when a directory actually holds a reference clip, not just transcripts."""
    try:
        return any(
            entry.is_file() and entry.suffix.lower() in INDICF5_AUDIO_EXTENSIONS
            for entry in directory.iterdir()
        )
    except OSError:
        return False


def _indicf5_voices_dir() -> Path:
    """Where the IndicF5 reference clips live.

    The resolution order matters because the layouts disagree. The client seeds a fixed
    writable directory on a rented instance (INDICF5_VOICES_DIR), a checked-out repo keeps
    them under the git-ignored examples tree, and a repo that ships them for deployment puts
    them in public/audio/prompts/indicf5. An explicit override always wins, and a directory
    holding only transcripts must not be chosen over one holding actual audio.
    """
    configured = os.environ.get("INDICF5_VOICES_DIR", "").strip()
    if configured:
        return Path(configured)
    repo = Path(__file__).resolve().parent.parent
    shipped = repo / "public" / "audio" / "prompts" / "indicf5"
    if _voices_dir_has_clips(shipped):
        return shipped
    return repo / "examples" / "indicf5" / "voices"


def _prepare_indicf5_reference(voice: str) -> tuple[str, str]:
    """Prepare an IndicF5 reference clip as a 24 kHz mono PCM WAV."""
    import numpy as np
    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly

    stem = _indicf5_voice_stem(voice)
    voices_dir = _indicf5_voices_dir()
    source = next((voices_dir / f"{stem}{extension}" for extension in INDICF5_AUDIO_EXTENSIONS
                   if (voices_dir / f"{stem}{extension}").is_file()), None)
    transcript = voices_dir / f"{stem}.txt"
    if source is None or not transcript.is_file():
        raise RuntimeError(f"IndicF5 voice '{stem}' has no reference clip/transcript in {voices_dir}")
    reference_text = " ".join(transcript.read_text(encoding="utf-8-sig").split())
    if not reference_text:
        raise RuntimeError(f"IndicF5 voice '{stem}' has an empty reference transcript")
    try:
        audio, sample_rate = sf.read(str(source), dtype="float32", always_2d=True)
    except Exception as err:  # noqa: BLE001 - include the voice context in this preflight error
        raise RuntimeError(f"IndicF5 voice '{stem}' reference clip cannot be decoded: {err}") from err
    audio = audio.mean(axis=1)
    if sample_rate <= 0:
        raise RuntimeError(f"IndicF5 voice '{stem}' reference clip has an invalid sample rate")
    if sample_rate != INDICF5_SAMPLE_RATE:
        factor = gcd(sample_rate, INDICF5_SAMPLE_RATE)
        audio = resample_poly(audio, INDICF5_SAMPLE_RATE // factor, sample_rate // factor).astype(np.float32)
    if _audio_is_bad(audio):
        raise RuntimeError(f"IndicF5 voice '{stem}' reference clip is silent/non-finite")
    peak = float(np.max(np.abs(audio)))
    prepared = voices_dir / f"_prepared_{stem}.wav"
    try:
        voices_dir.mkdir(parents=True, exist_ok=True)
        sf.write(str(prepared), audio / peak * 0.9, INDICF5_SAMPLE_RATE, subtype="PCM_16")
    except OSError as err:
        raise RuntimeError(f"IndicF5 voice directory is not writable: {voices_dir} ({err})") from err
    return str(prepared), reference_text


def indicf5_voice_error(voice: str = "indicf5_aarav") -> str | None:
    """Return an IndicF5 reference-readiness reason without loading the model."""
    try:
        _prepare_indicf5_reference(voice)
    except Exception as err:  # noqa: BLE001 - health must remain a JSON response
        return f"{type(err).__name__}: {err}"
    return None


def load_indicf5_engine():
    """Load one resident AI4Bharat IndicF5 model and patch reference decoding safely."""
    global _INDICF5_ENGINE
    if _INDICF5_ENGINE is not None:
        return _INDICF5_ENGINE
    problem = indicf5_import_error()
    if problem:
        raise RuntimeError(f"IndicF5 runtime unavailable: {problem}")

    _release_other_models("indicf5")
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    from transformers import AutoModel

    # New torchaudio versions may require TorchCodec merely to decode a WAV. IndicF5 only
    # needs the decoded reference waveform, so use SoundFile just as the local batch path does.
    def load_with_soundfile(path, *args, **kwargs):
        data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
        return torch.from_numpy(np.ascontiguousarray(data.T)), sample_rate

    torchaudio.load = load_with_soundfile
    print(f"[Vast Server] Loading IndicF5 model {INDICF5_REPOSITORY} on {DEVICE}...")
    _INDICF5_ENGINE = AutoModel.from_pretrained(
        INDICF5_REPOSITORY, trust_remote_code=True, token=_hub_token()
    )
    if hasattr(_INDICF5_ENGINE, "to"):
        _INDICF5_ENGINE.to(DEVICE)
    if hasattr(_INDICF5_ENGINE, "eval"):
        _INDICF5_ENGINE.eval()
    print("[Vast Server] IndicF5 model loaded successfully.")
    return _INDICF5_ENGINE


def synthesize_indicf5(text: str, voice: str):
    """Generate one valid IndicF5 waveform from its selected cloning reference."""
    import numpy as np
    import torch

    model = load_indicf5_engine()
    reference_path, reference_text = _prepare_indicf5_reference(voice)
    started = time.time()
    with torch.inference_mode():
        audio = model(text, ref_audio_path=reference_path, ref_text=reference_text)
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    audio_np = np.asarray(audio).squeeze().astype(np.float32)
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=0)
    peak = float(np.max(np.abs(audio_np))) if audio_np.size else 0.0
    if peak > 1.0:
        audio_np = audio_np / peak
    if _audio_is_bad(audio_np):
        raise RuntimeError("IndicF5 produced silent/non-finite audio")
    return audio_np, round(time.time() - started, 3)


def synthesize_requested_voice(text: str, voice: str, speed: float):
    """Dispatch public audio endpoints by voice family, retaining Parler compatibility."""
    if (voice or "").strip().lower().startswith("indicf5_"):
        return synthesize_indicf5(text, voice)
    engine = load_indic_parler_engine()
    return _synthesize_with_retries(engine, text, _resolve_indic_description(voice, speed), speed)


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
            # status stays "healthy": this server is alive and the GPU is usable. Engine
            # readiness is reported separately because model imports may be lazy/optional.
            tts_error = tts_import_error()
            indicf5_error = indicf5_import_error()
            indicf5_reference_error = None if indicf5_error else indicf5_voice_error()
            self._send_json({
                "status": "healthy",
                "device": DEVICE,
                "vram": vram,
                "flux_loaded": _FLUX_PIPE is not None,
                "ltx_loaded": _LTX_PIPE is not None,
                "tts_loaded": _INDIC_PARLER_ENGINE is not None,
                "tts_greedy": getattr(_INDIC_PARLER_ENGINE, "do_sample", True) is False,
                "tts_ready": tts_error is None,
                "tts_error": tts_error,
                "indicf5_loaded": _INDICF5_ENGINE is not None,
                # Reported separately from indicf5_error so the client can tell a gated-repo
                # 401 (fix: supply a token, which needs a NEW container) from a broken install
                # (fix: repair in place). The two have different remedies, and one of them
                # cannot be applied to a container that already exists.
                "hf_token_present": bool(_hub_token()),
                "indicf5_ready": indicf5_error is None and indicf5_reference_error is None,
                "indicf5_error": indicf5_error or indicf5_reference_error,
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

        import soundfile as sf
        audio_np, gen_time = synthesize_requested_voice(text, voice, speed)
        if audio_np is None:
            self._send_json(
                {
                    "status": "error",
                    "error": "Indic TTS produced silent/non-finite audio after retries",
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
                audio_np, _gen = synthesize_requested_voice(text, voice, speed)
            except Exception as err:  # noqa: BLE001 - report per-item failure to the client
                results.append({"id": entry_id, "error": f"{type(err).__name__}: {err}"})
                continue
            if audio_np is None:
                results.append({"id": entry_id, "error": "silent/non-finite audio"})
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
                audio_np, _gen = synthesize_requested_voice(narration_text, voice, speed)
                if audio_np is None:
                    continue
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
