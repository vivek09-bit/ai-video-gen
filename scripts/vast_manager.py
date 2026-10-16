#!/usr/bin/env python3
"""Vast.ai GPU Cloud Manager & Client.

Image generation uses one RTX 3090; LTX-Video uses one RTX 4090. Both can also
run remote Indic Parler-TTS narration before the selected instance is stopped.

An already-rented instance of the right GPU is reused before anything new is
rented, and every offer is filtered against a per-GPU all-in price ceiling
($0.40/hr for an RTX 3090, $0.50/hr for an RTX 4090). Override with
VAST_MAX_DPH_RTX_3090 / VAST_MAX_DPH_RTX_4090, or --max-price.

Setup dominates the cost of a fresh rental, so: the container install is guarded by a
sentinel (a restarted instance never reinstalls), apt/deps/model-download run
concurrently, deps install with uv, and fast hosts are preferred. A host that is slow
rather than broken is stopped and kept, not destroyed, so its install survives.

Tuning knobs:
  VAST_MIN_INET_DOWN        minimum download Mbps for the strict offer filter (1000)
  VAST_MIN_DISK_BW          minimum disk MB/s for the strict offer filter (500)
  VAST_OFFER_WAIT_SECONDS   seconds to wait before re-searching when no offers (15)
  VAST_OFFER_SEARCH_ATTEMPTS  how many offer searches before giving up (4)
  VAST_CONTAINER_TIMEOUT    seconds to wait for the container to reach 'running' (600)
  VAST_SERVER_STALL_TIMEOUT seconds of no install progress before giving up (600)
  VAST_PROGRESS_MIN_DELTA_GB  minimum disk growth that counts as progress (0.05)
  VAST_INSTALL_FLOOR_GB       real install growth below this means it never ran (1.5)
  VAST_INSTALL_GRACE_SECONDS  seconds after the container starts to reach that growth (300)
  VAST_DEAD_POLLS             consecutive polls confirming a dead container before failing (2)
  VAST_SERVER_TIMEOUT       hard cap for the whole boot, in seconds (3600)
  VAST_SERVER_PREFETCH      comma-separated Hugging Face repos to prefetch
  VAST_UV_TORCH_BACKEND     torch index for `uv pip install --torch-backend` (default: derived from the image tag)

Boot-stall policy (Option B): a stalled-but-alive install is STOPPED and kept (resumable),
while a host that provably installed nothing (peak disk < VAST_INSTALL_FLOOR_GB) or whose
image never pulled is treated as broken and destroyed -- a stopped empty install is not
worth the storage it would keep billing.
  VAST_SERVER_IMAGE         container image override

Usage:
  # Search cheapest RTX 4090 offers within budget
  python scripts/vast_manager.py search

  # Rent the cheapest in-budget instance, replacing hosts that fail to start
  python scripts/vast_manager.py create [--disk 50]

  # Start existing instance
  python scripts/vast_manager.py start <INSTANCE_ID>

  # Check instance health / status
  python scripts/vast_manager.py status <INSTANCE_ID>

  # Stop instance to pause billing
  python scripts/vast_manager.py stop <INSTANCE_ID>

  # Destroy broken, over-budget or long-idle instances
  python scripts/vast_manager.py prune [--dry-run] [--idle-hours 4]

  # Reapply the pinned server dependencies on an instance whose install is broken
  python scripts/vast_manager.py repair <INSTANCE_ID>
  python scripts/vast_manager.py hosts [--clear]
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

# Ensure project root is in path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from generate_images import load_env_file

load_env_file()

# ---------------------------------------------------------------------------
# Hourly price ceilings in USD/hour, all-in (GPU + storage + bandwidth).
# Override any model with an env var: VAST_MAX_DPH_RTX_3090=0.35
# ---------------------------------------------------------------------------
DEFAULT_GPU_BUDGETS: dict[str, float] = {
    # RTX 4090 is the only production target; retain the 3090 ceiling for legacy CLI
    # compatibility and old-instance cleanup, but orchestration never selects it.
    "RTX 3090": 0.40,
    "RTX 4090": 0.50,
}


def _gpu_slug(gpu_name: str) -> str:
    """Normalise a GPU name for env-var lookup: 'RTX 3090' -> 'RTX_3090'."""
    slug = "".join(ch if ch.isalnum() else "_" for ch in gpu_name.upper()).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug


def _image_torch_version(image: str) -> str:
    """Extract the torch version a base-image tag pins, e.g. '2.4.0'.

    The image is `pytorch/pytorch:2.4.0-cuda12.4-...`, so the tag names the exact torch
    build. torchaudio must be pinned to that same version: an unpinned resolution selects a
    newer torchaudio whose libtorchaudio.so calls torch C++ symbols the image's libtorch
    does not export, and the failure appears only at first synthesis as an undefined
    symbol (_ZN2at4_ops...) -- which the client sees as an opaque HTTP 500. Returns ""
    when the tag carries no usable version.
    """
    match = re.search(r"(\d+\.\d+\.\d+)", image or "")
    return match.group(1) if match else ""


def _image_torch_backend(image: str) -> str:
    """Derive uv's torch backend ('cu124') from an image tag such as 'cuda12.4'.

    The torch family must come from the index matching the image's own CUDA build.
    Without that, a `torchaudio` wheel built for a different CUDA version gets resolved
    and cannot be loaded by the image's torch: the Indic Parler-TTS engine then fails
    at first use with "Could not load .../torchaudio/lib/libtorchaudio.so", which reaches
    the client as an opaque HTTP 500. Falls back to uv's own detection when the image tag
    carries no CUDA version.
    """
    match = re.search(r"cuda(\d+)\.(\d+)", image or "")
    if match:
        return f"cu{match.group(1)}{match.group(2)}"
    if "2.4.0" in (image or ""):
        return "cu124"
    return "auto"


def gpu_budget(gpu_name: str) -> float:
    """Return the maximum $/hr we are willing to pay for this GPU model."""
    env_name = f"VAST_MAX_DPH_{_gpu_slug(gpu_name)}"
    env_value = os.environ.get(env_name)
    if env_value:
        try:
            return float(env_value)
        except ValueError:
            print(f"[Vast Manager Warning] Ignoring invalid {env_name}={env_value!r}", file=sys.stderr)
    normalised = gpu_name.upper().replace("NVIDIA GEFORCE ", "").strip()
    for name, limit in DEFAULT_GPU_BUDGETS.items():
        if name in normalised or normalised in name:
            return limit
    return float(os.environ.get("VAST_MAX_DPH", "0.50"))


def hourly_rate(record: dict) -> float:
    """Extract the all-in $/hr (GPU + storage + bandwidth) from an offer or instance."""
    for key in ("dph_total", "dph_total_adj", "dph_base", "dph"):
        value = record.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    return float("inf")


def format_rate(rate: float) -> str:
    """Render a $/hr figure, or 'n/a' when the API did not report one."""
    return "n/a" if rate == float("inf") else f"${rate:.3f}/hr"


# The container needs a Hugging Face token at RUNTIME, not only at install time. The
# narration model (ai4bharat/IndicF5) is a GATED repository: the Hub refuses to serve its
# files until the account has accepted the terms, so an unauthenticated container cannot
# prefetch it and cannot load it with trust_remote_code either. Both failures surface only
# after the GPU has been rented, so the token travels with the instance.
def hub_token() -> str:
    """The Hugging Face token to hand to a rented instance, or '' when none is configured.

    Both spellings are accepted because this repo's .env uses HUGGINGFACE_API_KEY while the
    inference server reads HF_TOKEN; only HF_TOKEN is put on the wire.
    """
    token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY") or "").strip()
    if not token:
        return ""
    # The token rides inside one space-separated docker-options string, beside the port
    # mapping. A stray space or quote would corrupt that string and yield a broken instance
    # that looks like a Vast fault, so refuse rather than transmit it.
    if any(character.isspace() or character in "\"'" for character in token):
        print(
            "[Vast Manager Warning] Ignoring a malformed HF token (it contains whitespace or "
            "quotes); the instance will not be able to download gated models.",
            file=sys.stderr,
        )
        return ""
    return token


_MISSING_TOKEN_WARNED = False


def _warn_missing_hub_token() -> None:
    """Say once per process that a rented instance will be unable to download gated models."""
    global _MISSING_TOKEN_WARNED
    if _MISSING_TOKEN_WARNED:
        return
    _MISSING_TOKEN_WARNED = True
    print(
        "[Vast Manager Warning] No HF_TOKEN / HUGGINGFACE_API_KEY in the environment. "
        "ai4bharat/IndicF5 is a gated repository, so this instance will fail its IndicF5 "
        "health check. Add the token to .env before renting.",
        file=sys.stderr,
    )


# Where the container keeps its IndicF5 reference clips. A fixed path is deliberate: the
# server writes a normalised `_prepared_<stem>.wav` beside them and a stopped instance keeps
# whatever it was given, so a stable location is what lets the client top the directory up
# over SSH (push_indicf5_voices) instead of hoping the deployed repo happens to ship audio.
CONTAINER_VOICES_DIR = "/workspace/indicf5-voices"

# Audio file types the IndicF5 runtime accepts as a cloning reference.
INDICF5_VOICE_EXTENSIONS: tuple[str, ...] = (".wav", ".flac", ".ogg", ".mp3")


def _http_json(req: urllib.request.Request, timeout: int) -> dict:
    """POST/GET JSON while surfacing the server's own error body.

    The Vast inference server answers handler failures with a 500 plus
    {"status": "error", "error": "..."}, but urlopen raises before the body can be read,
    so the reason was being thrown away. Every server-side fault looked identical
    ("HTTP Error 500: Internal Server Error"), including a model-load failure which the
    server names explicitly in that body.
    """
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        body = err.read().decode("utf-8", errors="ignore")
        detail: Any = body
        try:
            parsed = json.loads(body)
            detail = parsed.get("error") or parsed
        except Exception:
            pass
        raise RuntimeError(f"HTTP {err.code} from {req.full_url}: {str(detail)[:400]}") from err


class InstanceBroken(RuntimeError):
    """Raised when a rented instance cannot become healthy (bad host or image pull)."""


class InstanceSlow(TimeoutError):
    """Raised when a healthy instance simply needs longer to finish booting.

    Distinct from InstanceBroken: the host is fine, so the instance is stopped and
    kept rather than destroyed, preserving the dependency install it already paid for.
    """


# Python packages the inference server needs. Installed with uv (and falling back to
# pip) at container start; the CUDA build of torch ships in the base image and is left
# alone on purpose, so no --upgrade here. The f5_tts package is specifically
# AI4Bharat's IndicF5 fork: PyPI's similarly named f5-tts project is a different engine.
SERVER_DEPENDENCIES: tuple[str, ...] = (
    # Pinned, not floating. The LTX checkpoint declares `_diffusers_version: 0.32.0.dev0` and
    # names the exact component classes it ships (LTXPipeline, LTXVideoTransformer3DModel,
    # AutoencoderKLLTXVideo), so a newer diffusers may rename or re-signature them and the
    # failure would surface only at the first video request, long after the rental was paid.
    # 0.32.2 is also the version the local venv runs, so both sides agree.
    "diffusers==0.32.2",
    "accelerate==1.15.0",
    "sentencepiece",
    "protobuf==7.36.2",
    "imageio",
    "imageio-ffmpeg",
    # Pinned to the same commit as requirements.txt: the remote runtime and the local one
    # must be the same fork revision, or a synthesis difference between them is untraceable.
    "f5_tts @ git+https://github.com/AI4Bharat/IndicF5.git@13f7c4d627cc10111aea8fe9c0039462cacacdc7",
    "vocos==0.1.0",
    "transformers==4.49.0",
    "transformers-stream-generator==0.0.5",
    "numpy==1.26.4",
    "scipy==1.17.1",
    "soundfile==0.14.0",
    "huggingface-hub==0.36.2",
    "datasets==3.6.0",
    "pyarrow==20.0.0",
    # torchaudio is deliberately NOT listed here: it is pinned to the image's exact torch
    # version at install time (see server_onstart_command). Resolving it freely installs a
    # newer torchaudio than the image's torch, which then fails at import with an
    # unresolved C++ symbol.
)

# The import surface the server actually needs, including the modules it imports LAZILY
# inside a request handler. Probing only the top-level packages is not enough, and that gap
# is exactly how a broken instance was marked healthy: `f5_tts/__init__.py` never touches
# datasets or pyarrow -- only `f5_tts.infer.utils_infer` does, and the synthesis handler
# imports that at request time. A pyarrow built for a newer NumPy than the image ships
# therefore passed every top-level check, had the "deps ready" marker written for it, and
# still answered HTTP 500 to every synthesis request it ever received.
SERVER_DEPENDENCY_PROBE: str = (
    "import numpy, torch, torchaudio, diffusers, transformers, soundfile, scipy, vocos; "
    "from f5_tts.infer.utils_infer import infer_process; "
    # The pipeline classes the request handlers construct lazily. Naming them here is what
    # turns a diffusers version drift into a boot-time failure instead of an HTTP 500 at the
    # first image or video request. Deliberately absent: parler_tts, whose absence must never
    # block IndicF5 production readiness.
    "from diffusers import FluxPipeline, LTXPipeline"
)

# Hugging Face repos prefetched during the install so the first request is not stuck
# on a download. IndicF5 is the remote narration runtime; the legacy Parler repo remains
# available only for backward-compatible voices.
DEFAULT_PREFETCH_MODELS: tuple[str, ...] = ("ai4bharat/IndicF5",)

# Container image. Ships a CUDA build of torch, which the install must not upgrade.
DEFAULT_SERVER_IMAGE = "vive11/tube2dscript-server:2.4.0-cuda12.4-v1"

# Chromium shared libraries Remotion's headless shell needs, plus Noto fonts. The fonts are
# a correctness requirement, not a nicety: the theme fonts are Windows names (Georgia,
# Segoe UI) that Linux does not have, so Devanagari captions need a fallback family.
RENDER_STACK_PACKAGES: tuple[str, ...] = (
    "ca-certificates",
    "curl",
    "libnss3",
    "libatk1.0-0",
    "libatk-bridge2.0-0",
    "libcups2",
    "libdrm2",
    "libxkbcommon0",
    "libxcomposite1",
    "libxdamage1",
    "libxfixes3",
    "libxrandr2",
    "libgbm1",
    "libasound2",
    "fonts-noto-core",
    "fonts-noto-color-emoji",
)


def _wav_is_bad(wav_bytes: bytes) -> bool:
    """True when a returned WAV is empty, non-finite, silent or pure DC.

    Indic Parler-TTS can emit NaN on some draws, which libsndfile encodes as a constant
    -32768 (0x8000) DC WAV that is structurally valid but carries no speech. Rejecting it
    at the transport boundary stops silent audio from reaching the disk cache and render.
    """
    try:
        import io

        import soundfile as sf

        wav_np, _sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    except Exception:
        return True
    import numpy as np

    x = np.asarray(wav_np, dtype=np.float64).reshape(-1)
    if x.size == 0 or not np.all(np.isfinite(x)):
        return True
    if float(np.max(np.abs(x))) < 0.02:  # near-silence / blank output (~-34 dBFS peak)
        return True
    if float(np.sqrt(np.mean(x ** 2))) < 0.01:  # far too quiet to be speech
        return True
    if np.std(x) < 1e-5:  # constant / pure DC
        return True
    return False


class VastManager:
    """Manages Vast.ai GPU instance lifecycle and remote inference."""

    # Machine IDs that failed to start are remembered here so the same broken host
    # is never rented twice (the cheapest offer is often the flakiest one).
    BAD_HOSTS_FILE = ".vast_bad_hosts.json"
    # When each instance was last stopped, so `prune --idle-hours` can reclaim storage.
    IDLE_FILE = ".vast_idle.json"

    def __init__(self, api_key: str | None = None, server_url: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("VAST_API_KEY") or os.environ.get("VASTAI_API_KEY")
        self.server_url = (server_url or os.environ.get("VAST_SERVER_URL", "")).rstrip("/")
        self.bad_hosts = self._load_bad_hosts()

    @classmethod
    def _bad_hosts_path(cls) -> Path:
        return ROOT / cls.BAD_HOSTS_FILE

    def _load_bad_hosts(self) -> dict[str, str]:
        """Load machine IDs previously blacklisted as reasons for the blacklist."""
        path = self._bad_hosts_path()
        if not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[Vast Manager Warning] Ignoring unreadable {path.name}: {e}", file=sys.stderr)
            return {}
        machines = data.get("machines", {}) if isinstance(data, dict) else {}
        return {str(key): str(value) for key, value in machines.items()}

    def mark_bad_host(self, machine_id: Any, reason: str) -> None:
        """Persist a host that cannot start so no future run rents it again."""
        try:
            key = str(int(machine_id))
        except (TypeError, ValueError):
            return
        if key in self.bad_hosts:
            return
        self.bad_hosts[key] = reason[:200]
        path = self._bad_hosts_path()
        try:
            path.write_text(
                json.dumps({"machines": self.bad_hosts}, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(f"[Vast Manager] Blacklisted machine {key}: no future run will rent it ({reason[:120]})")
        except Exception as e:
            print(f"[Vast Manager Warning] Could not write {path.name}: {e}", file=sys.stderr)

    # Non-interactive SSH. BatchMode stops it prompting for a password, and
    # accept-new records a first-seen host key without asking. Without these the run
    # would hang forever on a prompt, which this project has been bitten by before.
    SSH_OPTIONS: tuple[str, ...] = (
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=20",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=40",
    )

    @staticmethod
    def _ssh_binary(name: str) -> str:
        found = shutil.which(name)
        if not found:
            raise RuntimeError(
                f"'{name}' was not found on PATH. Remote rendering needs the OpenSSH client "
                f"(Windows 10+ ships it as an optional feature)."
            )
        return found

    def ssh_target(self, instance_id: int) -> tuple[str, int, str]:
        """Resolve (host, port, user) for an instance, preferring Vast's own ssh-url."""
        try:
            raw = self.run_cli(["ssh-url", str(instance_id)]).strip()
        except Exception:
            raw = ""
        if raw.startswith("ssh://"):
            parsed = urllib.parse.urlparse(raw)
            if parsed.hostname:
                return parsed.hostname, int(parsed.port or 22), parsed.username or "root"
        info = self.get_instance_info(instance_id)
        ip = (info.get("public_ipaddr") or "").strip()
        mapping = (info.get("ports") or {}).get("22/tcp") or []
        if ip and mapping:
            return ip, int(mapping[0]["HostPort"]), "root"
        raise RuntimeError(
            f"Could not resolve an SSH address for instance {instance_id}. "
            "Was it created with --ssh, and is it running?"
        )

    def run_ssh(self, instance_id: int, command: str, timeout: int = 600) -> str:
        """Run a shell command on the instance over SSH and return its stdout."""
        host, port, user = self.ssh_target(instance_id)
        cmd = [self._ssh_binary("ssh"), *self.SSH_OPTIONS, "-p", str(port), f"{user}@{host}", command]
        try:
            res = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"SSH command timed out after {timeout}s on instance {instance_id}: {command[:120]}") from e
        if res.returncode != 0:
            tail = (res.stderr or res.stdout or "").strip().splitlines()[-12:]
            raise RuntimeError(
                f"SSH command failed on instance {instance_id} (exit {res.returncode}):\n  "
                + "\n  ".join(tail)
            )
        return res.stdout

    def _transfer(self, instance_id: int, local_path: str, remote_path: str, upload: bool) -> None:
        """Copy one file in either direction with scp (note scp uses -P, not -p)."""
        host, port, user = self.ssh_target(instance_id)
        remote = f"{user}@{host}:{remote_path}"
        source, destination = (local_path, remote) if upload else (remote, local_path)
        cmd = [self._ssh_binary("scp"), *self.SSH_OPTIONS, "-P", str(port), "-q", source, destination]
        try:
            res = subprocess.run(
                cmd, capture_output=True, text=True, timeout=3600, stdin=subprocess.DEVNULL
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"File transfer timed out for instance {instance_id}") from e
        if res.returncode != 0:
            tail = (res.stderr or res.stdout or "").strip().splitlines()[-6:]
            raise RuntimeError(
                f"scp failed on instance {instance_id} (exit {res.returncode}):\n  " + "\n  ".join(tail)
            )

    def push_file(self, instance_id: int, local_path: str | Path, remote_path: str) -> None:
        """Upload a local file to the instance."""
        self._transfer(instance_id, str(local_path), remote_path, upload=True)

    def pull_file(self, instance_id: int, remote_path: str, local_path: str | Path) -> None:
        """Download a remote file from the instance."""
        self._transfer(instance_id, str(local_path), remote_path, upload=False)

    def local_indicf5_voices_dir(self) -> Path:
        """The local directory holding the `<stem>.wav` + `<stem>.txt` reference pairs."""
        configured = os.environ.get("INDICF5_LOCAL_VOICES_DIR", "").strip()
        if configured:
            return Path(configured)
        return ROOT / "examples" / "indicf5" / "voices"

    def indicf5_voice_files(self) -> list[Path]:
        """Every usable reference pair locally: the clip AND the transcript, audio first.

        A clip without its transcript is unusable -- the reference text is what conditions the
        clone -- so an orphaned clip is reported and skipped rather than uploaded to produce a
        voice the server would then reject.
        """
        voices_dir = self.local_indicf5_voices_dir()
        if not voices_dir.is_dir():
            return []
        files: list[Path] = []
        for audio in sorted(voices_dir.iterdir()):
            if not audio.is_file() or audio.suffix.lower() not in INDICF5_VOICE_EXTENSIONS:
                continue
            if audio.name.startswith("_prepared_"):
                continue  # a normalised artefact of a previous run, not a source clip
            transcript = audio.with_suffix(".txt")
            if not transcript.is_file():
                print(
                    f"[Vast Manager Warning] Skipping reference clip {audio.name}: no matching "
                    f"{transcript.name} (IndicF5 needs the exact transcript of the clip).",
                    file=sys.stderr,
                )
                continue
            files.extend((audio, transcript))
        return files

    def push_indicf5_voices(self, instance_id: int, timeout: int = 300) -> int:
        """Upload the local IndicF5 reference clips to a running instance.

        The clips live in the examples tree, which is git-ignored, so a freshly cloned
        instance does NOT have them and its `/health` reports `indicf5_ready: false`. Seeding
        the directory the container was told about (CONTAINER_VOICES_DIR) makes the default
        narration path work with no manual step, and costs a few hundred kilobytes.

        Raises on an SSH failure. Callers that only need the common case to work should treat
        that as a warning: the health check that follows reports the real problem.
        """
        files = self.indicf5_voice_files()
        if not files:
            print(
                f"[Vast Manager Warning] No IndicF5 reference clips found in "
                f"{self.local_indicf5_voices_dir()}; the instance cannot narrate IndicF5.",
                file=sys.stderr,
            )
            return 0
        self.run_ssh(instance_id, f"mkdir -p {shlex.quote(CONTAINER_VOICES_DIR)}", timeout=timeout)
        for path in files:
            self.push_file(instance_id, path, f"{CONTAINER_VOICES_DIR}/{path.name}")
        print(
            f"[Vast Manager] Uploaded {len(files)} IndicF5 reference file(s) to "
            f"{CONTAINER_VOICES_DIR}."
        )
        return len(files)

    @staticmethod
    def container_env_spec() -> str:
        """Build the `--env` value for `vastai create instance`.

        `--env` is not only for environment variables: in the vastai CLI it is the whole
        docker-options field, so `-p` port mappings and `-e` variables travel together in this
        single string. The port mapping is mandatory (the client reaches /health and every
        inference endpoint through it), the voices directory must be pinned to the same path
        the client uploads to, and the token is mandatory too because IndicF5 is gated.
        """
        parts = ["-p 8000:8000", f"-e INDICF5_VOICES_DIR={CONTAINER_VOICES_DIR}"]
        token = hub_token()
        if token:
            parts.append(f"-e HF_TOKEN={token}")
        else:
            _warn_missing_hub_token()
        return " ".join(parts)

    def _safe_destroy(self, instance_id: int) -> None:
        """Destroy an instance without letting cleanup failure mask the real error."""
        try:
            self.destroy_instance(instance_id)
        except Exception as e:
            print(f"[Vast Manager Warning] Could not destroy {instance_id}: {e}", file=sys.stderr)

    def _safe_stop(self, instance_id: int) -> None:
        """Stop an instance, preserving its installed dependencies for a later run."""
        try:
            self.stop_instance(instance_id)
        except Exception as e:
            print(f"[Vast Manager Warning] Could not stop {instance_id}: {e}", file=sys.stderr)

    @classmethod
    def _idle_path(cls) -> Path:
        return ROOT / cls.IDLE_FILE

    def _load_idle(self) -> dict[str, float]:
        """Load the recorded stop time for each instance."""
        path = self._idle_path()
        if not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[Vast Manager Warning] Ignoring unreadable {path.name}: {e}", file=sys.stderr)
            return {}
        stopped = data.get("stopped", {}) if isinstance(data, dict) else {}
        out: dict[str, float] = {}
        for key, value in stopped.items():
            try:
                out[str(int(key))] = float(value)
            except (TypeError, ValueError):
                continue
        return out

    def _record_idle(self, instance_id: int) -> None:
        """Record that an instance was just stopped, so idle storage can be reclaimed."""
        path = self._idle_path()
        data = self._load_idle()
        data[str(int(instance_id))] = time.time()
        try:
            path.write_text(
                json.dumps({"stopped": data}, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except Exception as e:
            print(f"[Vast Manager Warning] Could not write {path.name}: {e}", file=sys.stderr)

    def _forget_idle(self, instance_id: int) -> None:
        """Drop an instance from the idle ledger once it is running or destroyed."""
        path = self._idle_path()
        data = self._load_idle()
        if str(int(instance_id)) not in data:
            return
        data.pop(str(int(instance_id)), None)
        try:
            path.write_text(
                json.dumps({"stopped": data}, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except Exception:
            pass

    def run_cli(self, args: list[str]) -> str:
        """Run a command using the vastai CLI tool."""
        # Look for vastai in the same directory as the python executable
        venv_bin = Path(sys.executable).parent
        cli = shutil.which("vastai") or shutil.which("vast") or str(venv_bin / "vastai.exe")
        
        if not cli or not Path(cli).exists():
            # Fallback to python module execution
            cmd = [sys.executable, "-m", "vastai"] + args
        else:
            cmd = [cli] + args

        if self.api_key:
            os.environ["VAST_API_KEY"] = self.api_key

        try:
            # stdin is closed so a command that unexpectedly prompts for confirmation
            # fails fast instead of blocking the whole run forever.
            res = subprocess.run(cmd, capture_output=True, text=True, check=True, stdin=subprocess.DEVNULL)
            return res.stdout.strip()
        except subprocess.CalledProcessError as e:
            err = e.stderr.strip() or e.stdout.strip()
            raise RuntimeError(f"Vast.ai CLI command failed: {err}") from e

    @staticmethod
    def server_dependency_packages(image: str | None = None) -> tuple[str, ...]:
        """The server's package set, with torchaudio pinned to the image's own torch version.

        Shared by the onstart install and `repair_dependencies` so a repair cannot drift
        away from the install that produced the problem in the first place.
        """
        image = image or os.environ.get("VAST_SERVER_IMAGE", DEFAULT_SERVER_IMAGE)
        torch_version = _image_torch_version(image)
        torchaudio_pin = f"torchaudio=={torch_version}" if torch_version else "torchaudio"
        return (*SERVER_DEPENDENCIES, torchaudio_pin)

    @classmethod
    def dependency_install_command(cls) -> str:
        """Shell snippet that installs the pinned server dependencies (uv, pip fallback).

        The torch family is resolved from the index matching the image's CUDA build via
        `--torch-backend`, and is never *upgraded* (there is no --upgrade here): the backend
        only decides which wheels are eligible, so the CUDA build of torch in the base image
        survives. This must stay byte-compatible with what the onstart runs, which is why
        both call it rather than each spelling out their own install line.
        """
        image = os.environ.get("VAST_SERVER_IMAGE", DEFAULT_SERVER_IMAGE)
        deps = " ".join(shlex.quote(package) for package in cls.server_dependency_packages(image))
        torch_backend = os.environ.get("VAST_UV_TORCH_BACKEND", "").strip() or _image_torch_backend(image)
        uv_backend_flag = f" --torch-backend={shlex.quote(torch_backend)}" if torch_backend else ""
        torch_index = (
            f"https://download.pytorch.org/whl/{torch_backend}" if torch_backend.startswith("cu") else ""
        )
        pip_index_args = (
            f"--index-url {torch_index} --extra-index-url https://pypi.org/simple " if torch_index else ""
        )
        return (
            "export PATH=\"$HOME/.local/bin:$PATH\"; "
            f"( uv pip install --system{uv_backend_flag} {deps} "
            f"|| python -m pip install {pip_index_args}{deps} )"
        )

    @classmethod
    def dependency_probe_command(cls, quiet: bool = False) -> str:
        """A `python -c` probe importing everything the server needs, lazy imports included.

        `quiet` sends stderr to /dev/null so the probe can gate a shell `if` without
        printing a traceback every time it is asked whether the packages are usable.
        """
        suffix = " 2>/dev/null" if quiet else ""
        return f"python -c {shlex.quote(SERVER_DEPENDENCY_PROBE)}{suffix}"

    @staticmethod
    def server_onstart_command(prefetch_models: list[str] | None = None) -> str:
        """Build the container start command.

        Setup dominates a fresh rental's cost, so the expensive work is run concurrently
        (apt, the Python deps, and the model prefetch are independent) and installed with
        uv rather than pip. uv is used *without* --upgrade so the CUDA build of torch in
        the base image is left untouched.

        The install marker lives in the container filesystem, not on the /workspace
        volume. That matters: if a stopped instance ever comes back with a reset
        container layer, a marker on the volume would claim the packages exist while
        site-packages is empty. The marker is also verified with a real import before it
        is trusted.

        Deliberately does NOT install the Remotion render stack (Node, Chromium libraries
        and fonts). That costs several minutes, only `--render-on vast` needs it, and
        keeping it out of this block means a reused instance can still acquire it later --
        see `render_stack_shell()`.
        """
        custom = os.environ.get("VAST_SERVER_ONSTART_CMD", "").strip()
        if custom:
            return custom
        repository = os.environ.get("VAST_SERVER_REPO_URL", "https://github.com/vivek09-bit/ai-video-gen.git")
        branch = os.environ.get("VAST_SERVER_GIT_REF", "main")
        # Use a token-authenticated URL if VAST_GITHUB_TOKEN is set (needed for private repos).
        token_repo = os.environ.get("VAST_GITHUB_TOKEN")
        if token_repo:
            repo_url = f"https://x-access-token:{token_repo}@github.com/vivek09-bit/ai-video-gen.git"
        else:
            repo_url = repository

        if prefetch_models is None:
            env_models = os.environ.get("VAST_SERVER_PREFETCH", "").strip()
            prefetch_models = (
                [item.strip() for item in env_models.split(",") if item.strip()]
                if env_models
                else list(DEFAULT_PREFETCH_MODELS)
            )

        deps_install = VastManager.dependency_install_command()
        # The probe covers the modules the server imports LAZILY as well as the top-level
        # packages; see SERVER_DEPENDENCY_PROBE for why that distinction cost a whole run.
        probe = VastManager.dependency_probe_command(quiet=True)
        prefetch = "; ".join(f"snapshot_download({model!r})" for model in prefetch_models)
        prefetch_cmd = f"python -c \"from huggingface_hub import snapshot_download; {prefetch}\""
        # torch is imported first and torchaudio second: a version mismatch raises an
        # unresolved-symbol OSError here, at setup, rather than fifteen minutes later at
        # synthesis where it is far harder to attribute. parler_tts and transformers are in
        # here for the same reason, and for a second one: whatever this probe accepts is
        # what the marker below certifies as "ready".
        verify_imports = (
            "python -c \"import numpy, torch, torchaudio, diffusers, transformers, soundfile, scipy, vocos; "
            "from f5_tts.infer.utils_infer import infer_process; "
            "assert torchaudio.__version__.split('+')[0] == torch.__version__.split('+')[0], "
            "'torch and torchaudio versions must match'; "
            "print('[onstart] IndicF5 imports OK', torch.__version__, torchaudio.__version__, "
            "'numpy', numpy.__version__, 'transformers', transformers.__version__)\" "
            "|| echo '[onstart] FAILED: IndicF5 dependency import failed'"
        )
        verify_torch = (
            "python -c \"import torch; print('[onstart] torch', torch.__version__, "
            "'cuda', torch.version.cuda, 'available', torch.cuda.is_available())\""
        )

        apt_packages = ["git", "ffmpeg", "libsndfile1", "ca-certificates", "curl"]
        apt_install = " ".join(apt_packages)
        # A warm-reused instance keeps whatever clone it was created with, so it would
        # otherwise run month-old server code and never receive a fix. Best effort: when the
        # fetch fails the existing checkout simply stays in place.
        branch_q = shlex.quote(branch)
        refresh_repo = (
            "if [ -d tube2dscript/.git ]; then "
            f"git -C tube2dscript fetch --depth 1 origin {branch_q} >/dev/null 2>&1 "
            "&& git -C tube2dscript reset --hard FETCH_HEAD >/dev/null 2>&1; "
            f"else git clone --depth 1 --branch {branch_q} {shlex.quote(repo_url)} tube2dscript; "
            "fi || true; "
        )
        return (
            "mkdir -p /workspace; exec > /workspace/onstart.log 2>&1; set -x; cd /workspace; "
            "if [ -f \"$HOME/.tube2dscript_deps_ready\" ] "
            f"&& {probe}; then "
            "echo '[onstart] dependencies already present, skipping install'; "
            "else "
            # The AI4Bharat f5_tts dependency is installed from Git, so package installation
            # must wait for apt to provide git; otherwise a cold host races and fails randomly.
            f"( apt-get update && apt-get install -y {apt_install} || true ) & APT_PID=$!; "
            "( curl -LsSf https://astral.sh/uv/install.sh | sh ) || true; "
            "wait $APT_PID; echo '[onstart] apt done'; "
            "command -v git >/dev/null 2>&1 || { echo '[onstart] FAILED: git is required for IndicF5'; exit 1; }; "
            f"( {deps_install} ) & PIP_PID=$!; "
            "wait $PIP_PID; echo '[onstart] python deps done'; "
            # snapshot_download needs huggingface_hub, which the dependency install provides.
            # Running it sooner races a cold image and wastes a misleading failed prefetch.
            f"( {prefetch_cmd} ) & MODEL_PID=$!; "
            "wait $MODEL_PID; echo '[onstart] model prefetch done'; "
            f"{verify_imports}; "
            f"{verify_torch}; "
            f"if {probe}; then "
            "touch \"$HOME/.tube2dscript_deps_ready\"; "
            "else echo '[onstart] FAILED: deps unusable, not marking them ready'; fi; "
            "fi; "
            f"{refresh_repo} "
            "cd /workspace/tube2dscript; "
            "echo 'starting server at $(hostname -i):8000'; "
            "nohup python scripts/vast_server.py > /workspace/server.log 2>&1 & "
            "sleep 5; echo '--- server.log after start ---'; tail -30 /workspace/server.log; "
            "echo 'onstart done, keeping container alive'; sleep infinity"
        )

    @staticmethod
    def render_stack_shell() -> str:
        """Idempotent shell snippet that makes an instance able to run `npx remotion render`.

        This is separate from `server_onstart_command` on purpose. The onstart only runs at
        container creation, so anything inside its dependency marker would never be added to
        an already-created instance -- which is exactly what warm reuse hands back. Running
        this over SSH before a remote render means the render stack can be added whenever it
        is needed, to whatever instance is in use.

        Node comes from NodeSource, the Chromium libraries and Noto fonts come from apt, and
        the Noto fonts matter for correctness rather than looks: the theme fonts are Windows
        names (Georgia, Segoe UI) that do not exist on Linux, so Devanagari captions would
        otherwise render as empty boxes.
        """
        packages = " ".join(RENDER_STACK_PACKAGES)
        return (
            "if command -v node >/dev/null 2>&1 && [ -d /workspace/tube2dscript/node_modules ]; then "
            "echo '[render] stack already present'; exit 0; fi; "
            f"apt-get update && apt-get install -y {packages} || echo '[render] WARNING: apt failed'; "
            "if ! command -v node >/dev/null 2>&1; then "
            "(curl -fsSL https://deb.nodesource.com/setup_20.x | bash -) && apt-get install -y nodejs "
            "|| echo '[render] WARNING: node install failed'; fi; "
            "command -v node >/dev/null 2>&1 || { echo '[render] ERROR: node is unavailable'; exit 1; }; "
            "cd /workspace/tube2dscript || { echo '[render] ERROR: repo missing'; exit 1; }; "
            "[ -d node_modules ] || ( npm ci || npm install ) "
            "|| { echo '[render] ERROR: npm install failed'; exit 1; }; "
            "npx remotion browser ensure || { echo '[render] ERROR: browser ensure failed'; exit 1; }; "
            "echo '[render] stack ready'"
        )

    def search_gpu(
        self,
        gpu_name: str,
        max_dph: float | None = None,
        min_disk: int = 40,
        min_inet_down: int | None = None,
        min_disk_bw: int | None = None,
    ) -> list[dict]:
        """Search for a single GPU of the requested model inside the budget ceiling.

        Setup time is dominated by downloading, so fast hosts are preferred. The speed
        filter is applied in tiers and relaxed automatically: a strict floor can match
        zero offers, which would block renting entirely.

        Only the SPEED preference is ever relaxed. One GPU, the disk floor, the RAM floor,
        the direct-port floor and the price ceiling are structural and apply to every tier
        including the last-resort query, because an offer that cannot host the job costs more
        than no offer at all: the failure would otherwise surface far later, as a confusing
        start-up error on an instance that was never viable. The direct-port floor is here
        because every instance is created with `--direct` (SSH and the exposed inference port
        both depend on it), and a host that advertises none would fail at create time.
        """
        limit = gpu_budget(gpu_name) if max_dph is None else max_dph
        if min_inet_down is None:
            min_inet_down = int(os.environ.get("VAST_MIN_INET_DOWN", "1000"))
        if min_disk_bw is None:
            min_disk_bw = int(os.environ.get("VAST_MIN_DISK_BW", "500"))
        gpu_filter = gpu_name.upper().replace("NVIDIA GEFORCE ", "").replace(" ", "_")

        tiers: list[tuple[int, int, str]] = []
        if min_inet_down > 100 or min_disk_bw > 0:
            tiers.append((min_inet_down, min_disk_bw, "fast host"))
        tiers.append((100, 0, "any host"))

        offers: list[dict] = []
        for index, (inet, disk_bw, label) in enumerate(tiers):
            bw_clause = f" disk_bw>={disk_bw}" if disk_bw else ""
            query = (
                f"gpu_name={gpu_filter} num_gpus=1 rented=False dph<{limit} "
                f"disk_space>={min_disk} direct_port_count>0 inet_down>{inet} cpu_ram>=32{bw_clause}"
            )
            try:
                offers = json.loads(self.run_cli(["search", "offers", query, "-o", "dph+", "--raw"]))
            except Exception as e:
                print(f"[Vast Manager Warning] {gpu_name} offer search failed ({e})", file=sys.stderr)
                offers = []
            if offers:
                if index > 0:
                    print(f"[Vast Manager] Relaxed the speed filter to '{label}' to find a {gpu_name}.")
                break
            if index < len(tiers) - 1:
                print(
                    f"[Vast Manager] No {gpu_name} offers matched the {label} filter "
                    f"(>={inet} Mbps, disk >={disk_bw} MB/s); relaxing."
                )

        if not offers:
            # Last resort: relax only the speed preference. The structural constraints and
            # the price ceiling stay in the query, so this cannot hand back an offer that
            # is too small for the job. Returning nothing lets the caller fail loudly.
            print(
                f"[Vast Manager] No {gpu_name} offer matched the speed tiers; retrying "
                f"without the inet/disk-bandwidth preference."
            )
            try:
                offers = json.loads(
                    self.run_cli(
                        [
                            "search",
                            "offers",
                            f"gpu_name={gpu_filter} num_gpus=1 rented=False dph<{limit} "
                            f"disk_space>={min_disk} direct_port_count>0 cpu_ram>=32",
                            "-o",
                            "dph+",
                            "--raw",
                        ]
                    )
                )
            except Exception as e:
                print(
                    f"[Vast Manager Warning] {gpu_name} fallback offer search failed ({e})",
                    file=sys.stderr,
                )
                return []

        # Vast filters on `dph`, which excludes storage and bandwidth. `dph_total`
        # is the rate that actually gets billed, so enforce the ceiling on that too.
        affordable = [
            offer
            for offer in offers
            if hourly_rate(offer) <= limit and str(int(offer.get("machine_id") or 0)) not in self.bad_hosts
        ]
        priced_out = len(offers) - len([o for o in offers if hourly_rate(o) <= limit])
        if priced_out:
            print(
                f"[Vast Manager] Dropped {priced_out} {gpu_name} offer(s) priced above "
                f"the ${limit:.2f}/hr all-in ceiling."
            )
        blocked = len([o for o in offers if hourly_rate(o) <= limit]) - len(affordable)
        if blocked:
            print(f"[Vast Manager] Skipped {blocked} {gpu_name} offer(s) on blacklisted machines.")
        affordable.sort(key=hourly_rate)
        return affordable

    def search_rtx_4090(self, max_dph: float | None = None, min_disk: int = 40) -> list[dict]:
        """Backwards-compatible search for RTX 4090 offers."""
        return self.search_gpu("RTX 4090", max_dph=max_dph, min_disk=min_disk)

    def search_rtx_3090(self, max_dph: float | None = None, min_disk: int = 40) -> list[dict]:
        """Search for RTX 3090 24GB offers."""
        return self.search_gpu("RTX 3090", max_dph=max_dph, min_disk=min_disk)

    def _offer_wait(self, seconds: float) -> None:
        """Sleep between offer searches (seam so tests need not really wait)."""
        time.sleep(seconds)

    def search_gpu_with_wait(
        self,
        gpu_name: str,
        max_dph: float | None = None,
        min_disk: int = 40,
        wait_sec: float | None = None,
        attempts: int | None = None,
    ) -> list[dict]:
        """Search for offers, re-trying while the marketplace momentarily has none.

        A live marketplace can transiently report zero matches -- hosts fill up, or the
        strict speed tier happens to exclude everything. Aborting on the first empty
        result would kill a run that would have succeeded seconds later.

        Deliberately NOT part of `search_gpu`: that is also used by the CLI `search`
        command and by tests, neither of which should block for up to a minute.
        """
        if wait_sec is None:
            wait_sec = float(os.environ.get("VAST_OFFER_WAIT_SECONDS", "15"))
        if attempts is None:
            attempts = int(os.environ.get("VAST_OFFER_SEARCH_ATTEMPTS", "4"))
        attempts = max(1, attempts)

        for attempt in range(1, attempts + 1):
            offers = self.search_gpu(gpu_name, max_dph=max_dph, min_disk=min_disk)
            if offers:
                if attempt > 1:
                    print(f"[Vast Manager] Found {len(offers)} {gpu_name} offer(s) on attempt {attempt}.")
                return offers
            if attempt < attempts:
                print(
                    f"[Vast Manager] No {gpu_name} offer available right now; waiting "
                    f"{wait_sec:g}s before re-searching (attempt {attempt}/{attempts})..."
                )
                self._offer_wait(wait_sec)
        print(f"[Vast Manager] Still no {gpu_name} offers after {attempts} attempt(s).", file=sys.stderr)
        return []

    def create_instance(
        self,
        offer_id: int | None = None,
        disk_gb: int = 50,
        onstart_cmd: str | None = None,
        gpu_name: str = "RTX 4090",
        max_dph: float | None = None,
        offer_rate: float | None = None,
        prefetch_models: list[str] | None = None,
        image: str | None = None,
    ) -> int:
        """Create a one-GPU Vast.ai instance of the requested model inside the budget."""
        limit = gpu_budget(gpu_name) if max_dph is None else max_dph
        if image is None:
            image = os.environ.get("VAST_SERVER_IMAGE", DEFAULT_SERVER_IMAGE)
        rate: float | None = None
        explicit_offer = offer_id is not None
        if offer_id is None:
            offers = self.search_gpu_with_wait(gpu_name, max_dph=limit, min_disk=disk_gb)
            if not offers:
                raise RuntimeError(
                    f"No available {gpu_name} offers under ${limit:.2f}/hr with at least {disk_gb}GB disk."
                )
            best = offers[0]
            offer_id = int(best["id"])
            rate = hourly_rate(best)
            print(
                f"[Vast Manager] Selected cheapest {gpu_name} offer: {offer_id} "
                f"({format_rate(rate)}, budget ${limit:.2f}/hr)"
            )
        else:
            rate = offer_rate
            if rate is None:
                # Vast's offer search has no `id=` field, and an offer already rented by
                # this account disappears from results, so an explicit offer cannot be
                # priced up front. The billed rate is verified once it exists instead.
                print(f"[Vast Manager] Offer {offer_id} cannot be priced up front; verifying after creation...")
            elif rate > limit:
                raise RuntimeError(
                    f"Offer {offer_id} costs ${rate:.3f}/hr, above the ${limit:.2f}/hr {gpu_name} budget."
                )
            else:
                print(f"[Vast Manager] Offer {offer_id} is ${rate:.3f}/hr (budget ${limit:.2f}/hr)")

        print(f"[Vast Manager] Launching instance on offer {offer_id} with {disk_gb}GB disk...")
        args = [
            "create",
            "instance",
            str(offer_id),
            "--image",
            image,
            "--disk",
            str(disk_gb),
            "--env",
            self.container_env_spec(),
            "--ssh",
            "--direct",
            "--onstart-cmd",
            onstart_cmd or self.server_onstart_command(prefetch_models),
            "--raw",
        ]

        raw = self.run_cli(args)
        res = json.loads(raw)
        instance_id = int(res.get("new_contract") or res.get("id"))
        print(f"[Vast Manager] Instance created successfully. ID: {instance_id}")

        if explicit_offer and rate is None:
            # Could not price the offer before renting, so check the billed rate now and
            # undo the rental if it turns out to be over budget.
            billed = hourly_rate(self.get_instance_info(instance_id))
            if billed == float("inf"):
                print(
                    f"[Vast Manager Warning] Vast reported no price for instance {instance_id}; "
                    f"could not verify it against the ${limit:.2f}/hr budget.",
                    file=sys.stderr,
                )
            elif billed > limit:
                self.destroy_instance(instance_id)
                raise RuntimeError(
                    f"Instance {instance_id} bills at ${billed:.3f}/hr, above the ${limit:.2f}/hr "
                    f"{gpu_name} budget; it has been destroyed."
                )
            else:
                print(f"[Vast Manager] Instance {instance_id} bills at ${billed:.3f}/hr (budget ${limit:.2f}/hr)")
        return instance_id

    def provision_instance(
        self,
        gpu_name: str,
        disk_gb: int = 50,
        attempts: int = 3,
        max_dph: float | None = None,
        prefetch_models: list[str] | None = None,
    ) -> tuple[int, str]:
        """Rent a GPU, moving to the next offer when a host cannot pull the image.

        The cheapest in-budget offers are tried in turn. A host that reports a fatal
        start-up error (e.g. a Docker registry TLS timeout) is destroyed and blacklisted.
        A host that is merely *slow* is stopped and kept instead, because destroying it
        would throw away the dependency install it has already paid for; the caller gets a
        resume hint rather than quietly burning a second setup.
        """
        limit = gpu_budget(gpu_name) if max_dph is None else max_dph
        offers = self.search_gpu_with_wait(gpu_name, max_dph=limit, min_disk=disk_gb)
        if not offers:
            raise RuntimeError(
                f"No available {gpu_name} offers under ${limit:.2f}/hr with at least {disk_gb}GB "
                f"disk after repeated searches."
            )
        total = min(attempts, len(offers))
        last_error: Exception | None = None
        for index, offer in enumerate(offers[:total], start=1):
            offer_id = int(offer["id"])
            machine_id = offer.get("machine_id")
            rate = hourly_rate(offer)
            print(
                f"[Vast Manager] Attempt {index}/{total}: offer {offer_id} "
                f"({format_rate(rate)}, machine {machine_id})"
            )
            # The create itself is inside the retry. An offer can be taken by someone else
            # between the search and the rent, and a host that advertises direct ports can
            # still refuse a --direct create; both are offer-specific, so the next offer is
            # the fix rather than aborting a provisioning loop the caller asked to be resilient.
            # The host is deliberately NOT blacklisted here: a lost race says nothing about
            # the machine, and blacklisting would retire a good cheap host permanently.
            try:
                instance_id = self.create_instance(
                    offer_id=offer_id,
                    disk_gb=disk_gb,
                    gpu_name=gpu_name,
                    max_dph=limit,
                    offer_rate=rate,
                    prefetch_models=prefetch_models,
                )
            except Exception as create_error:
                last_error = create_error
                print(
                    f"[Vast Manager Warning] Could not create an instance on offer {offer_id} "
                    f"(machine {machine_id}): {create_error}",
                    file=sys.stderr,
                )
                continue
            try:
                return instance_id, self.wait_until_ready(instance_id)
            except InstanceBroken as error:
                last_error = error
                print(f"[Vast Manager Warning] Instance {instance_id} is broken: {error}", file=sys.stderr)
                print(f"[Vast Manager] Destroying instance {instance_id} to stop storage charges...")
                self._safe_destroy(instance_id)
                if machine_id is not None:
                    self.mark_bad_host(machine_id, str(error))
            except InstanceSlow as error:
                last_error = error
                print(f"[Vast Manager Warning] Instance {instance_id} is still installing: {error}", file=sys.stderr)
                print(f"[Vast Manager] Stopping instance {instance_id} instead of destroying it, so the")
                print(f"[Vast Manager] dependencies it already installed are preserved for reuse.")
                self._safe_stop(instance_id)
                raise RuntimeError(
                    f"Instance {instance_id} needs longer to finish its setup. It has been "
                    f"stopped (not destroyed) and its install is intact. Re-run the same "
                    f"command with --vast-instance {instance_id} to resume it."
                ) from error
            except Exception:
                # Anything else is likely our own bug; do not destroy capacity over it.
                print(f"[Vast Manager Warning] Unexpected error waiting on instance {instance_id}.", file=sys.stderr)
                raise
        raise RuntimeError(
            f"Could not rent a working {gpu_name} instance after {total} attempt(s): {last_error}"
        ) from last_error

    def start_instance(self, instance_id: int) -> bool:
        """Start an existing stopped instance."""
        print(f"[Vast Manager] Starting instance {instance_id}...")
        self.run_cli(["start", "instance", str(instance_id)])
        self._forget_idle(instance_id)
        return True

    def list_instances(self) -> list[dict]:
        """Return every instance this account owns, or [] if the query fails."""
        try:
            return json.loads(self.run_cli(["show", "instances", "--raw"]))
        except Exception as e:
            print(f"[Vast Manager Warning] Failed to list instances: {e}", file=sys.stderr)
            return []

    def conflicting_instances(self, expected_gpu: str) -> list[dict]:
        """Return reusable instances of a DIFFERENT GPU model that we already own.

        Vast cannot change an instance's GPU type, so a leftover instance of the wrong
        model both blocks a clean reuse and keeps billing storage while a second instance
        is rented. Reporting them lets the caller decide instead of silently paying twice.
        """
        wanted = expected_gpu.upper().replace("NVIDIA GEFORCE ", "").replace(" ", "")
        conflicts: list[dict] = []
        for inst in self.list_instances():
            status = str(inst.get("actual_status") or inst.get("status") or "").lower()
            if status not in ("running", "stopped", "created", "exited"):
                continue
            gpu_name = str(inst.get("gpu_name") or "unknown").upper().replace("NVIDIA GEFORCE ", "")
            if wanted in gpu_name.replace(" ", ""):
                continue
            conflicts.append(
                {
                    "id": int(inst.get("id") or 0),
                    "gpu_name": gpu_name,
                    "status": status,
                    "disk_space": int(float(inst.get("disk_space") or 0)),
                }
            )
        return conflicts

    def find_existing_instance(
        self,
        expected_gpu: str,
        min_disk: int = 50,
        max_dph: float | None = None,
        destroy_broken: bool = True,
    ) -> tuple[int | None, str | None]:
        """Reuse an already-rented instance of the right GPU that still fits the budget.

        Instances left behind by a failed host (as opposed to ones we deliberately
        stopped) can never be reused, so they are destroyed by default instead of
        being skipped while they keep billing for disk.
        """
        limit = gpu_budget(expected_gpu) if max_dph is None else max_dph
        print(f"[Vast Manager] Searching for a reusable {expected_gpu} instance (budget ${limit:.2f}/hr)...")
        try:
            instances = json.loads(self.run_cli(["show", "instances", "--raw"]))
        except Exception as e:
            print(f"[Vast Manager Warning] Failed to query existing instances: {e}", file=sys.stderr)
            return None, None

        wanted = expected_gpu.upper().replace("NVIDIA GEFORCE ", "").replace(" ", "")
        for inst in instances:
            instance_id = int(inst.get("id") or 0)
            status = str(inst.get("actual_status") or inst.get("status") or "").lower()
            gpu_name = str(inst.get("gpu_name") or "unknown").upper().replace("NVIDIA GEFORCE ", "")
            disk_space = int(float(inst.get("disk_space") or 0))
            rate = hourly_rate(inst)
            status_msg = str(inst.get("status_msg") or "")

            if wanted not in gpu_name.replace(" ", ""):
                continue

            machine_key = str(int(inst.get("machine_id") or 0))
            if machine_key in self.bad_hosts:
                print(f"  Skipping instance {instance_id}: machine {machine_key} is blacklisted.")
                continue

            if status in ("error", "offline") or "error response from daemon" in status_msg.lower():
                print(f"  Instance {instance_id} ({gpu_name}) is unusable [{status or 'error'}]: {status_msg or 'no status message'}")
                if destroy_broken:
                    print(f"  Destroying broken instance {instance_id} so it stops billing storage...")
                    try:
                        self.destroy_instance(instance_id)
                    except Exception as e:
                        print(f"  [Vast Manager Warning] Could not destroy {instance_id}: {e}", file=sys.stderr)
                continue

            if status not in ("running", "stopped", "created", "exited"):
                print(f"  Skipping instance {instance_id}: status {status!r} is not reusable yet.")
                continue
            if disk_space < min_disk:
                print(f"  Skipping instance {instance_id}: {disk_space}GB disk is below the {min_disk}GB minimum.")
                continue
            if rate > limit:
                print(f"  Skipping instance {instance_id}: {format_rate(rate)} exceeds the ${limit:.2f}/hr budget.")
                continue

            server_url = self._resolve_server_url(inst) if status == "running" else None
            print(
                f"  Reusing instance {instance_id} (status={status}, GPU={gpu_name}, "
                f"disk={disk_space}GB, {format_rate(rate)})"
            )
            return instance_id, server_url
        return None, None

    def stop_instance(self, instance_id: int) -> bool:
        """Stop an instance to halt GPU compute billing."""
        print(f"[Vast Manager] Stopping instance {instance_id} to prevent idle charges...")
        self.run_cli(["stop", "instance", str(instance_id)])
        self._record_idle(instance_id)
        return True

    def destroy_instance(self, instance_id: int) -> bool:
        """Permanently delete an instance and release its disk."""
        print(f"[Vast Manager] Destroying instance {instance_id}...")
        # -y is required: without it the CLI waits for a confirmation prompt.
        self.run_cli(["destroy", "instance", str(instance_id), "-y"])
        return True

    def repair_dependencies(self, instance_id: int, timeout: int = 2700) -> str:
        """Reapply the pinned dependency install on an existing instance, over SSH.

        A warm-reused instance short-circuits its install behind the marker, so when that
        install turns out to be broken -- a wheel resolved against the wrong NumPy, say --
        the instance can never repair itself: the marker keeps claiming the packages are
        fine while every request fails. This clears the marker, reinstalls with the very
        same pinned set the onstart uses, and only restores the marker once the probe
        passes. Destroying and re-renting would also work, but this preserves an install
        that has already been paid for.

        Needs working SSH to the instance. Note that this cannot be done by hand either:
        the onstart command is baked in when an instance is CREATED, so a client-side fix
        to it -- including the pinned dependency list -- never reaches an existing
        instance. Only a repair over SSH, or a fresh rental, changes its environment.
        """
        install = VastManager.dependency_install_command()
        probe = VastManager.dependency_probe_command()
        command = (
            "cd /workspace/tube2dscript 2>/dev/null || cd /workspace; "
            "rm -f \"$HOME/.tube2dscript_deps_ready\"; "
            f"( {install} ) && {probe} "
            "&& touch \"$HOME/.tube2dscript_deps_ready\" "
            "&& echo '[repair] dependencies verified and marked ready'"
        )
        print(f"[Vast Manager] Repairing dependencies on instance {instance_id} over SSH...")
        output = self.run_ssh(instance_id, command, timeout=timeout)
        print(output.strip() or "[Vast Manager] Repair produced no output.")
        return output

    def prune_instances(self, dry_run: bool = False, idle_hours: float | None = None) -> list[int]:
        """Destroy rented instances that can never be reused.

        Always removes broken hosts and over-budget instances. When `idle_hours` is given,
        also removes instances that have been stopped for longer than that, because a
        stopped instance still bills storage ($0.77/day for a 50GB disk) while its
        finished install sits unused.
        """
        try:
            instances = json.loads(self.run_cli(["show", "instances", "--raw"]))
        except Exception as e:
            print(f"[Vast Manager Warning] Failed to list instances: {e}", file=sys.stderr)
            return []

        removed: list[int] = []
        if not instances:
            print("[Vast Manager] No rented instances to prune.")
            return removed

        idle_ledger = self._load_idle()
        now = time.time()
        for inst in instances:
            instance_id = int(inst.get("id") or 0)
            status = str(inst.get("actual_status") or inst.get("status") or "").lower()
            gpu_name = str(inst.get("gpu_name") or "unknown")
            status_msg = str(inst.get("status_msg") or "")
            rate = hourly_rate(inst)
            limit = gpu_budget(gpu_name)
            broken = status in ("error", "offline") or "error response from daemon" in status_msg.lower()
            over_budget = rate > limit

            idle_for: float | None = None
            stopped_at = idle_ledger.get(str(instance_id))
            if stopped_at is not None and status in ("stopped", "exited"):
                idle_for = (now - stopped_at) / 3600.0
            is_idle = idle_hours is not None and idle_for is not None and idle_for >= idle_hours

            if not broken and not over_budget and not is_idle:
                detail = f", idle {idle_for:.1f}h" if idle_for is not None else ""
                print(f"  Keeping {instance_id} ({gpu_name}, {status or 'unknown'}, {format_rate(rate)}{detail})")
                continue

            if broken:
                reason = "broken host / image pull failure"
            elif over_budget:
                reason = f"{format_rate(rate)} is above the ${limit:.2f}/hr budget"
            else:
                reason = f"stopped for {idle_for:.1f}h (limit {idle_hours}h), still billing storage"

            if dry_run:
                print(f"  [dry-run] Would destroy {instance_id} ({gpu_name}): {reason}")
                continue
            print(f"  Destroying {instance_id} ({gpu_name}): {reason}")
            try:
                self.destroy_instance(instance_id)
                removed.append(instance_id)
                idle_ledger.pop(str(instance_id), None)
            except Exception as e:
                print(f"  [Vast Manager Warning] Could not destroy {instance_id}: {e}", file=sys.stderr)

        if removed and not dry_run:
            path = self._idle_path()
            try:
                path.write_text(
                    json.dumps({"stopped": idle_ledger}, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
            except Exception:
                pass
        return removed

    def get_instance_info(self, instance_id: int) -> dict[str, Any]:
        """Fetch status and connection info for a specific instance."""
        try:
            raw = self.run_cli(["show", "instances", "--raw"])
            instances = json.loads(raw)
            for inst in instances:
                if int(inst.get("id", 0)) == int(instance_id):
                    return inst
        except Exception as e:
            print(f"[Vast Manager Warning] Failed to get instance status: {e}", file=sys.stderr)
        return {}

    def _resolve_server_url(self, info: dict) -> str:
        """Build the public URL from the host port Vast mapped to container port 8000."""
        ip = (info.get("public_ipaddr") or "").strip()
        mapping = (info.get("ports") or {}).get("8000/tcp") or []
        if ip and mapping:
            return f"http://{ip}:{mapping[0]['HostPort']}"
        return ""

    def wait_until_ready(
        self,
        instance_id: int,
        timeout_sec: int | None = None,
        container_timeout_sec: int | None = None,
    ) -> str:
        """Wait until the inference server answers /health.

        Three phases. The container must reach 'running' within `container_timeout_sec`,
        otherwise the host is treated as broken. The onstart install then has until the
        hard cap (`timeout_sec`) to boot the server.

        Inside that budget the wait is *progress*-based rather than a flat wall clock: as
        long as the instance is visibly still doing work (its disk usage keeps growing)
        the deadline is extended, and only `VAST_SERVER_STALL_TIMEOUT` seconds of no
        progress aborts. A fixed 30-minute cutoff used to destroy healthy machines that
        simply had a slow package index.
        """
        if timeout_sec is None:
            timeout_sec = int(os.environ.get("VAST_SERVER_TIMEOUT", "3600"))
        if container_timeout_sec is None:
            container_timeout_sec = int(os.environ.get("VAST_CONTAINER_TIMEOUT", "600"))
        stall_timeout = int(os.environ.get("VAST_SERVER_STALL_TIMEOUT", "600"))
        # A real install moves gigabytes. Log files and overlay churn move a few megabytes,
        # so progress needs a floor: without one, jitter resets the stall timer and a host
        # that installed nothing looks merely slow.
        min_progress_gb = float(os.environ.get("VAST_PROGRESS_MIN_DELTA_GB", "0.05"))
        # Growth below this across the whole wait means the onstart never installed anything.
        install_floor_gb = float(os.environ.get("VAST_INSTALL_FLOOR_GB", "1.5"))
        # How long a host gets to move that much *after its container starts* before it is
        # written off. Measured from container start on purpose: image-pull time is not
        # install time, and counting it here declared a working host dead at 303s.
        install_grace_sec = int(os.environ.get("VAST_INSTALL_GRACE_SECONDS", "300"))
        # Consecutive polls that must report the container dead before the host is written
        # off, so one blip in the status API cannot destroy a rented instance.
        dead_polls_needed = int(os.environ.get("VAST_DEAD_POLLS", "2"))
        print(
            f"[Vast Manager] Waiting for instance {instance_id} and its inference server "
            f"(container up to {container_timeout_sec}s, stall {stall_timeout}s, cap {timeout_sec}s)..."
        )
        start_t, container_up_at, last_msg = time.time(), None, ""
        last_progress_at = start_t
        last_disk: float | None = None
        peak_disk: float | None = None
        # The image pull happens before the container exists, so it gets its own deadline:
        # extended while docker's status text keeps changing, bounded by the overall cap.
        pull_deadline = start_t + container_timeout_sec
        last_status_msg = ""
        # True once a real (non-negative) disk reading has been seen. Deliberately NOT
        # latched during the unknown phase: Vast reports -1.0 until the container is up, so
        # latching "no signal" there would disable stall detection for the entire run.
        progress_signal = False
        stall_note = ""
        dead_polls = 0
        while True:
            now = time.time()
            elapsed = now - start_t
            if elapsed > timeout_sec:
                self._diagnose_boot(instance_id, f"still installing after the {timeout_sec}s cap")
                raise InstanceSlow(
                    f"instance {instance_id} was still installing after the {timeout_sec}s cap "
                    f"(last status {last_msg or 'unknown'}). The log is printed above."
                )
            info = self.get_instance_info(instance_id)
            status = info.get("actual_status") or ""
            status_msg = str(info.get("status_msg") or "")
            disk_used = self._disk_used_gb(info)
            if disk_used is not None:
                if not progress_signal:
                    progress_signal = True
                    stall_note = ""  # a real reading arrived, so stall detection is live
            elif container_up_at is not None and not stall_note:
                stall_note = " (this API does not report disk usage, so stalls cannot be detected)"
            # A host that cannot pull the image will never become healthy; fail fast
            # so the caller can destroy it and try a different offer. "No such container"
            # is excluded on purpose: Vast reports it transiently while (re)creating a
            # container, and treating it as fatal would destroy a healthy host.
            fatal_status = status in ("error", "offline") or (
                "error response from daemon" in status_msg.lower()
                and "no such container" not in status_msg.lower()
            )
            if fatal_status:
                raise InstanceBroken(
                    f"instance {instance_id} failed to start [{status or 'error'}]: {status_msg or 'no status message'}"
                )
            # The pull is visible only through docker's progress text, which changes as
            # layers complete (observed at 49s, 173s, 186s, 211s on a 291s pull). Extend the
            # container deadline while that keeps happening, bounded by the overall cap, so
            # a slow-but-working registry is not written off mid-pull. Vast reports no
            # usable disk figure until the container exists, so this is the only liveness
            # signal available during this phase.
            if container_up_at is None and status_msg and status_msg != last_status_msg:
                pull_deadline = min(max(pull_deadline, now + stall_timeout), start_t + timeout_sec)
                last_status_msg = status_msg
            # A container that reached 'running' and then died will never serve the API: its
            # onstart crashed. Waiting out the stall budget here wastes ten minutes and then
            # re-admits the failure as "slow", so the host gets stopped and kept instead of
            # destroyed. Two consecutive polls guard against a transient API blip.
            if status in ("exited", "stopped") and container_up_at is not None:
                dead_polls += 1
                if dead_polls >= dead_polls_needed:
                    raise InstanceBroken(
                        f"instance {instance_id} container was running and then went {status!r} "
                        f"{int(now - container_up_at)}s after it first came up, so its onstart "
                        f"crashed and the inference server can never answer "
                        f"(last status {last_msg or 'unknown'}; {status_msg or 'no status message'})."
                    )
            else:
                dead_polls = 0
            if status == "running":
                if container_up_at is None:
                    container_up_at = now
                    # The install clock starts here, not at create time: a cold pull of
                    # pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime measured 291s, which
                    # alone exhausted a 300s install grace and made a working host look
                    # dead. Reset the baseline too, so the pull's own bytes are not counted
                    # as install progress.
                    last_progress_at = now
                    last_disk = None
                    peak_disk = None
            elif container_up_at is None and now > pull_deadline:
                # Still created/loading with no container after the (progress-extended)
                # deadline, e.g. a host whose registry is unreachable and reports no error.
                raise InstanceBroken(
                    f"instance {instance_id} container never reached 'running' within "
                    f"{container_timeout_sec}s (stuck at {status or 'unknown'!r}): "
                    f"{status_msg or 'no status message'}"
                )
            # Progress must be *meaningful*. Counting any increase let a few megabytes of
            # log-file churn reset the stall timer indefinitely: a host that had installed
            # nothing kept a run alive for 22 minutes on a 600s budget.
            if disk_used is not None:
                if peak_disk is None or disk_used > peak_disk:
                    peak_disk = disk_used
                if last_disk is None or disk_used >= last_disk + min_progress_gb:
                    last_progress_at = now
                    last_disk = disk_used
            # Both verdicts below are gated on the container having started: during the image
            # pull there is no install yet, so "no install progress" carries no information.
            if container_up_at is not None and progress_signal and now - last_progress_at > stall_timeout:
                machine_id = info.get("machine_id")
                self._diagnose_boot(instance_id, f"no install progress for {stall_timeout}s")
                if (peak_disk is None or peak_disk < install_floor_gb) and machine_id is not None:
                    # The disk never moved beyond log-file noise, so nothing was installed.
                    # Stopping this instance would preserve an empty install and keep billing
                    # storage for it, so treat the host as broken instead of merely slow.
                    self.mark_bad_host(machine_id, "onstart installed nothing")
                    raise InstanceBroken(
                        f"instance {instance_id} installed nothing in {int(elapsed)}s "
                        f"(peak disk {peak_disk}GB). Host {machine_id} blacklisted; the "
                        "install log is printed above."
                    )
                raise InstanceSlow(
                    f"instance {instance_id} made no install progress for {stall_timeout}s "
                    f"(disk stuck at {last_disk}GB, last status {last_msg or 'unknown'}). "
                    "Its install is intact; re-run with --vast-instance "
                    f"{instance_id} to resume it. The log is printed above."
                )
            # Fail fast instead of waiting out the stall budget. A host whose disk has barely
            # moved after the grace period never started installing: the onstart swallows
            # every failure with `|| true`, so this is the only signal that it ran at all.
            if (
                container_up_at is not None
                and progress_signal
                and now - container_up_at > install_grace_sec
                and (peak_disk is None or peak_disk < install_floor_gb)
            ):
                machine_id = info.get("machine_id")
                self._diagnose_boot(
                    instance_id,
                    f"installed {peak_disk}GB in {int(now - container_up_at)}s after the container started",
                )
                if machine_id is not None:
                    self.mark_bad_host(machine_id, "onstart installed nothing")
                raise InstanceBroken(
                    f"instance {instance_id} installed nothing in {int(now - container_up_at)}s "
                    f"after its container started (peak disk {peak_disk}GB, last status {last_msg or 'unknown'}). The "
                    "onstart never completed, and it hides failures, so the install log is "
                    "printed above to show which step broke."
                    + (f" Host {machine_id} blacklisted." if machine_id is not None else "")
                )
            url = self._resolve_server_url(info) if status == "running" else ""
            msg = f"status={status!r} url={url or 'port 8000 not mapped yet'} elapsed={int(elapsed)}s"
            if disk_used is not None:
                msg = f"{msg} disk={disk_used}GB"
            if status_msg:
                msg = f"{msg} msg={status_msg}"
            if msg != last_msg:
                print(f"[Vast Manager] {msg}{stall_note}")
                last_msg = msg
            if url:
                try:
                    if self.check_health(url).get("status") == "healthy":
                        self.server_url = url
                        self._forget_idle(instance_id)
                        return url
                except Exception as e:
                    print(f"[Vast Manager]   server not up yet: {type(e).__name__}")
            time.sleep(10)

    def fetch_instance_logs(self, instance_id: int, tail: int = 80) -> str:
        """Fetch the container's stdout from the Vast agent.

        `run_ssh` is useless exactly when it matters most: if the container exited there is
        nothing to SSH into, so every SSH probe reports "unavailable" and the real cause is
        lost -- which is how the last two failures became undiagnosable. The agent on the
        host still holds the container log, so ask the API instead. Raises when this CLI
        build has no `logs` subcommand, so callers must treat it as best effort.
        """
        raw = self.run_cli(["logs", str(instance_id)])
        lines = (raw or "").splitlines()
        return "\n".join(lines[-tail:])

    def _diagnose_boot(self, instance_id: int, reason: str) -> None:
        """Reach into a stuck instance and print why its install never completed.

        Without this the caller only sees a stall timeout, and a host that silently failed
        to install is indistinguishable from one that is merely slow. Best effort by design:
        a diagnosis failure must never mask the original error.
        """
        print(f"[Vast Manager] Diagnosing instance {instance_id}: {reason}", file=sys.stderr)
        # Ask the API first: it is the only probe that still works once the container has
        # exited, which is precisely when a diagnosis is wanted.
        try:
            api_log = self.fetch_instance_logs(instance_id)
            if api_log.strip():
                print("[Vast Manager]   --- container log (vastai logs) ---", file=sys.stderr)
                for line in api_log.splitlines()[-40:]:
                    print(f"[Vast Manager]   {line}", file=sys.stderr)
            else:
                print("[Vast Manager]   container log: (empty)", file=sys.stderr)
        except Exception as err:  # noqa: BLE001 - diagnosis must never mask the cause
            print(f"[Vast Manager]   container log: unavailable ({type(err).__name__})", file=sys.stderr)
        probes = (
            ("onstart.log", "tail -40 /workspace/onstart.log 2>&1"),
            ("onstart failures", "grep -n 'FAILED' /workspace/onstart.log 2>/dev/null | tail -10"),
            ("server.log", "tail -20 /workspace/server.log 2>&1"),
            ("site-packages", "du -sh /usr/local/lib/python*/site-packages 2>/dev/null || echo absent"),
            (
                "outbound network",
                "timeout 8 python -c \"import urllib.request as u; "
                "print(u.urlopen('https://pypi.org/simple/', timeout=6).status)\" 2>&1 | tail -1",
            ),
        )
        for label, command in probes:
            try:
                output = self.run_ssh(instance_id, command, timeout=45)
            except Exception as err:  # noqa: BLE001 - diagnosis must never mask the cause
                print(f"[Vast Manager]   {label}: unavailable ({type(err).__name__})", file=sys.stderr)
                continue
            text = (output or "").strip()
            if not text:
                print(f"[Vast Manager]   {label}: (empty)", file=sys.stderr)
                continue
            print(f"[Vast Manager]   --- {label} ---", file=sys.stderr)
            for line in text.splitlines()[-40:]:
                print(f"[Vast Manager]   {line}", file=sys.stderr)

    @staticmethod
    def _disk_used_gb(info: dict) -> float | None:
        """Best-effort disk usage for an instance, used as a boot-progress signal.

        The field name has varied across Vast API versions, so probe the known spellings
        and return None when none is present rather than guessing zero. Vast reports -1.0
        while there is no reading yet (the whole image-pull phase), and taking that at face
        value poisoned every downstream comparison: peak_disk became -1.0, which is below
        any install floor, so a host that was merely pulling the image was declared to have
        "installed nothing" and destroyed.
        """
        for key in ("disk_util", "disk_utilization", "disk_usage", "disk_util_gb", "cur_disk_util"):
            value = info.get(key)
            if isinstance(value, (int, float)) and value >= 0:
                return float(value)
        return None

    def check_health(self, url: str | None = None) -> dict[str, Any]:
        """Query the remote /health endpoint."""
        target = (url or self.server_url or "").rstrip("/") + "/health"
        req = urllib.request.Request(target, headers={"User-Agent": "tube2dscript/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def require_gpu(
        self,
        expected_gpu: str,
        url: str | None = None,
        require_tts: bool = False,
    ) -> dict[str, Any]:
        """Reject a server whose reported device or requested Indic TTS runtime is unusable.

        With `require_tts`, a server whose IndicF5 (or legacy TTS) import path is broken is
        rejected too. That state is invisible to a liveness check but makes every
        synthesis request answer HTTP 500, so without this the failure only surfaces
        after the rental has been paid for and narration has already started.

        A server that reports neither readiness field is accepted: instances cloned
        before this check existed cannot answer it, and refusing them would strand them.
        """
        last_err = None
        for attempt in range(3):
            try:
                health = self.check_health(url)
                break
            except Exception as e:
                last_err = e
                print(f"[Vast Manager]   health check attempt {attempt + 1}/3 failed: {type(e).__name__}; retrying...")
                time.sleep(10)
        else:
            raise RuntimeError(f"Vast server not reachable at {url}: {last_err}") from last_err
        actual = str(health.get("vram", {}).get("device_name", "")).upper()
        expected = expected_gpu.upper().replace("NVIDIA GEFORCE ", "")
        if expected not in actual:
            raise RuntimeError(
                f"Vast server GPU mismatch: expected {expected_gpu}, got "
                f"{health.get('vram', {}).get('device_name', 'unknown')}"
            )
        if require_tts:
            # Prefer the new IndicF5 contract, while accepting old servers that only
            # expose tts_ready so a warm legacy instance remains usable for Parler voices.
            # A caller may opt into legacy-only validation with VAST_TTS_ENGINE=parler.
            engine = os.environ.get("VAST_TTS_ENGINE", "indicf5").strip().lower()
            if engine != "parler" and "indicf5_ready" in health and health.get("indicf5_ready") is False:
                detail = health.get("indicf5_error") or "the IndicF5 import path is unusable on this instance"
                # A container with no token cannot download a GATED repo, and its environment
                # is fixed at create time, so "repair in place" is the wrong advice for it.
                # Separate the two so the operator takes the fix that matches the cause.
                if health.get("hf_token_present") is False:
                    raise RuntimeError(
                        f"Vast server {url or self.server_url} cannot synthesise IndicF5: {detail}. "
                        "This instance has no HF_TOKEN, and ai4bharat/IndicF5 is a GATED Hugging "
                        "Face repository. Accept its terms on the Hub, put HF_TOKEN (or "
                        "HUGGINGFACE_API_KEY) in .env, and rent a new instance: a container's "
                        "environment cannot be changed after it is created."
                    )
                # Name the command, not just the intent. "Repair the instance" left the
                # operator to find the subcommand, and this same branch also covers a wheel
                # resolved against the wrong NumPy, which is a repair of a different kind.
                raise RuntimeError(
                    f"Vast server {url or self.server_url} cannot synthesise IndicF5: {detail}. "
                    "Repair it (python scripts/vast_manager.py repair <instance-id>) "
                    "or rent a fresh one."
                )
            if (engine == "parler" or "indicf5_ready" not in health) and health.get("tts_ready") is False:
                detail = health.get("tts_error") or "the legacy TTS import path is unusable on this instance"
                raise RuntimeError(f"Vast server {url or self.server_url} cannot synthesise remote TTS: {detail}")
        return health

    def generate_image(
        self,
        prompt: str,
        width: int = 1080,
        height: int = 1920,
        steps: int = 4,
        seed: int | None = None,
        url: str | None = None,
    ) -> bytes:
        """Generate FLUX.1 image remotely on Vast.ai RTX 4090."""
        target = (url or self.server_url or "").rstrip("/") + "/v1/images/generate"
        payload = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "seed": seed,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            target,
            data=data,
            headers={"Content-Type": "application/json", "User-Agent": "tube2dscript/1.0"},
        )
        res = _http_json(req, timeout=600)
        if res.get("status") != "success":
            raise RuntimeError(f"Remote FLUX generation failed: {res.get('error')}")
        return base64.b64decode(res["image_base64"])

    def generate_video(
        self,
        prompt: str,
        width: int,
        height: int,
        num_frames: int,
        fps: int = 24,
        steps: int = 8,
        seed: int | None = None,
        model: str = "jbilcke-hf/LTX-Video-2b-0-9-8-distilled-HFIE",
        url: str | None = None,
    ) -> bytes:
        """Generate an MP4 clip remotely with the LTX-Video distilled model."""
        target = (url or self.server_url or "").rstrip("/") + "/v1/videos/generate"
        payload = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "fps": fps,
            "steps": steps,
            "seed": seed,
            "model": model,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            target,
            data=data,
            headers={"Content-Type": "application/json", "User-Agent": "tube2dscript/1.0"},
        )
        res = _http_json(req, timeout=900)
        if res.get("status") != "success":
            raise RuntimeError(f"Remote LTX-Video generation failed: {res.get('error')}")
        return base64.b64decode(res["video_base64"])

    def synthesize_tts(
        self,
        text: str,
        voice: str = "indic_parler_aarav",
        speed: float = 1.0,
        url: str | None = None,
    ) -> tuple[bytes, float]:
        """Synthesize an Indic voice remotely on Vast.ai RTX 4090."""
        target = (url or self.server_url or "").rstrip("/") + "/v1/audio/synthesize"
        payload = {
            "text": text,
            "voice": voice,
            "speed": speed,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            target,
            data=data,
            headers={"Content-Type": "application/json", "User-Agent": "tube2dscript/1.0"},
        )
        res = _http_json(req, timeout=600)
        if res.get("status") != "success":
            raise RuntimeError(f"Remote Indic TTS failed: {res.get('error')}")
        audio_bytes = base64.b64decode(res["audio_base64"])
        if _wav_is_bad(audio_bytes):
            raise RuntimeError("Remote Indic TTS returned silent/non-finite audio")
        dur = float(res.get("audio_duration_sec", 0.0))
        return audio_bytes, dur

    def synthesize_tts_batch(
        self,
        items: list[dict],
        voice: str = "indic_parler_aarav",
        speed: float = 1.0,
        url: str | None = None,
        timeout: int = 3600,
    ) -> dict[str, tuple[bytes, float]]:
        """Synthesise many narration segments in one round trip.

        A 10-minute script is hundreds of sentences, so per-sentence HTTP requests waste
        the round-trip cost. Returns {item_id: (wav_bytes, duration_sec)}; items that
        failed are omitted so the caller can retry them individually.
        """
        if not items:
            return {}
        target = (url or self.server_url or "").rstrip("/") + "/v1/audio/synthesize_batch"
        payload = {"items": items, "voice": voice, "speed": speed}
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            target,
            data=data,
            headers={"Content-Type": "application/json", "User-Agent": "tube2dscript/1.0"},
        )
        res = _http_json(req, timeout=timeout)
        if res.get("status") != "success":
            raise RuntimeError(f"Remote Indic TTS batch synthesis failed: {res.get('error')}")

        out: dict[str, tuple[bytes, float]] = {}
        errors: list[str] = []
        for entry in res.get("results") or []:
            if entry.get("error"):
                errors.append(f"{entry.get('id')}: {entry['error']}")
                continue
            audio_b64 = entry.get("audio_base64")
            if not audio_b64:
                continue
            audio_bytes = base64.b64decode(audio_b64)
            if _wav_is_bad(audio_bytes):
                errors.append(f"{entry.get('id')}: silent/non-finite audio")
                continue
            out[str(entry.get("id"))] = (
                audio_bytes,
                float(entry.get("audio_duration_sec", 0.0)),
            )
        if errors:
            print(
                f"[Vast Manager Warning] {len(errors)} batch segment(s) failed: {'; '.join(errors[:5])}",
                file=sys.stderr,
            )
        return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Vast.ai GPU Cloud Manager")
    subparsers = parser.add_subparsers(dest="action", help="Action to perform")

    # Search
    p_search = subparsers.add_parser("search", help="Search for a one-GPU Vast.ai offer")
    p_search.add_argument("--gpu", choices=["RTX 3090", "RTX 4090"], default="RTX 4090")
    p_search.add_argument("--max-price", type=float, default=None, help="Max all-in $/hr (default: per-GPU budget)")
    p_search.add_argument("--disk", type=int, default=30, help="Min disk space in GB")

    # Create
    p_create = subparsers.add_parser("create", help="Rent a one-GPU Vast.ai instance within budget")
    p_create.add_argument("--gpu", choices=["RTX 3090", "RTX 4090"], default="RTX 4090")
    p_create.add_argument("--offer", type=int, default=None, help="Specific offer ID")
    p_create.add_argument("--disk", type=int, default=60, help="Disk space in GB")
    p_create.add_argument("--max-price", type=float, default=None, help="Max all-in $/hr (default: per-GPU budget)")

    # Prune
    p_prune = subparsers.add_parser("prune", help="Destroy broken, over-budget or long-idle instances")
    p_prune.add_argument("--dry-run", action="store_true", help="List what would be destroyed and exit")
    p_prune.add_argument(
        "--idle-hours",
        type=float,
        default=None,
        help="also destroy instances stopped for at least this many hours (they still bill storage)",
    )

    # Hosts
    p_hosts = subparsers.add_parser("hosts", help="Show or clear the blacklisted-machine list")
    p_hosts.add_argument("--clear", action="store_true", help="Forget every blacklisted machine")

    # Start
    p_start = subparsers.add_parser("start", help="Start existing instance")
    p_start.add_argument("instance_id", type=int, help="Instance ID")

    # Stop
    p_stop = subparsers.add_parser("stop", help="Stop instance")
    p_stop.add_argument("instance_id", type=int, help="Instance ID")

    # Restart: stop then start, so the container's onstart re-runs and re-fetches the repo
    # code. A running instance keeps the code it booted with, so a pushed fix never reaches
    # its inference server until it is restarted.
    p_restart = subparsers.add_parser(
        "restart",
        help="Stop then start an instance so its onstart re-fetches the latest repo code",
    )
    p_restart.add_argument("instance_id", type=int, help="Instance ID")

    # Status
    p_status = subparsers.add_parser("status", help="Get instance details")
    p_status.add_argument("instance_id", type=int, help="Instance ID")

    # Health
    p_health = subparsers.add_parser("health", help="Check server health")
    p_health.add_argument("--url", help="Vast server URL")

    # Repair
    p_repair = subparsers.add_parser(
        "repair", help="Reapply the pinned server dependencies on an existing instance"
    )
    p_repair.add_argument("instance_id", type=int, help="Instance ID")
    p_repair.add_argument("--timeout", type=int, default=2700, help="SSH timeout in seconds")

    args = parser.parse_args()
    mgr = VastManager()

    if args.action == "search":
        limit = args.max_price if args.max_price is not None else gpu_budget(args.gpu)
        offers = mgr.search_gpu(args.gpu, max_dph=limit, min_disk=args.disk)
        print(f"\nFound {len(offers)} available {args.gpu} offers under ${limit:.2f}/hr:\n")
        print(f"{'Offer ID':<10} {'$/hr':<8} {'GPU':<14} {'VRAM':<8} {'DL (Mbps)':<12} {'Location'}")
        print("-" * 65)
        for o in offers[:10]:
            oid = o.get("id")
            dph = format_rate(hourly_rate(o))
            gpu = o.get("gpu_name", "RTX 4090")
            vram = f"{o.get('gpu_ram', 24)}GB"
            dl = f"{o.get('inet_down', 0):.0f}"
            loc = o.get("geolocation", "Global")
            print(f"{oid:<10} {dph:<8} {gpu:<14} {vram:<8} {dl:<12} {loc}")
        return 0

    elif args.action == "create":
        if args.offer is not None:
            inst_id = mgr.create_instance(
                offer_id=args.offer,
                disk_gb=args.disk,
                gpu_name=args.gpu,
                max_dph=args.max_price,
            )
            print(f"Created instance: {inst_id}")
        else:
            # Rent within budget and fall through to another host if the first fails.
            inst_id, url = mgr.provision_instance(
                gpu_name=args.gpu,
                disk_gb=args.disk,
                max_dph=args.max_price,
            )
            print(f"Created instance: {inst_id} (server: {url})")
        return 0

    elif args.action == "prune":
        removed = mgr.prune_instances(dry_run=args.dry_run, idle_hours=args.idle_hours)
        if removed:
            print(f"Destroyed {len(removed)} instance(s): {', '.join(map(str, removed))}")
        return 0

    elif args.action == "hosts":
        if args.clear:
            path = mgr._bad_hosts_path()
            if path.is_file():
                path.unlink()
                print(f"Cleared {path.name}; previously blocked machines may be rented again.")
            else:
                print("No blacklist file to clear.")
            return 0
        if not mgr.bad_hosts:
            print("No blacklisted machines.")
            return 0
        print(f"Blacklisted machines ({len(mgr.bad_hosts)}):")
        for machine, reason in sorted(mgr.bad_hosts.items(), key=lambda kv: int(kv[0])):
            print(f"  machine {machine}: {reason}")
        return 0

    elif args.action == "start":
        mgr.start_instance(args.instance_id)
        return 0

    elif args.action == "stop":
        mgr.stop_instance(args.instance_id)
        return 0

    elif args.action == "restart":
        mgr.stop_instance(args.instance_id)
        if not mgr.start_instance(args.instance_id):
            print(f"[Vast Manager] could not restart instance {args.instance_id}", file=sys.stderr)
            return 1
        print(
            f"[Vast Manager] instance {args.instance_id} restarting; its onstart will "
            "git-fetch the latest repo code and relaunch the inference server."
        )
        return 0

    elif args.action == "status":
        info = mgr.get_instance_info(args.instance_id)
        print(json.dumps(info, indent=2))
        return 0

    elif args.action == "health":
        h = mgr.check_health(args.url)
        print(json.dumps(h, indent=2))
        return 0

    elif args.action == "repair":
        info = mgr.get_instance_info(args.instance_id)
        status = str(info.get("actual_status") or info.get("status") or "").lower()
        if status not in ("running",):
            print(f"[Vast Manager] Instance {args.instance_id} is {status or 'not running'}; starting it...")
            mgr.start_instance(args.instance_id)
            mgr.wait_until_ready(args.instance_id)
        try:
            mgr.repair_dependencies(args.instance_id, timeout=args.timeout)
        except Exception as err:
            print(f"[Vast Manager Error] Could not repair instance {args.instance_id}: {err}", file=sys.stderr)
            print(
                "  Repair needs working SSH. If this account has no SSH key attached\n"
                "  (check with `vastai show ssh-keys`), the only way forward is a fresh\n"
                "  rental, which runs the current onstart with the pinned dependencies:\n"
                "    python scripts/vast_manager.py prune --idle-hours 0\n"
                '    python scripts/vast_manager.py create --gpu "RTX 3090"\n',
                file=sys.stderr,
            )
            return 1
        return 0

    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
