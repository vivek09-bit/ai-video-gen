#!/usr/bin/env python3
"""Vast.ai GPU Cloud Manager & Client.

Image generation uses one RTX 3090; LTX-Video uses one RTX 4090. Both can also
run remote IndicF5 narration before the selected instance is stopped.

Usage:
  # Search cheapest RTX 4090 offers
  python scripts/vast_manager.py search

  # Create instance from best offer
  python scripts/vast_manager.py create [--disk 50]

  # Start existing instance
  python scripts/vast_manager.py start <INSTANCE_ID>

  # Check instance health / status
  python scripts/vast_manager.py status <INSTANCE_ID>

  # Stop instance to pause billing
  python scripts/vast_manager.py stop <INSTANCE_ID>
"""

from __future__ import annotations

import argparse
import base64
import json
import os
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


class VastManager:
    """Manages Vast.ai GPU instance lifecycle and remote inference."""

    def __init__(self, api_key: str | None = None, server_url: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("VAST_API_KEY") or os.environ.get("VASTAI_API_KEY")
        self.server_url = (server_url or os.environ.get("VAST_SERVER_URL", "")).rstrip("/")

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
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return res.stdout.strip()
        except subprocess.CalledProcessError as e:
            err = e.stderr.strip() or e.stdout.strip()
            raise RuntimeError(f"Vast.ai CLI command failed: {err}") from e

    @staticmethod
    def server_onstart_command() -> str:
        """Build a startup command that installs and launches this project's GPU API."""
        custom_command = os.environ.get("VAST_SERVER_ONSTART_CMD", "").strip()
        if custom_command:
            return custom_command

        repository = os.environ.get(
            "VAST_SERVER_REPO_URL",
            "https://github.com/vivek09-bit/ai-video-gen.git",
        )
        branch = os.environ.get("VAST_SERVER_GIT_REF", "main")
        return (
            "set -e; "
            "mkdir -p /workspace; cd /workspace; "
            f"git clone --depth 1 --branch {shlex.quote(branch)} "
            f"{shlex.quote(repository)} tube2dscript; "
            "cd /workspace/tube2dscript; "
            "python -m pip install --upgrade "
            "'diffusers>=0.33' accelerate transformers sentencepiece "
            "imageio imageio-ffmpeg soundfile f5-tts cached-path vocos safetensors; "
            "python scripts/vast_server.py"
        )

    def search_gpu(
        self,
        gpu_name: str,
        max_dph: float = 0.50,
        min_disk: int = 40,
    ) -> list[dict]:
        """Search for a single GPU of the requested model."""
        gpu_filter = gpu_name.upper().replace("NVIDIA GEFORCE ", "").replace(" ", "_")
        query = f"gpu_name={gpu_filter} num_gpus=1 rented=False dph<{max_dph} disk_space>={min_disk} inet_down>100"
        try:
            raw = self.run_cli(["search", "offers", query, "-o", "dph+", "--raw"])
            offers = json.loads(raw)
            return offers
        except Exception as e:
            print(f"[Vast Manager Warning] {gpu_name} offer search failed ({e}), trying standard search query...", file=sys.stderr)
            try:
                raw = self.run_cli(["search", "offers", f"gpu_name={gpu_filter} rented=False", "-o", "dph+", "--raw"])
                return json.loads(raw)
            except Exception:
                return []

    def search_rtx_4090(self, max_dph: float = 0.50, min_disk: int = 40) -> list[dict]:
        """Backwards-compatible search for RTX 4090 offers."""
        return self.search_gpu("RTX 4090", max_dph=max_dph, min_disk=min_disk)

    def search_rtx_3090(self, max_dph: float = 0.50, min_disk: int = 40) -> list[dict]:
        """Search for RTX 3090 24GB offers."""
        return self.search_gpu("RTX 3090", max_dph=max_dph, min_disk=min_disk)

    def create_instance(
        self,
        offer_id: int | None = None,
        disk_gb: int = 50,
        image: str = "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime",
        onstart_cmd: str | None = None,
        gpu_name: str = "RTX 4090",
    ) -> int:
        """Create a one-GPU Vast.ai instance of the requested model."""
        if offer_id is None:
            offers = self.search_gpu(gpu_name, min_disk=disk_gb)
            if not offers:
                raise RuntimeError(f"No available {gpu_name} offers found matching criteria.")
            best = offers[0]
            offer_id = int(best["id"])
            dph = best.get("dph_total", best.get("dph", "N/A"))
            print(f"[Vast Manager] Selected best {gpu_name} offer: {offer_id} (${dph}/hr)")

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
            "-p 8000:8000",
        "--ssh",
        "--direct",
        "--onstart-cmd",
        onstart_cmd or self.server_onstart_command(),
        "--raw",
        ]

        raw = self.run_cli(args)
        res = json.loads(raw)
        instance_id = res.get("new_contract") or res.get("id")
        print(f"[Vast Manager] Instance created successfully. ID: {instance_id}")
        return int(instance_id)

    def start_instance(self, instance_id: int) -> bool:
        """Start an existing stopped instance."""
        print(f"[Vast Manager] Starting instance {instance_id}...")
        self.run_cli(["start", "instance", str(instance_id)])
        return True

    def stop_instance(self, instance_id: int) -> bool:
        """Stop an instance to halt GPU compute billing."""
        print(f"[Vast Manager] Stopping instance {instance_id} to prevent idle charges...")
        self.run_cli(["stop", "instance", str(instance_id)])
        return True

    def destroy_instance(self, instance_id: int) -> bool:
        """Permanently delete an instance and release its disk."""
        print(f"[Vast Manager] Destroying instance {instance_id}...")
        self.run_cli(["destroy", "instance", str(instance_id)])
        return True

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

    def wait_until_ready(self, instance_id: int, timeout_sec: int = 300) -> str:
        """Wait for the provisioned inference HTTP service to pass its health check."""
        print(f"[Vast Manager] Waiting for instance {instance_id} and its inference server...")
        start_t = time.time()
        candidate_url = ""
        while time.time() - start_t < timeout_sec:
            info = self.get_instance_info(instance_id)
            status = info.get("actual_status") or info.get("status_msg") or ""
            if status == "running" and not candidate_url:
                public_ip = info.get("public_ipaddr") or info.get("ssh_host")
                direct_port = info.get("direct_port_start")
                ssh_port = info.get("ssh_port")
                print(f"[Vast Manager] Instance {instance_id} is running at {public_ip} (SSH: {ssh_port})")
                if direct_port:
                    candidate_url = f"http://{public_ip}:{direct_port}"
                else:
                    candidate_url = f"http://{public_ip}:8000"
            if status == "running" and candidate_url:
                try:
                    health = self.check_health(candidate_url)
                    if health.get("status") == "healthy":
                        self.server_url = candidate_url
                        return self.server_url
                except (urllib.error.URLError, TimeoutError, OSError, ValueError):
                    pass
            time.sleep(5)
        raise TimeoutError(
            f"Vast.ai inference server for instance {instance_id} did not become healthy "
            f"within {timeout_sec}s."
        )

    def check_health(self, url: str | None = None) -> dict[str, Any]:
        """Query the remote /health endpoint."""
        target = (url or self.server_url or "").rstrip("/") + "/health"
        req = urllib.request.Request(target, headers={"User-Agent": "tube2dscript/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def require_gpu(self, expected_gpu: str, url: str | None = None) -> dict[str, Any]:
        """Reject a server whose reported device does not match this generation mode."""
        health = self.check_health(url)
        actual = str(health.get("vram", {}).get("device_name", "")).upper()
        expected = expected_gpu.upper().replace("NVIDIA GEFORCE ", "")
        if expected not in actual:
            raise RuntimeError(
                f"Vast server GPU mismatch: expected {expected_gpu}, got "
                f"{health.get('vram', {}).get('device_name', 'unknown')}"
            )
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
        with urllib.request.urlopen(req, timeout=120) as resp:
            res = json.loads(resp.read().decode("utf-8"))
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
        with urllib.request.urlopen(req, timeout=900) as resp:
            res = json.loads(resp.read().decode("utf-8"))
            if res.get("status") != "success":
                raise RuntimeError(f"Remote LTX-Video generation failed: {res.get('error')}")
            return base64.b64decode(res["video_base64"])

    def synthesize_tts(
        self,
        text: str,
        voice: str = "indicf5_aarav",
        speed: float = 1.0,
        url: str | None = None,
    ) -> tuple[bytes, float]:
        """Synthesize IndicF5 speech remotely on Vast.ai RTX 4090."""
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
        with urllib.request.urlopen(req, timeout=120) as resp:
            res = json.loads(resp.read().decode("utf-8"))
            if res.get("status") != "success":
                raise RuntimeError(f"Remote IndicF5 TTS failed: {res.get('error')}")
            audio_bytes = base64.b64decode(res["audio_base64"])
            dur = float(res.get("audio_duration_sec", 0.0))
            return audio_bytes, dur


def main() -> int:
    parser = argparse.ArgumentParser(description="Vast.ai GPU Cloud Manager")
    subparsers = parser.add_subparsers(dest="action", help="Action to perform")

    # Search
    p_search = subparsers.add_parser("search", help="Search for a one-GPU Vast.ai offer")
    p_search.add_argument("--gpu", choices=["RTX 3090", "RTX 4090"], default="RTX 4090")
    p_search.add_argument("--max-price", type=float, default=0.50, help="Max $/hr")
    p_search.add_argument("--disk", type=int, default=40, help="Min disk space in GB")

    # Create
    p_create = subparsers.add_parser("create", help="Create a one-GPU Vast.ai instance")
    p_create.add_argument("--gpu", choices=["RTX 3090", "RTX 4090"], default="RTX 4090")
    p_create.add_argument("--offer", type=int, default=None, help="Specific offer ID")
    p_create.add_argument("--disk", type=int, default=50, help="Disk space in GB")

    # Start
    p_start = subparsers.add_parser("start", help="Start existing instance")
    p_start.add_argument("instance_id", type=int, help="Instance ID")

    # Stop
    p_stop = subparsers.add_parser("stop", help="Stop instance")
    p_stop.add_argument("instance_id", type=int, help="Instance ID")

    # Status
    p_status = subparsers.add_parser("status", help="Get instance details")
    p_status.add_argument("instance_id", type=int, help="Instance ID")

    # Health
    p_health = subparsers.add_parser("health", help="Check server health")
    p_health.add_argument("--url", help="Vast server URL")

    args = parser.parse_args()
    mgr = VastManager()

    if args.action == "search":
        offers = mgr.search_gpu(args.gpu, max_dph=args.max_price, min_disk=args.disk)
        print(f"\nFound {len(offers)} available {args.gpu} offers:\n")
        print(f"{'Offer ID':<10} {'$/hr':<8} {'GPU':<14} {'VRAM':<8} {'DL (Mbps)':<12} {'Location'}")
        print("-" * 65)
        for o in offers[:10]:
            oid = o.get("id")
            dph = f"${o.get('dph_total', o.get('dph', 0)):.3f}"
            gpu = o.get("gpu_name", "RTX 4090")
            vram = f"{o.get('gpu_ram', 24)}GB"
            dl = f"{o.get('inet_down', 0):.0f}"
            loc = o.get("geolocation", "Global")
            print(f"{oid:<10} {dph:<8} {gpu:<14} {vram:<8} {dl:<12} {loc}")
        return 0

    elif args.action == "create":
        inst_id = mgr.create_instance(offer_id=args.offer, disk_gb=args.disk, gpu_name=args.gpu)
        print(f"Created instance: {inst_id}")
        return 0

    elif args.action == "start":
        mgr.start_instance(args.instance_id)
        return 0

    elif args.action == "stop":
        mgr.stop_instance(args.instance_id)
        return 0

    elif args.action == "status":
        info = mgr.get_instance_info(args.instance_id)
        print(json.dumps(info, indent=2))
        return 0

    elif args.action == "health":
        h = mgr.check_health(args.url)
        print(json.dumps(h, indent=2))
        return 0

    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
