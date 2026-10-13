#!/usr/bin/env python3
"""Vast.ai GPU Cloud Manager & Client.

Image generation uses one RTX 3090; LTX-Video uses one RTX 4090. Both can also
run remote IndicF5 narration before the selected instance is stopped.

An already-rented instance of the right GPU is reused before anything new is
rented, and every offer is filtered against a per-GPU all-in price ceiling
($0.40/hr for an RTX 3090, $0.50/hr for an RTX 4090). Override with
VAST_MAX_DPH_RTX_3090 / VAST_MAX_DPH_RTX_4090, or --max-price.

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

  # Destroy broken or over-budget instances
  python scripts/vast_manager.py prune [--dry-run]

  # Show or clear hosts that failed to start
  python scripts/vast_manager.py hosts [--clear]
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

# ---------------------------------------------------------------------------
# Hourly price ceilings in USD/hour, all-in (GPU + storage + bandwidth).
# Override any model with an env var: VAST_MAX_DPH_RTX_3090=0.35
# ---------------------------------------------------------------------------
DEFAULT_GPU_BUDGETS: dict[str, float] = {
    "RTX 3090": 0.40,
    "RTX 4090": 0.50,
}


def _gpu_slug(gpu_name: str) -> str:
    """Normalise a GPU name for env-var lookup: 'RTX 3090' -> 'RTX_3090'."""
    slug = "".join(ch if ch.isalnum() else "_" for ch in gpu_name.upper()).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug


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


class InstanceBroken(RuntimeError):
    """Raised when a rented instance cannot become healthy (bad host or image pull)."""


class VastManager:
    """Manages Vast.ai GPU instance lifecycle and remote inference."""

    # Machine IDs that failed to start are remembered here so the same broken host
    # is never rented twice (the cheapest offer is often the flakiest one).
    BAD_HOSTS_FILE = ".vast_bad_hosts.json"

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
    def server_onstart_command() -> str:
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
        return (
            "mkdir -p /workspace; exec > /workspace/onstart.log 2>&1; set -x; "
            "cd /workspace; "
            "apt-get update && apt-get install -y git ffmpeg libsndfile1 || true; "
            f"[ -d tube2dscript ] || git clone --depth 1 --branch {shlex.quote(branch)} "
            f"{shlex.quote(repo_url)} tube2dscript || true; "
            "cd /workspace/tube2dscript; "
            "python -m pip install --upgrade 'diffusers>=0.33' accelerate transformers "
            "sentencepiece protobuf imageio imageio-ffmpeg soundfile f5-tts cached-path "
            "vocos safetensors huggingface_hub || true; "
            "echo 'starting server at $(hostname -i):8000'; "
            "nohup python scripts/vast_server.py > /workspace/server.log 2>&1 & "
                        "sleep 5; echo '--- server.log after start ---'; tail -30 /workspace/server.log; "
                        "echo 'onstart done, keeping container alive'; sleep infinity"
                    )
    

    def search_gpu(
        self,
        gpu_name: str,
        max_dph: float | None = None,
        min_disk: int = 40,
    ) -> list[dict]:
        """Search for a single GPU of the requested model inside the budget ceiling."""
        limit = gpu_budget(gpu_name) if max_dph is None else max_dph
        gpu_filter = gpu_name.upper().replace("NVIDIA GEFORCE ", "").replace(" ", "_")
        query = f"gpu_name={gpu_filter} num_gpus=1 rented=False dph<{limit} disk_space>={min_disk} inet_down>100 cpu_ram>=32"
        offers: list[dict] = []
        try:
            offers = json.loads(self.run_cli(["search", "offers", query, "-o", "dph+", "--raw"]))
        except Exception as e:
            print(f"[Vast Manager Warning] {gpu_name} offer search failed ({e}), trying standard search query...", file=sys.stderr)
            try:
                offers = json.loads(self.run_cli(["search", "offers", f"gpu_name={gpu_filter} rented=False", "-o", "dph+", "--raw"]))
            except Exception:
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

    def create_instance(
        self,
        offer_id: int | None = None,
        disk_gb: int = 50,
        image: str = "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime",
        onstart_cmd: str | None = None,
        gpu_name: str = "RTX 4090",
        max_dph: float | None = None,
        offer_rate: float | None = None,
    ) -> int:
        """Create a one-GPU Vast.ai instance of the requested model inside the budget."""
        limit = gpu_budget(gpu_name) if max_dph is None else max_dph
        rate: float | None = None
        explicit_offer = offer_id is not None
        if offer_id is None:
            offers = self.search_gpu(gpu_name, max_dph=limit, min_disk=disk_gb)
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
            "-p 8000:8000",
            "--ssh",
            "--direct",
            "--onstart-cmd",
            onstart_cmd or self.server_onstart_command(),
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
    ) -> tuple[int, str]:
        """Rent a GPU, moving to the next offer when a host cannot pull the image.

        The cheapest in-budget offers are tried in turn. An instance that reports a
        fatal start-up error (e.g. a Docker registry TLS timeout) is destroyed so it
        stops billing storage before the next offer is attempted.
        """
        limit = gpu_budget(gpu_name) if max_dph is None else max_dph
        offers = self.search_gpu(gpu_name, max_dph=limit, min_disk=disk_gb)
        if not offers:
            raise RuntimeError(
                f"No available {gpu_name} offers under ${limit:.2f}/hr with at least {disk_gb}GB disk."
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
            instance_id = self.create_instance(
                offer_id=offer_id,
                disk_gb=disk_gb,
                gpu_name=gpu_name,
                max_dph=limit,
                offer_rate=rate,
            )
            try:
                return instance_id, self.wait_until_ready(instance_id)
            except Exception as error:
                last_error = error
                print(f"[Vast Manager Warning] Instance {instance_id} did not come up: {error}", file=sys.stderr)
                print(f"[Vast Manager] Destroying instance {instance_id} to stop storage charges...")
                try:
                    self.destroy_instance(instance_id)
                except Exception as destroy_err:
                    print(f"[Vast Manager Warning] Could not destroy {instance_id}: {destroy_err}", file=sys.stderr)
                if machine_id is not None:
                    self.mark_bad_host(machine_id, str(error))
        raise RuntimeError(
            f"Could not rent a working {gpu_name} instance after {total} attempt(s): {last_error}"
        ) from last_error

    def start_instance(self, instance_id: int) -> bool:
        """Start an existing stopped instance."""
        print(f"[Vast Manager] Starting instance {instance_id}...")
        self.run_cli(["start", "instance", str(instance_id)])
        return True

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
        return True

    def destroy_instance(self, instance_id: int) -> bool:
        """Permanently delete an instance and release its disk."""
        print(f"[Vast Manager] Destroying instance {instance_id}...")
        # -y is required: without it the CLI waits for a confirmation prompt.
        self.run_cli(["destroy", "instance", str(instance_id), "-y"])
        return True

    def prune_instances(self, dry_run: bool = False) -> list[int]:
        """Destroy rented instances that can never be reused (broken hosts, over budget)."""
        try:
            instances = json.loads(self.run_cli(["show", "instances", "--raw"]))
        except Exception as e:
            print(f"[Vast Manager Warning] Failed to list instances: {e}", file=sys.stderr)
            return []

        removed: list[int] = []
        if not instances:
            print("[Vast Manager] No rented instances to prune.")
            return removed

        for inst in instances:
            instance_id = int(inst.get("id") or 0)
            status = str(inst.get("actual_status") or inst.get("status") or "").lower()
            gpu_name = str(inst.get("gpu_name") or "unknown")
            status_msg = str(inst.get("status_msg") or "")
            rate = hourly_rate(inst)
            limit = gpu_budget(gpu_name)
            broken = status in ("error", "offline") or "error response from daemon" in status_msg.lower()
            over_budget = rate > limit
            if not broken and not over_budget:
                print(f"  Keeping {instance_id} ({gpu_name}, {status or 'unknown'}, {format_rate(rate)})")
                continue
            reason = "broken host / image pull failure" if broken else f"{format_rate(rate)} is above the ${limit:.2f}/hr budget"
            if dry_run:
                print(f"  [dry-run] Would destroy {instance_id} ({gpu_name}): {reason}")
                continue
            print(f"  Destroying {instance_id} ({gpu_name}): {reason}")
            try:
                self.destroy_instance(instance_id)
                removed.append(instance_id)
            except Exception as e:
                print(f"  [Vast Manager Warning] Could not destroy {instance_id}: {e}", file=sys.stderr)
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
        timeout_sec: int = 1800,
        container_timeout_sec: int | None = None,
    ) -> str:
        """Wait until the inference server answers /health.

        Two phases. The container must reach 'running' within `container_timeout_sec`
        (a host that never pulls the image, or stalls with no disk allocated, is
        treated as broken). The onstart script then has the rest of `timeout_sec`
        to boot the inference server, which legitimately takes 10-20 min.
        """
        if container_timeout_sec is None:
            container_timeout_sec = int(os.environ.get("VAST_CONTAINER_TIMEOUT", "600"))
        print(
            f"[Vast Manager] Waiting for instance {instance_id} and its inference server "
            f"(container up to {container_timeout_sec}s, server up to {timeout_sec}s)..."
        )
        start_t, container_up_at, last_msg = time.time(), None, ""
        while True:
            elapsed = time.time() - start_t
            if elapsed > timeout_sec:
                raise TimeoutError(
                    f"Instance {instance_id} not healthy after {timeout_sec}s. "
                    "SSH in and run: tail -50 /workspace/onstart.log"
                )
            info = self.get_instance_info(instance_id)
            status = info.get("actual_status") or ""
            status_msg = str(info.get("status_msg") or "")
            # A host that cannot pull the image will never become healthy; fail fast
            # so the caller can destroy it and try a different offer.
            if status in ("error", "offline") or "error response from daemon" in status_msg.lower():
                raise InstanceBroken(
                    f"instance {instance_id} failed to start [{status or 'error'}]: {status_msg or 'no status message'}"
                )
            if status == "running":
                container_up_at = container_up_at or time.time()
            elif container_up_at is None and elapsed > container_timeout_sec:
                # Still created/loading with no container after the grace period, e.g.
                # a host whose registry is unreachable and reports no error at all.
                raise InstanceBroken(
                    f"instance {instance_id} container never reached 'running' within "
                    f"{container_timeout_sec}s (stuck at {status or 'unknown'!r}): "
                    f"{status_msg or 'no status message'}"
                )
            url = self._resolve_server_url(info) if status == "running" else ""
            msg = f"status={status!r} url={url or 'port 8000 not mapped yet'} elapsed={int(elapsed)}s"
            if status_msg:
                msg = f"{msg} msg={status_msg}"
            if msg != last_msg:
                print(f"[Vast Manager] {msg}")
                last_msg = msg
            if url:
                try:
                    if self.check_health(url).get("status") == "healthy":
                        self.server_url = url
                        return url
                except Exception as e:
                    print(f"[Vast Manager]   server not up yet: {type(e).__name__}")
            time.sleep(10)

    def check_health(self, url: str | None = None) -> dict[str, Any]:
        """Query the remote /health endpoint."""
        target = (url or self.server_url or "").rstrip("/") + "/health"
        req = urllib.request.Request(target, headers={"User-Agent": "tube2dscript/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def require_gpu(self, expected_gpu: str, url: str | None = None) -> dict[str, Any]:
        """Reject a server whose reported device does not match this generation mode."""
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
        with urllib.request.urlopen(req, timeout=600) as resp:
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
        with urllib.request.urlopen(req, timeout=600) as resp:
            res = json.loads(resp.read().decode("utf-8"))
            if res.get("status") != "success":
                raise RuntimeError(f"Remote IndicF5 TTS failed: {res.get('error')}")
            audio_bytes = base64.b64decode(res["audio_base64"])
            dur = float(res.get("audio_duration_sec", 0.0))
            return audio_bytes, dur

    def synthesize_tts_batch(
        self,
        items: list[dict],
        voice: str = "indicf5_aarav",
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
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            res = json.loads(resp.read().decode("utf-8"))
        if res.get("status") != "success":
            raise RuntimeError(f"Remote IndicF5 batch synthesis failed: {res.get('error')}")

        out: dict[str, tuple[bytes, float]] = {}
        errors: list[str] = []
        for entry in res.get("results") or []:
            if entry.get("error"):
                errors.append(f"{entry.get('id')}: {entry['error']}")
                continue
            audio_b64 = entry.get("audio_base64")
            if not audio_b64:
                continue
            out[str(entry.get("id"))] = (
                base64.b64decode(audio_b64),
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
    p_prune = subparsers.add_parser("prune", help="Destroy broken or over-budget instances")
    p_prune.add_argument("--dry-run", action="store_true", help="List what would be destroyed and exit")

    # Hosts
    p_hosts = subparsers.add_parser("hosts", help="Show or clear the blacklisted-machine list")
    p_hosts.add_argument("--clear", action="store_true", help="Forget every blacklisted machine")

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
        removed = mgr.prune_instances(dry_run=args.dry_run)
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
