#!/usr/bin/env python3
"""Run many videos through a single GPU rental.

Ordering is the point. GPU time is billed hourly while the local render is free CPU
time, so the batch holds the GPU only while it is doing inference and releases it
before rendering anything:

    phase write   local + GPU   per job: research -> plan -> narration -> visuals
                                -> script.json, then snapshot the job's assets
    phase audio   GPU           per job: all narration synthesised and cached to disk
    phase render  local         per job: audio rebuilt from cache, then rendered

The GPU is provisioned once for the whole batch and stopped exactly once, after the
audio phase. Every artefact the GPU produces is written to local disk as it arrives,
so a crash mid-batch never re-pays for work already done.

Scene ids are reused across jobs (scene-001.png, scene-001.wav), so each job's assets
are parked under out/batch/jobs/<job-id>/ between their write and render phases.

new_video.py exposes the same early exits as flags: `--stop-after write|audio|metadata`
(the write phase passes `write`). phase_audio still drives tts.py directly so the GPU is
held for exactly one audio pass; it could instead call `--stop-after audio`.

Usage:
    python scripts/batch.py --jobs jobs.json
    python scripts/batch.py --jobs jobs.json --dry-run
    python scripts/batch.py --jobs jobs.json --phase render
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
STATE_FILE = ROOT / "out" / "batch" / "state.json"
SNAPSHOT_ROOT = ROOT / "out" / "batch" / "jobs"

# Directories a job owns while it is the "current" project. Everything else the pipeline
# writes is either content-addressed (out/gpu_audio_cache) or reproducible for free.
SNAPSHOT_DIRS = ("public/images", "public/videos")
SNAPSHOT_FILES = ("script.json",)

# Passed straight through to new_video.py as `--key value`.
JOB_KEYS = (
    "minutes",
    "genre",
    "voice",
    "audio-model",
    "model",
    "scene-seconds",
    "shots",
    "language",
    "format",
    "theme",
    "tone",
    "speed",
    "imageprovider",
    "imagemodel",
    "video-model",
    "video-seconds",
    "video-steps",
    "vast-strict",
)


def die(message: str) -> int:
    print(f"batch: {message}", file=sys.stderr)
    return 1


def load_manifest(path: Path) -> tuple[dict, list[dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    defaults = dict(data.get("defaults") or {})
    jobs = list(data.get("jobs") or [])
    if not jobs:
        raise ValueError("manifest contains no jobs")
    return defaults, jobs


def resolve_job(defaults: dict, job: dict, index: int) -> dict:
    """Merge defaults under one job and give it a stable id."""
    merged = {**defaults, **job}
    raw_id = str(merged.get("id") or merged.get("out") or merged.get("topic") or f"job{index + 1:02d}")
    safe = "".join(ch if (ch.isalnum() or ch in "-_") else "-" for ch in raw_id.lower())
    merged["_id"] = f"{index + 1:02d}-{safe.strip('-')[:48]}"
    if not merged.get("topic"):
        raise ValueError(f"job {index + 1} has no topic")
    merged.setdefault("out", f"out/batch-{merged['_id']}.mp4")
    return merged


def job_argv(job: dict, extra: list[str]) -> list[str]:
    argv = [sys.executable, str(SCRIPTS / "new_video.py"), "--topic", str(job["topic"])]
    argv += ["--out", str(job["out"])]
    for key in JOB_KEYS:
        if key in job and job[key] is not None:
            argv += [f"--{key}", str(job[key])]
    if job.get("video-gen"):
        argv.append("--video-gen")
    return argv + extra


# ---------------------------------------------------------------------------
# snapshots
# ---------------------------------------------------------------------------


def snapshot(job_id: str) -> Path:
    """Park the current project's per-job assets so the next job can overwrite them."""
    dest = SNAPSHOT_ROOT / job_id
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name in SNAPSHOT_FILES:
        src = ROOT / name
        if src.exists():
            shutil.copy2(src, dest / Path(name).name)
    for name in SNAPSHOT_DIRS:
        src = ROOT / name
        if not src.is_dir():
            continue
        target = dest / name
        target.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            if item.is_file():
                shutil.copy2(item, target / item.name)
    return dest


def restore(job_id: str) -> bool:
    """Put a job's assets back in place. Files not belonging to the job are left alone."""
    src_root = SNAPSHOT_ROOT / job_id
    if not src_root.exists():
        return False
    for name in SNAPSHOT_FILES:
        src = src_root / Path(name).name
        if src.exists():
            shutil.copy2(src, ROOT / name)
    for name in SNAPSHOT_DIRS:
        src = src_root / name
        if not src.is_dir():
            continue
        dest = ROOT / name
        dest.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            if item.is_file():
                shutil.copy2(item, dest / item.name)
    return True


# ---------------------------------------------------------------------------
# state ledger
# ---------------------------------------------------------------------------


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return {"jobs": {}}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def mark(state: dict, job_id: str, phase: str, ok: bool) -> None:
    entry = state.setdefault("jobs", {}).setdefault(job_id, {})
    entry[phase] = "ok" if ok else "failed"
    save_state(state)


def phase_done(state: dict, job_id: str, phase: str) -> bool:
    return (state.get("jobs", {}).get(job_id) or {}).get(phase) == "ok"


# ---------------------------------------------------------------------------
# GPU lifecycle - owned by the batch, not by the individual jobs
# ---------------------------------------------------------------------------


def needs_gpu(jobs: list[dict]) -> bool:
    for job in jobs:
        if job.get("video-gen"):
            return True
        if str(job.get("audio-model") or "").lower() == "indic_parler":
            return True
        if str(job.get("voice") or "").lower().startswith(("indic_parler", "indic_")):
            return True
    return False


def expected_gpu(jobs: list[dict]) -> str:
    return "RTX 4090" if any(job.get("video-gen") for job in jobs) else "RTX 3090"


def prefetch_for(jobs: list[dict]) -> list[str]:
    """Weights to pull while the container is still installing."""
    models: list[str] = []
    wants_indic_parler = any(
        str(job.get("audio-model") or "").lower() == "indic_parler"
        or str(job.get("voice") or "").lower().startswith(("indic_parler", "indic_"))
        for job in jobs
    )
    if wants_indic_parler:
        models.append("RXD03/indic-parler-tts")
    if any(job.get("video-gen") for job in jobs):
        video_models = [str(job["video-model"]) for job in jobs if job.get("video-model")]
        models.append(
            video_models[0] if video_models else "jbilcke-hf/LTX-Video-2b-0-9-8-distilled-HFIE"
        )
    return models


def provision(gpu_name: str, prefetch: list[str]) -> tuple[int, str]:
    """Reuse a matching instance if one exists, otherwise rent one. Returns (id, url)."""
    from vast_manager import VastManager

    mgr = VastManager()
    conflicts = mgr.conflicting_instances(gpu_name)
    for inst in conflicts:
        print(
            f"  [batch] instance {inst['id']} is a {inst['gpu_name']} ({inst['status']}, "
            f"{inst['disk_space']}GB) but this batch needs {gpu_name}.",
            file=sys.stderr,
        )
    if conflicts:
        raise RuntimeError(
            f"{len(conflicts)} existing instance(s) of the wrong GPU model would keep billing "
            f"storage alongside a new {gpu_name}. Prune them first."
        )

    instance_id, url = mgr.find_existing_instance(gpu_name, min_disk=50)
    if instance_id:
        if not url:
            print(f"  [batch] starting existing instance {instance_id}...")
            mgr.start_instance(instance_id)
        url = mgr.wait_until_ready(instance_id)
    else:
        instance_id, url = mgr.provision_instance(
            gpu_name=gpu_name,
            disk_gb=50,
            prefetch_models=prefetch or None,
        )
    mgr.require_gpu(gpu_name, url)
    return instance_id, url


def stop(instance_id: int) -> None:
    from vast_manager import VastManager

    try:
        print(f"\n[batch] Stopping instance {instance_id} to end compute billing...")
        VastManager().stop_instance(instance_id)
    except Exception as err:  # noqa: BLE001 - never fail a finished batch over this
        print(f"  [batch] warning: could not stop instance {instance_id}: {err}", file=sys.stderr)


# ---------------------------------------------------------------------------
# phases
# ---------------------------------------------------------------------------


def run_stage(
    label: str,
    argv: list[str] | str,
    env: dict | None = None,
    shell: bool = False,
) -> bool:
    print(f"\n--- {label} ---")
    result = subprocess.run(argv, cwd=str(ROOT), env=env, shell=shell)
    if result.returncode != 0:
        print(f"{label} failed with exit code {result.returncode}", file=sys.stderr)
        return False
    return True


def phase_write(args, jobs, state) -> int:
    for job in jobs:
        jid = job["_id"]
        if phase_done(state, jid, "write") and not args.force:
            print(f"[batch] {jid}: write already done, skipping.")
            continue
        print(f"\n{'=' * 74}\n[batch] WRITE {jid}  ({job['topic']})\n{'=' * 74}")
        ok = run_stage(f"write {jid}", job_argv(job, ["--stop-after", "write"]))
        if ok:
            snapshot(jid)
        mark(state, jid, "write", ok)
    return 0


def phase_audio(args, jobs, state, instance_id: int, url: str) -> int:
    env = {**os.environ, "VAST_SERVER_URL": url, "USE_VAST": "1"}
    failures = 0
    for job in jobs:
        jid = job["_id"]
        if not phase_done(state, jid, "write"):
            print(f"[batch] {jid}: write phase did not complete; skipping audio.", file=sys.stderr)
            failures += 1
            continue
        if phase_done(state, jid, "audio") and not args.force:
            print(f"[batch] {jid}: audio already done, skipping.")
            continue
        print(f"\n{'=' * 74}\n[batch] AUDIO {jid}  ({job['topic']})\n{'=' * 74}")
        if not restore(jid):
            print(f"[batch] {jid}: no snapshot to restore; skipping audio.", file=sys.stderr)
            failures += 1
            mark(state, jid, "audio", False)
            continue
        ok = run_stage(
            f"audio {jid}",
            [
                sys.executable,
                str(SCRIPTS / "tts.py"),
                str(ROOT / "script.json"),
                "--vast",
                "--vast-url",
                url,
                "--require-vast",
            ],
            env=env,
        )
        mark(state, jid, "audio", ok)
        if not ok:
            failures += 1
    return 1 if failures else 0


def phase_render(args, jobs, state) -> int:
    failures = 0
    for job in jobs:
        jid = job["_id"]
        out_path = ROOT / job["out"]
        if phase_done(state, jid, "render") and out_path.exists() and not args.force:
            print(f"[batch] {jid}: already rendered, skipping.")
            continue
        print(f"\n{'=' * 74}\n[batch] RENDER {jid}  ({job['topic']})\n{'=' * 74}")
        if not restore(jid):
            print(f"[batch] {jid}: no snapshot to restore; skipping render.", file=sys.stderr)
            failures += 1
            mark(state, jid, "render", False)
            continue
        # No VAST_SERVER_URL here on purpose: every segment was cached to disk during the
        # audio phase, so this run rebuilds the local WAVs from that cache and never
        # touches the network.
        env = {k: v for k, v in os.environ.items() if k not in ("VAST_SERVER_URL", "USE_VAST")}
        if not run_stage(
            f"narration rebuild from cache {jid}",
            [sys.executable, str(SCRIPTS / "tts.py"), str(ROOT / "script.json")],
            env=env,
        ):
            failures += 1
            mark(state, jid, "render", False)
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # shell=True because npx resolves to npx.cmd on Windows, which a bare argv list
        # cannot launch. new_video.py makes the same call the same way.
        render_cmd = f'npx remotion render Main "{job["out"]}" --concurrency={args.concurrency}'
        if args.gpu_render:
            render_cmd += " --gl=angle"
        ok = run_stage(f"render {jid}", render_cmd, env=env, shell=True)
        mark(state, jid, "render", ok)
        if not ok:
            failures += 1
    return 1 if failures else 0


# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--jobs", required=True, help="path to the batch manifest JSON")
    parser.add_argument(
        "--phase",
        choices=("all", "write", "audio", "render"),
        default="all",
        help="run only one phase (default: all)",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    parser.add_argument(
        "--force",
        action="store_true",
        help="redo phases that the ledger already marks complete",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1 if sys.platform == "win32" else 2,
        help="Remotion render concurrency per job",
    )
    parser.add_argument("--gpu-render", action="store_true", help="add --gl=angle to the render")
    args = parser.parse_args()

    manifest_path = Path(args.jobs)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    if not manifest_path.exists():
        return die(f"{manifest_path} not found")

    defaults, raw_jobs = load_manifest(manifest_path)
    jobs = [resolve_job(defaults, job, index) for index, job in enumerate(raw_jobs)]
    state = load_state()

    print(f"[batch] {len(jobs)} job(s) from {manifest_path.name}")
    for job in jobs:
        done = [p for p in ("write", "audio", "render") if phase_done(state, job["_id"], p)]
        print(
            f"  {job['_id']}: {job['topic']}  -> {job['out']}"
            + (f"   (done: {', '.join(done)})" if done else "")
        )

    if args.dry_run:
        gpu = expected_gpu(jobs) if needs_gpu(jobs) else "none (all local)"
        print(f"\n[batch] GPU: {gpu}")
        print(f"[batch] prefetch: {', '.join(prefetch_for(jobs)) or '-'}")
        print(f"[batch] snapshot root: {SNAPSHOT_ROOT.relative_to(ROOT)}")
        print(f"[batch] ledger: {STATE_FILE.relative_to(ROOT)}")
        return 0

    want_gpu = needs_gpu(jobs)
    instance_id: int | None = None
    url = ""

    try:
        if want_gpu:
            gpu_name = expected_gpu(jobs)
            print(f"\n{'=' * 74}\n[batch] GPU PHASE SETUP ({gpu_name})\n{'=' * 74}")
            instance_id, url = provision(gpu_name, prefetch_for(jobs))
            print(f"[batch] instance {instance_id} ready at {url}")

        if args.phase in ("all", "write"):
            rc = phase_write(args, jobs, state)
            if rc:
                return rc

        if args.phase in ("all", "audio") and want_gpu:
            rc = phase_audio(args, jobs, state, instance_id or 0, url)
            if rc:
                print("[batch] audio phase had failures; continuing to render what is available.", file=sys.stderr)

        # The GPU has done everything it was rented for. Release it BEFORE rendering, so
        # the multi-hour render is not billed as GPU time.
        if instance_id is not None:
            stop(instance_id)
            instance_id = None

        if args.phase in ("all", "render"):
            rc = phase_render(args, jobs, state)
            if rc:
                return rc
    except Exception as err:  # noqa: BLE001 - always try to release the GPU
        print(f"[batch] error: {type(err).__name__}: {err}", file=sys.stderr)
        if instance_id is not None:
            stop(instance_id)
        return 1

    print("\n[batch] done")
    for job in jobs:
        marks = state.get("jobs", {}).get(job["_id"], {})
        print(f"  {job['_id']}: " + ", ".join(f"{k}={v}" for k, v in marks.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
