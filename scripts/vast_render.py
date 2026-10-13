#!/usr/bin/env python3
"""Render the assembled Remotion project on the rented Vast.ai instance.

Rendering locally is the default; this module is the `--render-on vast` path.

Why the payload is computed rather than copied
----------------------------------------------
`public/` cannot simply be shipped. A long-lived workspace accumulates assets from every
past run -- 287 MB of narration audio across 348 files and 118 MB of images here -- so
tarring the tree would push hundreds of megabytes for a project that needs a few tens.
The payload is therefore derived from `script.json` and `src/generated/project.json`, using
the same asset-resolution rules the renderer itself uses (see `resolveAudioPath` in
`src/Main.tsx`).

Everything the render reads
---------------------------
  * `src/**` and the root configs (Remotion sources, tiny)
  * `script.json` and `src/generated/**` (the project the renderer consumes)
  * `public/textures/**` (grain overlays, small)
  * narration + word timings + mouth shapes for the scenes in this script
  * the ambience / sfx / music beds this script names
  * the scene backgrounds it references (images or LTX clips)
"""

from __future__ import annotations

import json
import re
import sys
import tarfile
import time
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

# Remote layout, matching what server_onstart_command sets up.
REMOTE_REPO = "/workspace/tube2dscript"
REMOTE_STAGE = "/workspace/vast_render"
REMOTE_PAYLOAD = f"{REMOTE_STAGE}/payload.tar.gz"
REMOTE_OUTPUT = f"{REMOTE_REPO}/out/video.mp4"

MEDIA_EXTENSIONS = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".mp4", ".png", ".jpg", ".jpeg", ".webp")
ALWAYS_FILES = ("package.json", "package-lock.json", "tsconfig.json", "remotion.config.ts", "script.json")
ALWAYS_DIRS = ("src", "public/textures")
# Named beds resolved by src/Main.tsx by convention rather than by literal path.
NAMED_ASSET_RE = re.compile(r"\b(amb_[a-z0-9_]+|sfx_[a-z0-9_]+|[a-z0-9_]*music[a-z0-9_]*)\b", re.IGNORECASE)


def _walk_strings(node: Any) -> Iterable[str]:
    """Yield every string nested anywhere inside a parsed JSON document."""
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from _walk_strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk_strings(value)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def collect_payload_files(root: Path = ROOT) -> tuple[list[Path], list[str]]:
    """Work out which files the render needs. Returns (files, warnings)."""
    warnings: list[str] = []
    files: set[Path] = set()

    for name in ALWAYS_FILES:
        candidate = root / name
        if candidate.is_file():
            files.add(candidate)
        else:
            warnings.append(f"missing expected file: {name}")

    for relative in ALWAYS_DIRS:
        directory = root / relative
        if directory.is_dir():
            files.update(path for path in directory.rglob("*") if path.is_file())

    script = _load_json(root / "script.json") or {}
    scenes = script.get("scenes") or []
    scene_ids = [str(scene.get("id")) for scene in scenes if scene.get("id")]

    # 1. Literal asset paths. script.json stores backgrounds as "videos/x.mp4" and
    #    project.json stores narration as "audio/voice/x.wav"; both are public-relative.
    public = root / "public"
    literal_texts: list[str] = []
    for document in (script, _load_json(root / "src" / "generated" / "project.json")):
        literal_texts.extend(_walk_strings(document))
    for text in literal_texts:
        cleaned = text.strip()
        if not cleaned.lower().endswith(MEDIA_EXTENSIONS) or "/" not in cleaned:
            continue
        candidate = public / cleaned
        if candidate.is_file():
            files.add(candidate)
            # A referenced audio or image file almost always has companions that only
            # exist by naming convention: the word timings, the Rhubarb mouth shapes and
            # the dialog transcript share its stem, and Remotion reads them directly.
            # Missing them renders without captions or lip sync, so take the whole stem.
            for sibling in candidate.parent.glob(f"{candidate.stem}*"):
                if sibling.is_file() and sibling.suffix.lower() in MEDIA_EXTENSIONS + (".json", ".txt", ".srt"):
                    files.add(sibling)
        else:
            warnings.append(f"referenced asset not found: {cleaned}")

    # 2. Named beds (amb_*, sfx_*, *music*) are resolved by convention across several
    #    folders, so ship every file whose stem matches, wherever it lives.
    named: set[str] = set()
    for text in literal_texts:
        for match in NAMED_ASSET_RE.findall(text):
            named.add(match.lower())
    if named and (root / "public" / "audio").is_dir():
        for path in (root / "public" / "audio").rglob("*"):
            if path.is_file() and path.stem.lower() in named:
                files.add(path)

    # 3. Narration companions. tts.py names these by scene id, so a prefix glob picks up
    #    the wav, the word timings, the mouth shapes and the dialog transcript together.
    voice_dir = public / "audio" / "voice"
    if voice_dir.is_dir() and scene_ids:
        for path in voice_dir.iterdir():
            if path.is_file() and any(path.name.startswith(f"{sid}.") or path.name.startswith(f"{sid}-") for sid in scene_ids):
                files.add(path)

    # 4. Scene backgrounds by scene id, for anything step 1 did not already cover.
    for folder in ("images", "videos"):
        directory = public / folder
        if not directory.is_dir():
            continue
        for path in directory.iterdir():
            if path.is_file() and any(path.stem == sid or path.stem.startswith(f"{sid}-") for sid in scene_ids):
                files.add(path)

    existing = sorted(path for path in files if path.is_file())
    if not scene_ids:
        warnings.append("script.json contained no scenes; only the base project was packaged")
    return existing, warnings


def build_payload(root: Path = ROOT, out_tar: Path | None = None) -> tuple[Path, int, list[str]]:
    """Write a gzipped tar of everything the render needs. Returns (tar, file count, warnings)."""
    out_tar = out_tar or (root / "out" / "_vast_render_payload.tar.gz")
    out_tar.parent.mkdir(parents=True, exist_ok=True)
    files, warnings = collect_payload_files(root)
    if not files:
        raise RuntimeError("Nothing to package for the remote render.")

    with tarfile.open(out_tar, "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=str(path.relative_to(root)).replace("\\", "/"))
    return out_tar, len(files), warnings


def format_payload_summary(size_bytes: int, file_count: int) -> str:
    return f"{file_count} file(s), {size_bytes / (1024 * 1024):.1f} MB compressed"


def ensure_render_stack(manager: Any, instance_id: int, timeout: int = 2400) -> None:
    """Make an instance capable of running `npx remotion render`, if it is not already.

    Run over SSH rather than relying on the onstart script, because the onstart only fires
    at container creation: a warm-reused instance predating this feature would otherwise
    have no Node and the render would fail. Idempotent and cheap when already installed.
    """
    from vast_manager import VastManager

    print("[Vast Render] Ensuring Node, Chromium libraries, fonts and npm deps...")
    manager.run_ssh(instance_id, VastManager.render_stack_shell(), timeout=timeout)


def render_on_vast(
    manager: Any,
    instance_id: int,
    out_local: str | Path,
    *,
    concurrency: int = 8,
    gl: str | None = None,
    render_timeout: int = 7200,
    root: Path = ROOT,
) -> Path:
    """Render on the instance and pull the finished MP4 back to `out_local`."""
    out_local = Path(out_local)
    print("[Vast Render] Building the render payload from script.json and project.json...")
    tar_path, file_count, warnings = build_payload(root)
    for warning in warnings:
        print(f"  [Vast Render Warning] {warning}", file=sys.stderr)
    print(f"[Vast Render] Packaged {format_payload_summary(tar_path.stat().st_size, file_count)}")

    # Fail before transferring anything if SSH is not usable, so the error is about
    # access rather than a confusing scp failure.
    print("[Vast Render] Checking SSH access...")
    manager.run_ssh(instance_id, "echo ssh-ok", timeout=90)

    ensure_render_stack(manager, instance_id)

    print(f"[Vast Render] Uploading payload to {REMOTE_STAGE}...")
    manager.run_ssh(instance_id, f"mkdir -p {REMOTE_STAGE}", timeout=120)
    manager.push_file(instance_id, tar_path, REMOTE_PAYLOAD)

    gl_flag = f" --gl={gl}" if gl else ""
    remote_cmd = (
        f"cd {REMOTE_REPO} && "
        f"tar -xzf {REMOTE_PAYLOAD} && "
        f"rm -f out/video.mp4 && "
        f"npx remotion render Main out/video.mp4 --concurrency={concurrency}{gl_flag}"
    )
    print(f"[Vast Render] Rendering on instance {instance_id} (concurrency {concurrency})...")
    started = time.time()
    manager.run_ssh(instance_id, remote_cmd, timeout=render_timeout)
    print(f"[Vast Render] Render finished in {time.time() - started:.0f}s")

    print(f"[Vast Render] Downloading the result to {out_local}...")
    out_local.parent.mkdir(parents=True, exist_ok=True)
    # Download to a plain relative path first. scp reads a colon as a remote host
    # separator, so handing it an absolute Windows path like "C:\...\video.mp4" would be
    # misparsed as host "C".
    staged = Path("_vast_render_download.mp4")
    manager.pull_file(instance_id, REMOTE_OUTPUT, staged)
    if out_local.exists():
        out_local.unlink()
    import shutil as _shutil

    _shutil.move(str(staged), str(out_local))

    # The payload is large and no longer needed; leave the finished MP4 in place.
    try:
        manager.run_ssh(instance_id, f"rm -f {REMOTE_PAYLOAD}", timeout=120)
    except Exception:
        pass
    try:
        tar_path.unlink()
    except OSError:
        pass

    size = out_local.stat().st_size if out_local.is_file() else 0
    print(f"[Vast Render] Retrieved {out_local} ({size / (1024 * 1024):.1f} MB)")
    return out_local


def main() -> int:
    import argparse

    from vast_manager import VastManager

    parser = argparse.ArgumentParser(description="Render the project on a running Vast.ai instance")
    parser.add_argument("instance_id", type=int, help="running Vast.ai instance ID")
    parser.add_argument("--out", default="out/video.mp4", help="local output path")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--gl", default=None, help="Remotion GL backend, e.g. angle")
    parser.add_argument("--dry-run", action="store_true", help="build and report the payload, then stop")
    args = parser.parse_args()

    if args.dry_run:
        tar_path, files, warnings = build_payload()
        for warning in warnings:
            print(f"  [Vast Render Warning] {warning}", file=sys.stderr)
        print(format_payload_summary(tar_path.stat().st_size, files))
        tar_path.unlink()
        return 0

    render_on_vast(
        VastManager(),
        args.instance_id,
        args.out,
        concurrency=args.concurrency,
        gl=args.gl,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
