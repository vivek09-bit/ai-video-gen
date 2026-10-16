#!/usr/bin/env python3
"""Generate one LTX-Video clip per scene through a Vast.ai GPU server."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from generate_images import make_prompt
from vast_manager import VastManager


DEFAULT_MODEL = "jbilcke-hf/LTX-Video-2b-0-9-8-distilled-HFIE"


def _video_dimensions(width: int, height: int, max_dimension: int = 768) -> tuple[int, int]:
    """Fit the requested aspect ratio into LTX's practical 24GB generation size."""
    scale = min(1.0, max_dimension / max(width, height))

    def align(value: float) -> int:
        aligned = max(8, round(value / 32) * 32)
        return min(max_dimension, aligned // 32 * 32)

    return align(width * scale), align(height * scale)


def _frame_count(seconds: float, fps: int) -> int:
    """Return an LTX-compatible frame count (8n+1)."""
    requested = max(9, round(seconds * fps))
    return ((requested - 1 + 7) // 8) * 8 + 1


def _load_scene_still(scene_id: str, images_dir: Path) -> bytes | None:
    """Return the PNG bytes of a scene's generated still, or None when there is none.

    The image generator writes ``public/images/<scene_id>.png``. Animating from it makes
    the clip image-to-video, which keeps the motion consistent with the still art.
    """
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        candidate = images_dir / f"{scene_id}{ext}"
        if candidate.is_file() and candidate.stat().st_size > 1000:
            return candidate.read_bytes()
    return None


def generate_videos_for_script(
    script_data: dict[str, Any],
    *,
    out_dir: str | Path = "public/videos",
    images_dir: str | Path = "public/images",
    model: str = DEFAULT_MODEL,
    duration_seconds: float = 4.0,
    fps: int = 24,
    steps: int = 8,
    manager: VastManager | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Generate video backgrounds and replace scene shots to prevent image/video mixing.

    Clips are minutes of GPU work each, so an existing file is reused by default rather
    than regenerated: a run that dies at scene 15 must not re-pay for scenes 1-14.
    Pass `force=True` to regenerate.

    When a scene's still exists under ``images_dir`` it is used as the starting frame
    (image-to-video), so the clip matches the generated art for that scene.
    """
    scenes = script_data.get("scenes", [])
    if not scenes:
        raise ValueError("Cannot generate video: script contains no scenes.")
    if duration_seconds <= 0:
        raise ValueError("Video clip duration must be greater than zero.")
    if not 1 <= steps <= 50:
        raise ValueError("LTX inference steps must be between 1 and 50.")

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stills_path = Path(images_dir)
    topic = script_data.get("topic", script_data.get("title", ""))
    genre = script_data.get("genre")
    visual_style = script_data.get("visualStyle")
    # Research keywords (Phase 1) bias the LTX prompt for scenes without explicit keywords.
    script_keywords = script_data.get("keywords") or None
    # Theme (Phase 5) pushes the palette/mood into the visual prompt.
    theme = script_data.get("theme")
    width, height = _video_dimensions(
        int(script_data.get("width") or 1920),
        int(script_data.get("height") or 1080),
    )
    frame_total = _frame_count(duration_seconds, fps)
    # Built lazily: a run where every clip is already on disk never touches the GPU, so
    # it must not require a server URL to exist.
    client: VastManager | None = manager

    for index, scene in enumerate(scenes):
        scene_id = scene.get("id", f"scene-{index + 1:03d}")
        prompt = make_prompt(
            scene,
            topic,
            genre=genre,
            visual_style=visual_style,
            index=index,
            script_keywords=script_keywords,
            theme=theme,
        )
        video_name = f"{scene_id}.mp4"
        video_path = output_dir / video_name
        seed = int(hashlib.sha256(f"{topic}_{scene_id}_{index}".encode("utf-8")).hexdigest()[:8], 16)
        if not force and video_path.exists() and video_path.stat().st_size > 0:
            print(f"  [{index + 1}/{len(scenes)}] {video_name} already generated, reusing.")
        else:
            # Animate FROM the scene's still when one exists (image-to-video), so the
            # moving art matches the image that was generated for the same scene. Without
            # a still the server falls back to text-to-video.
            init_image = _load_scene_still(scene_id, stills_path)
            mode = "image" if init_image else "text"
            print(f"  [{index + 1}/{len(scenes)}] Generating LTX-Video clip for {scene_id} ({mode}-to-video)...")
            client = client or VastManager()
            video_data = client.generate_video(
                prompt,
                width=width,
                height=height,
                num_frames=frame_total,
                fps=fps,
                steps=steps,
                seed=seed,
                model=model,
                image_bytes=init_image,
            )
            if not video_data:
                raise RuntimeError(f"LTX-Video returned an empty clip for scene {scene_id}.")
            video_path.write_bytes(video_data)

        # Use a video background and discard planned image shots: a video-mode
        # render must never mix generated image assets with generated clips.
        scene["background"] = {"type": "video", "src": f"videos/{video_name}", "dim": 0.08}
        scene["shots"] = []

    script_data["visualMode"] = "video"
    script_data["videoModel"] = model
    return script_data
