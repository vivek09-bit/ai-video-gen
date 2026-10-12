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


def generate_videos_for_script(
    script_data: dict[str, Any],
    *,
    out_dir: str | Path = "public/videos",
    model: str = DEFAULT_MODEL,
    duration_seconds: float = 4.0,
    fps: int = 24,
    steps: int = 8,
    manager: VastManager | None = None,
) -> dict[str, Any]:
    """Generate video backgrounds and replace scene shots to prevent image/video mixing."""
    scenes = script_data.get("scenes", [])
    if not scenes:
        raise ValueError("Cannot generate video: script contains no scenes.")
    if duration_seconds <= 0:
        raise ValueError("Video clip duration must be greater than zero.")
    if not 1 <= steps <= 50:
        raise ValueError("LTX inference steps must be between 1 and 50.")

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    topic = script_data.get("topic", script_data.get("title", ""))
    genre = script_data.get("genre")
    visual_style = script_data.get("visualStyle")
    width, height = _video_dimensions(
        int(script_data.get("width") or 1920),
        int(script_data.get("height") or 1080),
    )
    frame_total = _frame_count(duration_seconds, fps)
    client = manager or VastManager()

    for index, scene in enumerate(scenes):
        scene_id = scene.get("id", f"scene-{index + 1:03d}")
        prompt = make_prompt(
            scene,
            topic,
            genre=genre,
            visual_style=visual_style,
            index=index,
        )
        video_name = f"{scene_id}.mp4"
        video_path = output_dir / video_name
        seed = int(hashlib.sha256(f"{topic}_{scene_id}_{index}".encode("utf-8")).hexdigest()[:8], 16)
        print(f"  [{index + 1}/{len(scenes)}] Generating LTX-Video clip for {scene_id}...")
        video_data = client.generate_video(
            prompt,
            width=width,
            height=height,
            num_frames=frame_total,
            fps=fps,
            steps=steps,
            seed=seed,
            model=model,
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
