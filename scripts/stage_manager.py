#!/usr/bin/env python3
"""Pipeline Stage Manager for Tube2DScript.

Provides structured multi-stage execution with pause, resume, single-stage targeting,
interactive stage approval, and persistent checkpointing across runs.

The 4 Canonical Stages:
  1. script_seo : Story blueprint, scene narration writing, and multi-platform SEO metadata.
  2. voice      : TTS audio synthesis (Kokoro, IndicF5, Indic Parler), Rhubarb phonemes & timings.
  3. images     : Scene background image generation (or LTX-Video clips) for all scenes.
  4. render     : Remotion final video composition, subtitle burning, MP4 rendering & upload.

Usage examples:
    # Run in staged mode with pauses between stages:
    python scripts/new_video.py --topic "khooni haveli" --pause

    # Resume a paused run from where it left off:
    python scripts/new_video.py --resume

    # Run only one specific stage:
    python scripts/new_video.py --stage voice
    python scripts/new_video.py --stage images
    python scripts/new_video.py --stage render

    # Pause after a specific stage:
    python scripts/new_video.py --topic "khooni haveli" --pause-after script

    # View stage status and artifacts:
    python scripts/new_video.py --pipeline-status
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STATE_FILE = ROOT / "out" / "pipeline_state.json"

STAGE_SCRIPT_SEO = "script_seo"
STAGE_VOICE = "voice"
STAGE_IMAGES = "images"
STAGE_RENDER = "render"

STAGES = [STAGE_SCRIPT_SEO, STAGE_VOICE, STAGE_IMAGES, STAGE_RENDER]
CANONICAL_STAGES = STAGES

STAGE_ALIASES: dict[str, str] = {
    # Stage 1: Script & SEO
    "1": STAGE_SCRIPT_SEO,
    "stage1": STAGE_SCRIPT_SEO,
    "script": STAGE_SCRIPT_SEO,
    "seo": STAGE_SCRIPT_SEO,
    "script_seo": STAGE_SCRIPT_SEO,
    "script-seo": STAGE_SCRIPT_SEO,
    "write": STAGE_SCRIPT_SEO,
    "story": STAGE_SCRIPT_SEO,

    # Stage 2: Voice
    "2": STAGE_VOICE,
    "stage2": STAGE_VOICE,
    "voice": STAGE_VOICE,
    "audio": STAGE_VOICE,
    "tts": STAGE_VOICE,
    "narration": STAGE_VOICE,

    # Stage 3: Images / Visuals
    "3": STAGE_IMAGES,
    "stage3": STAGE_IMAGES,
    "images": STAGE_IMAGES,
    "image": STAGE_IMAGES,
    "visual": STAGE_IMAGES,
    "visuals": STAGE_IMAGES,
    "video_gen": STAGE_IMAGES,
    "videos": STAGE_IMAGES,

    # Stage 4: Render / Editing
    "4": STAGE_RENDER,
    "stage4": STAGE_RENDER,
    "render": STAGE_RENDER,
    "export": STAGE_RENDER,
    "editing": STAGE_RENDER,
    "edit": STAGE_RENDER,
    "final": STAGE_RENDER,
    "video": STAGE_RENDER,
}

STAGE_TITLES: dict[str, str] = {
    STAGE_SCRIPT_SEO: "Stage 1: Script & SEO Generation",
    STAGE_VOICE: "Stage 2: Voice Generation (Audio & Lip Sync)",
    STAGE_IMAGES: "Stage 3: Image / Visual Generation",
    STAGE_RENDER: "Stage 4: Final Editing & Render (Remotion)",
}

STAGE_DESCRIPTIONS: dict[str, str] = {
    STAGE_SCRIPT_SEO: "Researches viral hooks, writes continuous story narration, creates script.json & multi-platform SEO tags",
    STAGE_VOICE: "Generates high-retention TTS voiceover, computes Rhubarb phoneme shapes, and locks word-level scene timings",
    STAGE_IMAGES: "Generates scene background images (or animated LTX-Video clips) matching visual keywords in script.json",
    STAGE_RENDER: "Composes audio, subtitles, visuals, and ambiances into the final MP4 video via Remotion (and optional auto-upload)",
}


def normalize_stage(name: str | None) -> str | None:
    """Normalize user-supplied stage name or alias to canonical stage string."""
    if not name:
        return None
    cleaned = str(name).strip().lower().replace("_", "").replace("-", "")
    for alias, canonical in STAGE_ALIASES.items():
        if cleaned == alias.replace("_", "").replace("-", ""):
            return canonical
    return None


def stage_index(stage: str | None) -> int:
    """Return 1-based index of a canonical stage (1..4), or 0 if unknown."""
    canonical = normalize_stage(stage)
    if not canonical:
        return 0
    try:
        return STAGES.index(canonical) + 1
    except ValueError:
        return 0


class PipelineStateManager:
    """Manages persistent stage execution, checkpoints, and artifact verification."""

    def __init__(self, state_file: Path | None = None) -> None:
        self.state_file = Path(state_file or DEFAULT_STATE_FILE)
        self.data: dict[str, Any] = self._load()

    @property
    def state(self) -> dict[str, Any]:
        """Expose current state data dictionary."""
        return self.data

    def _default_data(self) -> dict[str, Any]:
        return {
            "version": "1.0",
            "topic": "",
            "created_at": None,
            "updated_at": None,
            "args": {},
            "status": "not_started",
            "last_completed_stage": None,
            "paused_at_stage": None,
            "stages": {
                STAGE_SCRIPT_SEO: {"status": "pending", "started_at": None, "completed_at": None, "artifacts": [], "details": {}},
                STAGE_VOICE: {"status": "pending", "started_at": None, "completed_at": None, "artifacts": [], "details": {}},
                STAGE_IMAGES: {"status": "pending", "started_at": None, "completed_at": None, "artifacts": [], "details": {}},
                STAGE_RENDER: {"status": "pending", "started_at": None, "completed_at": None, "artifacts": [], "details": {}},
            },
        }

    def _load(self) -> dict[str, Any]:
        if self.state_file.exists():
            try:
                raw = json.loads(self.state_file.read_text(encoding="utf-8"))
                default = self._default_data()
                # Ensure all stages exist in loaded data
                for s in STAGES:
                    if s not in raw.get("stages", {}):
                        raw.setdefault("stages", {})[s] = default["stages"][s]
                return raw
            except Exception:
                pass
        return self._default_data()

    def save(self) -> None:
        """Atomically persist pipeline state to JSON file."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.data["updated_at"] = datetime.datetime.now().isoformat()
        temp_file = self.state_file.with_suffix(".tmp")
        temp_file.write_text(json.dumps(self.data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temp_file.replace(self.state_file)

    def reset(self) -> None:
        """Clear the pipeline state file."""
        if self.state_file.exists():
            try:
                self.state_file.unlink()
            except OSError:
                pass
        self.data = self._default_data()

    def sync_args(self, args: argparse.Namespace) -> None:
        """Store CLI arguments into state so --resume can recall options."""
        stored = {}
        for k, v in vars(args).items():
            # Filter non-serializable objects and transient stage flags
            if k in ("resume", "pause", "step", "interactive", "pipeline_status", "reset_pipeline"):
                continue
            if isinstance(v, (str, int, float, bool, list, dict)) or v is None:
                stored[k] = v
        self.data["args"] = stored
        self.data["params"] = stored
        if getattr(args, "topic", None):
            self.data["topic"] = args.topic
        if not self.data.get("created_at"):
            self.data["created_at"] = datetime.datetime.now().isoformat()
        self.save()

    def save_args(self, args: argparse.Namespace) -> None:
        """Alias for sync_args."""
        self.sync_args(args)

    def restore_args(self, current_args: argparse.Namespace) -> argparse.Namespace:
        """Populate empty fields in current_args from saved state during --resume."""
        saved_args = self.data.get("args") or self.data.get("params") or {}
        for k, v in saved_args.items():
            if hasattr(current_args, k):
                curr_val = getattr(current_args, k)
                # If current_args has default None, topic not supplied, or flag was enabled in saved state
                if curr_val is None or (k == "topic" and not curr_val):
                    setattr(current_args, k, v)
                elif isinstance(v, bool) and v and not curr_val:
                    setattr(current_args, k, v)
        return current_args

    def mark_started(self, stage: str) -> None:
        canonical = normalize_stage(stage)
        if not canonical:
            return
        stage_entry = self.data["stages"].setdefault(canonical, {})
        stage_entry["status"] = "in_progress"
        stage_entry["started_at"] = datetime.datetime.now().isoformat()
        self.data["status"] = "in_progress"
        self.data["paused_at_stage"] = None
        self.save()

    def record_stage_start(self, stage: str) -> None:
        """Record the start of a stage."""
        self.mark_started(stage)

    def mark_completed(self, stage: str, artifacts: list[str | Path] | None = None, details: dict | None = None) -> None:
        canonical = normalize_stage(stage)
        if not canonical:
            return
        stage_entry = self.data["stages"].setdefault(canonical, {})
        stage_entry["status"] = "completed"
        stage_entry["completed_at"] = datetime.datetime.now().isoformat()
        if artifacts:
            stage_entry["artifacts"] = [str(a) for a in artifacts]
        if details:
            stage_entry["details"] = details
        self.data["last_completed_stage"] = canonical
        self.data["paused_at_stage"] = None

        # Check if all stages are done
        if all(self.data["stages"].get(s, {}).get("status") == "completed" for s in STAGES):
            self.data["status"] = "completed"
        self.save()

    def record_stage_complete(
        self,
        stage: str,
        artifacts: list[str | Path] | None = None,
        details: dict | None = None,
    ) -> None:
        """Record the completion of a stage."""
        self.mark_completed(stage, artifacts=artifacts, details=details)

    def mark_paused(self, stage: str) -> None:
        canonical = normalize_stage(stage)
        if not canonical:
            return
        self.data["status"] = "paused"
        self.data["paused_at_stage"] = canonical
        self.save()

    def mark_failed(self, stage: str, error_message: str | None = None) -> None:
        canonical = normalize_stage(stage)
        if not canonical:
            return
        stage_entry = self.data["stages"].setdefault(canonical, {})
        stage_entry["status"] = "failed"
        if error_message:
            stage_entry["error"] = error_message
        self.data["status"] = "failed"
        self.save()

    def is_stage_completed(self, stage: str) -> bool:
        canonical = normalize_stage(stage)
        if not canonical:
            return False
        return self.data.get("stages", {}).get(canonical, {}).get("status") == "completed"

    def get_next_stage(self) -> str | None:
        """Return the first uncompleted canonical stage, or None if all are completed."""
        for s in STAGES:
            if not self.is_stage_completed(s):
                return s
        return None

    def verify_artifacts(self, stage: str) -> tuple[bool, list[str]]:
        """Verify if the expected disk artifacts for a given stage exist."""
        canonical = normalize_stage(stage)
        found: list[str] = []
        missing: list[str] = []

        if canonical == STAGE_SCRIPT_SEO:
            script_file = ROOT / "script.json"
            meta_file = ROOT / "out" / "metadata.json"
            if script_file.exists() and script_file.stat().st_size > 0:
                found.append(str(script_file.relative_to(ROOT)))
            else:
                missing.append("script.json")

            if meta_file.exists() and meta_file.stat().st_size > 0:
                found.append(str(meta_file.relative_to(ROOT)))

            return (len(missing) == 0 and len(found) > 0, found)

        elif canonical == STAGE_VOICE:
            proj_file = ROOT / "src" / "generated" / "project.json"
            audio_dir = ROOT / "public" / "audio"
            has_proj = proj_file.exists() and proj_file.stat().st_size > 0
            if has_proj:
                found.append(str(proj_file.relative_to(ROOT)))
            else:
                missing.append("src/generated/project.json")

            if audio_dir.is_dir() and any(audio_dir.glob("*.wav")):
                found.append(f"public/audio/ ({len(list(audio_dir.glob('*.wav')))} WAV files)")

            return (has_proj, found)

        elif canonical == STAGE_IMAGES:
            img_dir = ROOT / "public" / "images"
            vid_dir = ROOT / "public" / "videos"
            imgs = list(img_dir.glob("scene-*.png")) if img_dir.is_dir() else []
            vids = list(vid_dir.glob("scene-*.mp4")) if vid_dir.is_dir() else []
            if imgs:
                found.append(f"public/images/ ({len(imgs)} scene images)")
            if vids:
                found.append(f"public/videos/ ({len(vids)} scene clips)")
            return (bool(imgs or vids), found)

        elif canonical == STAGE_RENDER:
            out_str = self.data.get("args", {}).get("out", "out/video.mp4")
            out_file = ROOT / out_str
            if out_file.exists() and out_file.stat().st_size > 1024:
                found.append(str(out_file.relative_to(ROOT)))
                return (True, found)
            return (False, [])

        return (False, [])

    def prompt_after_stage(
        self,
        current_stage: str,
        interactive: bool = False,
        pause_after: str | None = None,
        on_pause: Any | None = None,
    ) -> bool:
        """Handle pause decision at stage boundaries.

        Returns:
            True if the pipeline should proceed to the next stage.
            False if the pipeline should pause and exit.
        """
        canonical_cur = normalize_stage(current_stage)
        idx = stage_index(canonical_cur)
        if idx <= 0 or idx >= len(STAGES):
            # Final stage finished or unknown, no pause needed
            return True

        next_stage = STAGES[idx]  # idx is 1-based, so STAGES[idx] is the next stage

        # Check explicit --pause-after / --stop-after match
        pause_target = normalize_stage(pause_after)
        if pause_target and pause_target == canonical_cur:
            print(f"\n[PAUSED] Stage checkpoint: reached --pause-after {canonical_cur}.")
            self.mark_paused(canonical_cur)
            if on_pause:
                on_pause()
            self._print_resume_instructions(next_stage)
            return False

        # If interactive mode is enabled (--pause / --step), prompt the operator
        if interactive:
            print()
            print("-" * 74)
            print(f"STAGE CHECKPOINT: {STAGE_TITLES[canonical_cur]} COMPLETED")
            print(f"Upcoming: {STAGE_TITLES[next_stage]}")
            print(f"          ({STAGE_DESCRIPTIONS[next_stage]})")
            print("-" * 74)
            print("Options:")
            print("  [Enter]  : Continue to next stage immediately")
            print("  [p/q]    : Pause pipeline here (resume anytime with --resume)")
            print("  [s]      : Show full pipeline status")
            print("-" * 74)

            while True:
                try:
                    choice = input("Enter choice ([Enter] to continue, 'p' to pause): ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    print()
                    choice = "p"

                if choice in ("", "c", "continue", "y", "yes"):
                    return True
                elif choice in ("p", "pause", "q", "quit", "exit"):
                    print(f"\n[PAUSED] Pipeline paused after {canonical_cur}.")
                    self.mark_paused(canonical_cur)
                    if on_pause:
                        on_pause()
                    self._print_resume_instructions(next_stage)
                    return False
                elif choice in ("s", "status"):
                    self.print_status()
                else:
                    print("Unrecognized option. Press Enter to continue, or 'p' to pause.")

        return True

    def should_execute_stage(
        self,
        stage: str,
        *,
        explicit_stage: str | None = None,
        from_stage: str | None = None,
        to_stage: str | None = None,
        resume: bool = False,
    ) -> bool:
        """Determine whether a given stage should be executed."""
        return should_execute_stage(
            stage,
            explicit_stage=explicit_stage,
            from_stage=from_stage,
            to_stage=to_stage,
            resume=resume,
            state_mgr=self,
        )

    def _print_resume_instructions(self, next_stage: str) -> None:
        print(f"Pipeline state saved in: {self.state_file.relative_to(ROOT) if self.state_file.is_relative_to(ROOT) else self.state_file}")
        print("\nTo resume this project right from the next stage, run:")
        print("  python scripts/new_video.py --resume")
        print("or with interactive pauses:")
        print("  python scripts/new_video.py --resume --pause\n")

    def print_status(self) -> None:
        """Render a clean status dashboard showing each stage and disk artifacts."""
        print()
        print("=" * 74)
        print("PIPELINE STAGE DASHBOARD")
        print("=" * 74)
        topic = self.data.get("topic") or self.data.get("args", {}).get("topic") or "No active topic"
        status_label = self.data.get("status", "not_started").upper()
        print(f"  Topic        : {topic}")
        print(f"  Overall State: {status_label}")
        if self.data.get("created_at"):
            print(f"  Started      : {self.data['created_at']}")
        if self.data.get("last_completed_stage"):
            print(f"  Last Done    : {self.data['last_completed_stage']}")
        print("-" * 74)

        for i, s in enumerate(STAGES, start=1):
            s_data = self.data.get("stages", {}).get(s, {})
            st = s_data.get("status", "pending")
            has_artifacts, disk_files = self.verify_artifacts(s)

            if st == "completed":
                badge = "[✓ COMPLETED]"
            elif st == "in_progress":
                badge = "[⏳ IN PROGRESS]"
            elif st == "failed":
                badge = "[✗ FAILED]"
            elif self.data.get("paused_at_stage") == s:
                badge = "[⏸️ PAUSED HERE]"
            else:
                badge = "[  PENDING  ]"

            title = STAGE_TITLES[s]
            print(f"  {badge:<16} {title}")
            print(f"                   {STAGE_DESCRIPTIONS[s]}")

            if disk_files:
                for f in disk_files:
                    print(f"                   ↳ Artifact: {f}")
            elif st == "completed":
                print("                   ↳ Completed in previous session")
            print()

        next_s = self.get_next_stage()
        print("-" * 74)
        if next_s:
            print(f"Next Action : Ready to execute {STAGE_TITLES[next_s]}")
            print("Resume Command : python scripts/new_video.py --resume")
        else:
            print("Next Action : All stages completed! Final video is ready.")
        print("=" * 74)
        print()


def should_execute_stage(
    stage: str,
    *,
    specific_stage: str | None = None,
    explicit_stage: str | None = None,
    from_stage: str | None = None,
    to_stage: str | None = None,
    resume: bool = False,
    state_mgr: PipelineStateManager | None = None,
) -> bool:
    """Determine whether a given stage should be executed under current filter rules."""
    canonical = normalize_stage(stage)
    if not canonical:
        return False

    target_stage = specific_stage or explicit_stage
    idx = stage_index(canonical)

    # 1. Exact stage targeting: --stage <name>
    if target_stage:
        target = normalize_stage(target_stage)
        return target == canonical

    # 2. Starting range: --from-stage <name>
    if from_stage:
        f_idx = stage_index(normalize_stage(from_stage))
        if f_idx > 0 and idx < f_idx:
            return False

    # 3. Ending range: --to-stage <name>
    if to_stage:
        t_idx = stage_index(normalize_stage(to_stage))
        if t_idx > 0 and idx > t_idx:
            return False

    # 4. Resume mode: skip already completed stages
    if resume and state_mgr:
        if state_mgr.is_stage_completed(canonical):
            return False

    return True
