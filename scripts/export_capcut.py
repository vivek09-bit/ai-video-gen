#!/usr/bin/env python3
"""Export generated assets into a native CapCut Desktop draft project."""
import argparse
import json
import os
import re
import shutil
import sys
import time
import uuid
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def get_capcut_font_path() -> str:
    """Find system font from installed CapCut Desktop app."""
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        apps_dir = Path(local_app_data) / "CapCut" / "Apps"
        if apps_dir.exists():
            fonts = list(apps_dir.glob("**/Resources/Font/SystemFont/en.ttf"))
            if not fonts:
                fonts = list(apps_dir.glob("**/Resources/Font/SystemFont/*.ttf"))
            if fonts:
                return str(fonts[0].resolve()).replace("\\", "/")
    return ""


def get_native_capcut_platform(capcut_dir: Path) -> dict:
    """Read host CapCut Desktop platform & device validation info from an existing draft."""
    for draft_json in capcut_dir.glob("*/draft_content.json"):
        try:
            data = json.loads(draft_json.read_text(encoding="utf-8"))
            plat = data.get("platform")
            if plat and plat.get("device_id"):
                return plat
        except Exception:
            continue
    return {
        "os": "windows",
        "os_version": "10.0.26300",
        "app_id": 359289,
        "app_version": "9.2.0",
        "app_source": "cc",
        "device_id": "a2dcc293f8807385dfbfecf5c6bc4982",
        "hard_disk_id": "",
        "mac_address": "ee69854a097c43ad7af2b27ccca2df35",
    }


def get_capcut_draft_dir() -> Path:
    """Find CapCut Desktop project directory on Windows or via env var."""
    custom_dir = os.environ.get("CAPCUT_PROJECTS_DIR")
    if custom_dir and Path(custom_dir).exists():
        return Path(custom_dir)

    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        capcut_path = Path(local_app_data) / "CapCut" / "User Data" / "Projects" / "com.lveditor.draft"
        if capcut_path.exists():
            return capcut_path

    # Fallback to standard user path
    user_home = Path.home()
    capcut_path = user_home / "AppData" / "Local" / "CapCut" / "User Data" / "Projects" / "com.lveditor.draft"
    capcut_path.mkdir(parents=True, exist_ok=True)
    return capcut_path


def parse_srt(srt_path: Path) -> list[dict]:
    """Parse SRT subtitles into start_us, duration_us, and text."""
    if not srt_path.exists():
        return []

    cues = []
    pattern = re.compile(
        r"(\d+)\s*\n"
        r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*\n"
        r"([\s\S]*?)(?=\n\s*\n|\Z)"
    )

    content = srt_path.read_text(encoding="utf-8", errors="replace")
    for match in pattern.finditer(content):
        h1, m1, s1, ms1 = map(int, match.group(2, 3, 4, 5))
        h2, m2, s2, ms2 = map(int, match.group(6, 7, 8, 9))
        text = match.group(10).strip().replace("\n", " ")

        start_us = int((h1 * 3600 + m1 * 60 + s1 + ms1 / 1000.0) * 1_000_000)
        end_us = int((h2 * 3600 + m2 * 60 + s2 + ms2 / 1000.0) * 1_000_000)
        duration_us = max(100_000, end_us - start_us)

        cues.append({
            "text": text,
            "start": start_us,
            "duration": duration_us,
        })

    return cues


def export_to_capcut(
    project_json_path: Path | None = None,
    draft_name: str | None = None,
    open_capcut: bool = False,
) -> Path | None:
    """Create a CapCut draft folder and register it in CapCut's recent drafts list."""
    capcut_dir = get_capcut_draft_dir()
    print(f"[CapCut Export] Target directory: {capcut_dir}")

    # 1. Load project data
    if project_json_path is None or not project_json_path.exists():
        project_json_path = ROOT / "src" / "generated" / "project.json"

    script_path = ROOT / "script.json"
    project_data = {}
    if project_json_path.exists():
        try:
            project_data = json.loads(project_json_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[CapCut Export Warning] Could not parse project.json ({e})")

    script_data = {}
    if script_path.exists():
        try:
            script_data = json.loads(script_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[CapCut Export Warning] Could not parse script.json ({e})")

    script_meta = project_data.get("script") or script_data or {}
    title = script_meta.get("title") or script_meta.get("topic") or "Tube2D_Video"
    title_slug = "".join(c if c.isalnum() else "_" for c in title).strip("_")[:30] or "Tube2D_Video"

    width = script_meta.get("width") or (1080 if script_meta.get("aspectRatio") == "9:16" else 1920)
    height = script_meta.get("height") or (1920 if script_meta.get("aspectRatio") == "9:16" else 1080)
    fps = float(script_meta.get("fps") or 30.0)

    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    draft_folder_name = draft_name or f"{title_slug}_{timestamp_str}"
    draft_folder = capcut_dir / draft_folder_name
    draft_folder.mkdir(parents=True, exist_ok=True)

    draft_id = str(uuid.uuid4()).upper()

    # 2. Extract scenes and durations
    timings = project_data.get("timings", {})
    timing_scenes = timings.get("scenes", [])
    script_scenes = script_meta.get("scenes", [])

    scene_map = {}
    for s in timing_scenes:
        sid = s.get("id")
        if sid:
            scene_map[sid] = s

    # 3. Build materials and track segments
    videos_mat = []
    audios_mat = []
    texts_mat = []

    video_segments = []
    audio_segments = []
    text_segments = []

    current_time_us = 0
    first_image_path = None

    scenes_to_process = script_scenes if script_scenes else timing_scenes
    for idx, sc in enumerate(scenes_to_process):
        sid = sc.get("id") or f"scene-{idx+1:03d}"
        t_info = scene_map.get(sid, {})

        duration_sec = t_info.get("duration") or sc.get("duration") or 5.0
        duration_us = int(float(duration_sec) * 1_000_000)

        # A. Video / Image Asset
        # Check video clip first, then image
        video_clip = None
        if sc.get("video"):
            v_cand = ROOT / "public" / sc["video"]
            if v_cand.exists():
                video_clip = v_cand
        if video_clip is None:
            for v_name in (f"{sid}.mp4", f"{sid}.mov"):
                for v_folder in ("videos", "video"):
                    v_cand = ROOT / "public" / v_folder / v_name
                    if v_cand.exists():
                        video_clip = v_cand
                        break
                if video_clip:
                    break

        image_clip = None
        if sc.get("image"):
            i_cand = ROOT / "public" / sc["image"]
            if i_cand.exists():
                image_clip = i_cand
        if image_clip is None:
            for ext in (".png", ".jpg", ".jpeg", ".webp"):
                i_cand = ROOT / "public" / "images" / f"{sid}{ext}"
                if i_cand.exists():
                    image_clip = i_cand
                    break

        visual_path = None
        is_video = False
        if video_clip and video_clip.exists():
            visual_path = video_clip
            is_video = True
        elif image_clip and image_clip.exists():
            visual_path = image_clip
            if first_image_path is None:
                first_image_path = image_clip

        if visual_path and visual_path.exists():
            v_mat_id = str(uuid.uuid4()).upper()
            v_path_str = str(visual_path.resolve()).replace("\\", "/")
            videos_mat.append({
                "id": v_mat_id,
                "type": "video" if is_video else "photo",
                "path": v_path_str,
                "duration": duration_us if is_video else 10800000000,
                "width": width,
                "height": height,
                "category_name": "local",
                "material_name": visual_path.name,
                "crop_ratio": "free",
                # Mute background video audio so it doesn't double-play with the narration track
                "has_audio": False,
            })

            video_segments.append({
                "id": str(uuid.uuid4()).upper(),
                "material_id": v_mat_id,
                "source_timerange": {"start": 0, "duration": duration_us},
                "target_timerange": {"start": current_time_us, "duration": duration_us},
                "speed": 1.0,
                "volume": 0.0,
                "clip": {
                    "scale": {"x": 1.0, "y": 1.0},
                    "rotation": 0.0,
                    "transform": {"x": 0.0, "y": 0.0},
                    "alpha": 1.0,
                },
                "visible": True,
            })

        # B. Voiceover Audio Asset
        audio_rel = t_info.get("audio") or f"audio/voice/{sid}.wav"
        audio_path = ROOT / "public" / audio_rel
        if not audio_path.exists():
            audio_path = ROOT / "public" / "audio" / f"{sid}.wav"

        if audio_path.exists():
            audio_dur_us = duration_us
            try:
                with wave.open(str(audio_path), "rb") as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    if rate > 0:
                        audio_dur_us = int(frames / float(rate) * 1_000_000)
            except Exception:
                audio_dur_us = duration_us

            a_mat_id = str(uuid.uuid4()).upper()
            a_path_str = str(audio_path.resolve()).replace("\\", "/")
            audios_mat.append({
                "id": a_mat_id,
                "name": audio_path.name,
                "path": a_path_str,
                "duration": audio_dur_us,
                "type": "extract_music",
                "category_name": "local",
            })

            audio_offset_us = int(float(t_info.get("audioStart", 0.0)) * 1_000_000)
            audio_segments.append({
                "id": str(uuid.uuid4()).upper(),
                "material_id": a_mat_id,
                "source_timerange": {"start": 0, "duration": audio_dur_us},
                "target_timerange": {"start": current_time_us + audio_offset_us, "duration": audio_dur_us},
                "volume": 1.0,
                "speed": 1.0,
            })

        current_time_us += duration_us

    total_duration_us = current_time_us

    # 4. Subtitles / Captions Track
    srt_file = ROOT / "src" / "generated" / "subtitles.srt"
    cues = parse_srt(srt_file)
    font_path = get_capcut_font_path()
    for cue in cues:
        cue_text = cue["text"]
        if not cue_text:
            continue

        t_mat_id = str(uuid.uuid4()).upper()
        style_entry = {
            "fill": {
                "content": {
                    "render_type": "solid",
                    "solid": {"color": [1.0, 1.0, 1.0]},
                }
            },
            "size": 9.0,
            "useLetterColor": True,
            "range": [0, len(cue_text)],
        }
        if font_path:
            style_entry["font"] = {"id": "", "path": font_path}

        content_obj = {
            "text": cue_text,
            "styles": [style_entry],
        }

        texts_mat.append({
            "id": t_mat_id,
            "type": "text",
            "content": json.dumps(content_obj, ensure_ascii=False),
            "font_path": font_path,
            "font_size": 9.0,
            "text_color": "#ffffff",
            "alignment": 1,
            "border_color": "#000000",
            "border_width": 0.08,
            "has_shadow": True,
            "shadow_color": "#000000",
            "shadow_alpha": 0.85,
        })

        text_segments.append({
            "id": str(uuid.uuid4()).upper(),
            "material_id": t_mat_id,
            "source_timerange": {"start": 0, "duration": cue["duration"]},
            "target_timerange": {"start": cue["start"], "duration": cue["duration"]},
            "clip": {
                "scale": {"x": 1.0, "y": 1.0},
                "transform": {"x": 0.0, "y": -0.72},
            },
            "visible": True,
        })

    # 5. Assemble Tracks
    tracks = []
    if video_segments:
        tracks.append({
            "id": str(uuid.uuid4()).upper(),
            "type": "video",
            "flag": 0,
            "attribute": 0,
            "segments": video_segments,
        })
    if audio_segments:
        tracks.append({
            "id": str(uuid.uuid4()).upper(),
            "type": "audio",
            "flag": 0,
            "attribute": 0,
            "segments": audio_segments,
        })
    if text_segments:
        tracks.append({
            "id": str(uuid.uuid4()).upper(),
            "type": "text",
            "flag": 0,
            "attribute": 0,
            "segments": text_segments,
        })

    # 6. Build draft_content.json
    now_ms = int(time.time() * 1_000_000)
    draft_content = {
        "id": draft_id,
        "version": 360000,
        "new_version": "181.0.0",
        "name": draft_folder_name,
        "duration": total_duration_us,
        "create_time": now_ms,
        "update_time": now_ms,
        "fps": fps,
        "color_space": -1,
        "canvas_config": {
            "ratio": "original",
            "width": width,
            "height": height,
            "background": None,
        },
        "tracks": tracks,
        "materials": {
            "videos": videos_mat,
            "audios": audios_mat,
            "texts": texts_mat,
            "images": [],
            "canvases": [],
            "effects": [],
            "stickers": [],
            "transitions": [],
            "audio_effects": [],
            "audio_fades": [],
            "beats": [],
            "speeds": [],
        },
        "platform": get_native_capcut_platform(capcut_dir),
        "last_modified_platform": get_native_capcut_platform(capcut_dir),
        "draft_type": "video",
        "source": "default",
    }

    content_path = draft_folder / "draft_content.json"
    content_path.write_text(json.dumps(draft_content, ensure_ascii=False, indent=2), encoding="utf-8")

    # 7. Build draft_meta_info.json
    draft_meta = {
        "draft_id": draft_id,
        "draft_name": draft_folder_name,
        "draft_fold_path": str(draft_folder).replace("\\", "/"),
        "draft_root_path": str(capcut_dir).replace("\\", "/"),
        "tm_draft_create": now_ms,
        "tm_draft_modified": now_ms,
        "tm_duration": total_duration_us,
        "draft_cover": "draft_cover.jpg",
        "draft_materials": [],
        "draft_need_rename_folder": False,
        "draft_removable_storage_device": "",
        "draft_timeline_materials_size_": 1000000,
        "draft_is_ai_shorts": False,
        "draft_is_cloud_temp_draft": False,
        "draft_is_invisible": False,
        "draft_is_pippit_draft": False,
        "draft_is_web_article_video": False,
        "draft_materials_copied_info": [],
        "draft_segment_extra_info": [],
    }
    (draft_folder / "draft_meta_info.json").write_text(
        json.dumps(draft_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 7b. Write auxiliary config files expected by CapCut Desktop
    (draft_folder / "draft_agency_config.json").write_text(
        json.dumps({
            "is_auto_agency_enabled": False,
            "is_auto_agency_popup": False,
            "is_single_agency_mode": False,
            "marterials": None,
            "use_converter": False,
            "video_resolution": 720,
        }),
        encoding="utf-8",
    )
    (draft_folder / "draft_biz_config.json").write_text("", encoding="utf-8")

    # 8. Copy cover image if available
    if first_image_path and first_image_path.exists():
        try:
            shutil.copy2(first_image_path, draft_folder / "draft_cover.jpg")
        except Exception:
            pass

    # 9. Register in root_meta_info.json (with automatic safety backup)
    root_meta_file = capcut_dir / "root_meta_info.json"
    if root_meta_file.exists():
        try:
            # Create backup if not already present
            backup_path = root_meta_file.with_suffix(".json.bak")
            if not backup_path.exists():
                shutil.copy2(root_meta_file, backup_path)

            root_meta = json.loads(root_meta_file.read_text(encoding="utf-8"))
            all_drafts = root_meta.get("all_draft_store", [])
            # Filter out any duplicate of this draft
            all_drafts = [d for d in all_drafts if d.get("draft_fold_path") != str(draft_folder).replace("\\", "/")]

            new_entry = {
                "cloud_draft_cover": False,
                "cloud_draft_sync": False,
                "draft_cloud_last_action_download": False,
                "draft_cloud_purchase_info": "",
                "draft_cloud_template_id": "",
                "draft_cloud_tutorial_info": "",
                "draft_cloud_videocut_purchase_info": "",
                "draft_cover": str(draft_folder / "draft_cover.jpg").replace("\\", "/"),
                "draft_fold_path": str(draft_folder).replace("\\", "/"),
                "draft_id": draft_id,
                "draft_is_ai_shorts": False,
                "draft_is_cloud_temp_draft": False,
                "draft_is_invisible": False,
                "draft_is_pippit_draft": False,
                "draft_is_web_article_video": False,
                "draft_json_file": str(content_path).replace("\\", "/"),
                "draft_name": draft_folder_name,
                "draft_new_version": "",
                "draft_root_path": str(capcut_dir).replace("\\", "/"),
                "draft_timeline_materials_size": 1000000,
                "draft_type": "",
                "draft_web_article_video_enter_from": "",
                "pippit_avatar_url": "",
                "pippit_extra_info": "",
                "pippit_id": "",
                "pippit_user_name": "",
                "streaming_edit_draft_ready": True,
                "tm_draft_cloud_completed": "",
                "tm_draft_cloud_entry_id": -1,
                "tm_draft_cloud_modified": 0,
                "tm_draft_cloud_parent_entry_id": -1,
                "tm_draft_cloud_space_id": -1,
                "tm_draft_cloud_user_id": -1,
                "tm_draft_create": now_ms,
                "tm_draft_modified": now_ms,
                "tm_draft_removed": 0,
                "tm_duration": total_duration_us,
            }

            all_drafts.insert(0, new_entry)
            root_meta["all_draft_store"] = all_drafts
            root_meta_file.write_text(json.dumps(root_meta, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[CapCut Export] Successfully registered draft in root_meta_info.json!")
        except Exception as e_reg:
            print(f"[CapCut Export Warning] Could not update root_meta_info.json ({e_reg})")

    total_sec = total_duration_us / 1_000_000.0
    print("\n" + "=" * 65)
    print("CapCut Desktop Project Created Successfully!")
    print("=" * 65)
    print(f"Project Name   : {draft_folder_name}")
    print(f"Location       : {draft_folder}")
    print(f"Canvas Size    : {width}x{height} @ {fps}fps")
    print(f"Total Duration : {total_sec:.1f}s ({total_sec/60.0:.2f} min)")
    print(f"Video Tracks   : {len(video_segments)} clips")
    print(f"Audio Tracks   : {len(audio_segments)} voice clips")
    print(f"Subtitle Cues  : {len(text_segments)} text cues")
    print("=" * 65)
    print("You can now open CapCut Desktop and find this draft in 'Recent Drafts'.\n")

    if open_capcut:
        try:
            os.system('start "" capcut://')
        except Exception:
            pass

    return draft_folder


def main():
    parser = argparse.ArgumentParser(description="Export tube2dscript assets to CapCut Desktop project")
    parser.add_argument("--name", type=str, default=None, help="Custom name for the CapCut draft")
    parser.add_argument("--project-json", type=str, default=None, help="Path to project.json")
    parser.add_argument("--open", action="store_true", help="Launch CapCut Desktop after export")
    args = parser.parse_args()

    project_path = Path(args.project_json) if args.project_json else None
    export_to_capcut(project_json_path=project_path, draft_name=args.name, open_capcut=args.open)


if __name__ == "__main__":
    main()
