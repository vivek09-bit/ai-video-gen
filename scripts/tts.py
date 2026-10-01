#!/usr/bin/env python3
"""Turn ``script.json`` into narration audio plus exact word timings.

This is the only step that touches the network. It uses `edge-tts`, which
speaks through Microsoft's free neural voices and -- crucially -- reports a
timestamp for every single word. That means captions are frame-accurate
without running any speech-recognition model, and audio duration is derived
from the MP3 bitrate, so FFmpeg and ffprobe are not required either.

Outputs:
  public/audio/<scene>.mp3    narration per scene
  public/audio/<scene>.json   cache sidecar (hash + word timings)
  src/generated/project.json  normalised script + timeline, read by Remotion

Usage:
  py scripts/tts.py                      # build everything
  py scripts/tts.py --force              # ignore the cache
  py scripts/tts.py --voices             # list free voices
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
from pathlib import Path

import edge_tts

ROOT = Path(__file__).resolve().parent.parent
PUBLIC = ROOT / "public"
AUDIO_DIR = PUBLIC / "audio"
GENERATED = ROOT / "src" / "generated"

# MPEG-1/2/2.5 Layer III bitrate tables (kbps), indexed by the 4-bit header field.
BITRATES = {
    "1": [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320],
    "2": [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160],
    "2.5": [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160],
}

DEFAULT_SCRIPT = {
    "title": "Untitled",
    "voice": "en-US-GuyNeural",
    "rate": "+0%",
    "pitch": "+0Hz",
    "volume": "+0%",
    "fps": 30,
    "width": 1920,
    "height": 1080,
    "tailPaddingSeconds": 1.2,
    "captions": {
        "enabled": True,
        "wordsPerCue": 4,
        "bottom": 0.07,
        "fontSize": 54,
        "uppercase": False,
        "highlight": True,
    },
    "music": None,
    "theme": {
        "accent": "#38bdf8",
        "text": "#ffffff",
        "muted": "#cbd5e1",
        "panel": "rgba(3, 7, 18, 0.55)",
        "headlineFont": "Georgia, 'Times New Roman', serif",
        "bodyFont": "'Segoe UI', system-ui, -apple-system, sans-serif",
    },
}

DEFAULT_SCENE = {
    "id": "",
    "narration": "",
    "headline": "",
    "subtext": "",
    "bullets": [],
    "headlinePosition": "center",
    "background": {"type": "gradient", "from": "#0b1220", "to": "#132a44", "angle": 160},
    "padBefore": 0.0,
    "padAfter": 0.55,
    "holdSeconds": 3.0,
}


def merge(defaults: dict, value: dict) -> dict:
    """Shallow merge, one level of nesting deep -- enough for this schema."""
    out = dict(defaults)
    for key, val in value.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = {**out[key], **val}
        else:
            out[key] = val
    return out


def normalise_percent(value: str, *, unit: str) -> str:
    """`8` / `8%` / `+8%` all become `+8%`; `0Hz` becomes `+0Hz`."""
    text = str(value).strip()
    if not text:
        return f"+0{unit}"
    if text[0] not in "+-":
        text = f"+{text}"
    if not text.endswith(unit):
        text = f"{text}{unit}" if unit != "%" else f"{text}%"
    return text


def mp3_bitrate_kbps(data: bytes) -> int:
    """Read the bitrate out of the first MPEG audio frame header."""
    offset = data.find(b"\xff")
    while 0 <= offset < len(data) - 4:
        header = int.from_bytes(data[offset : offset + 4], "big")
        if (header >> 21) & 0x7FF == 0x7FF:
            version_bits = (header >> 19) & 0x3
            layer_bits = (header >> 17) & 0x3
            bitrate_index = (header >> 12) & 0xF
            if layer_bits == 0x1 and 0 < bitrate_index < 15:
                version = {0: "2.5", 2: "2", 3: "1"}.get(version_bits)
                if version:
                    return BITRATES[version][bitrate_index]
        offset = data.find(b"\xff", offset + 1)
    return 48  # edge-tts default: 24 kHz mono, 48 kbps


def mp3_duration(data: bytes) -> float:
    bitrate = mp3_bitrate_kbps(data)
    return len(data) * 8 / (bitrate * 1000) if bitrate else 0.0


def scene_slug(scene_id: str, index: int) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", scene_id).strip("-")
    return slug or f"scene-{index + 1}"


async def synthesise(text: str, script: dict) -> tuple[list[dict], bytes]:
    """Speak one scene, returning word boundaries and raw MP3 bytes."""
    communicate = edge_tts.Communicate(
        text,
        script["voice"],
        rate=script["rate"],
        pitch=script["pitch"],
        volume=script["volume"],
        boundary="WordBoundary",
    )

    words: list[dict] = []
    audio = bytearray()

    async for chunk in communicate.stream():
        kind = chunk["type"]
        if kind == "audio":
            audio.extend(chunk["data"])
        elif kind == "WordBoundary":
            start = chunk["offset"] / 10_000_000
            words.append(
                {
                    "text": chunk["text"],
                    "start": round(start, 4),
                    "end": round(start + chunk["duration"] / 10_000_000, 4),
                }
            )

    if not audio:
        raise RuntimeError("edge-tts returned no audio (check voice name and network)")
    return words, bytes(audio)


def cache_key(script: dict, text: str) -> str:
    payload = json.dumps(
        [script["voice"], script["rate"], script["pitch"], script["volume"], text],
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


async def build_scene(
    index: int,
    scene: dict,
    script: dict,
    force: bool,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Returns a timeline entry for one scene."""
    slug = scene_slug(scene["id"], index)
    audio_path = AUDIO_DIR / f"{slug}.mp3"
    meta_path = AUDIO_DIR / f"{slug}.json"
    rel_audio = f"audio/{slug}.mp3"

    narration = " ".join(scene["narration"].split())
    pad_before = float(scene.get("padBefore") or 0.0)
    pad_after = float(scene.get("padAfter") or 0.0)

    if not narration:
        return {
            "slug": slug,
            "audio": None,
            "audio_duration": 0.0,
            "words": [],
            "pad_before": pad_before,
            "pad_after": max(pad_after, float(scene.get("holdSeconds") or 3.0)),
        }

    key = cache_key(script, narration)

    if not force and audio_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("hash") == key:
                print(f"  [{index + 1}] {slug}: cached ({meta['duration']:.1f}s)")
                return {
                    "slug": slug,
                    "audio": rel_audio,
                    "audio_duration": float(meta["duration"]),
                    "words": meta["words"],
                    "pad_before": pad_before,
                    "pad_after": pad_after,
                }
        except (json.JSONDecodeError, KeyError):
            pass

    async with semaphore:
        words, audio = await synthesise(narration, script)

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(audio)

    spoken_end = words[-1]["end"] if words else 0.0
    # Trust real audio length, but never cut off a word the engine reported.
    duration = round(max(mp3_duration(audio), spoken_end + 0.15), 3)

    meta_path.write_text(
        json.dumps({"hash": key, "duration": duration, "words": words}, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  [{index + 1}] {slug}: {duration:.1f}s, {len(words)} words")

    return {
        "slug": slug,
        "audio": rel_audio,
        "audio_duration": duration,
        "words": words,
        "pad_before": pad_before,
        "pad_after": pad_after,
    }


async def run(args: argparse.Namespace) -> int:
    script_path = Path(args.script).resolve()
    if not script_path.exists():
        print(f"error: {script_path} not found", file=sys.stderr)
        return 1

    raw = json.loads(script_path.read_text(encoding="utf-8"))
    script = merge(DEFAULT_SCRIPT, raw)
    for key in ("rate", "pitch", "volume"):
        unit = {"rate": "%", "pitch": "Hz", "volume": "%"}[key]
        script[key] = normalise_percent(script[key], unit=unit)

    scenes_in = raw.get("scenes") or []
    if not scenes_in:
        print("error: script.json has no scenes", file=sys.stderr)
        return 1

    scenes = [merge(DEFAULT_SCENE, s) for s in scenes_in]
    for i, scene in enumerate(scenes):
        scene["id"] = scene_slug(scene.get("id") or "", i)

    # Write the merged scenes back: the renderer relies on every default being
    # present (it reads scene.bullets.length and scene.subtext directly).
    script["scenes"] = scenes

    print(f"Voice: {script['voice']}  rate {script['rate']}  ({len(scenes)} scenes)")

    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    built = await asyncio.gather(
        *(build_scene(i, s, script, args.force, semaphore) for i, s in enumerate(scenes))
    )

    cursor = 0.0
    total_words = 0
    timings = []
    for i, (scene, entry) in enumerate(zip(scenes, built)):
        if entry["audio"]:
            length = entry["pad_before"] + entry["audio_duration"] + entry["pad_after"]
        else:
            length = entry["pad_before"] + entry["pad_after"]

        words = entry["words"]
        if words:
            audio_end = entry["pad_before"] + entry["audio_duration"]
            words = [
                {
                    "text": w["text"],
                    "start": round(min(w["start"], audio_end), 4),
                    "end": round(min(w["end"], audio_end), 4),
                }
                for w in words
            ]

        count = len(scene["narration"].split())
        total_words += count
        scene["words"] = count

        timings.append(
            {
                "id": scene["id"],
                "index": i,
                "start": round(cursor, 4),
                "duration": round(length, 4),
                "audioStart": round(cursor + entry["pad_before"], 4),
                "audio": entry["audio"],
                "words": words,
            }
        )
        cursor += length

    project = {
        "script": script,
        "timings": {
            "totalDuration": round(cursor, 4),
            "totalWords": total_words,
            "scenes": timings,
        },
    }

    GENERATED.mkdir(parents=True, exist_ok=True)
    (GENERATED / "project.json").write_text(
        json.dumps(project, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    minutes, seconds = divmod(cursor, 60)
    wpm = total_words / (cursor / 60) if cursor else 0
    print(
        f"\nTimeline: {int(minutes)}m {seconds:04.1f}s  |  {total_words} words  |  {wpm:.0f} wpm"
    )
    print(f"Wrote {GENERATED / 'project.json'}")
    if wpm > 185:
        print("Note: over 185 wpm is fast for narration; consider a slower voice or fewer words.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("script", nargs="?", default=str(ROOT / "script.json"))
    parser.add_argument("--force", action="store_true", help="re-synthesise cached scenes")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--voices", action="store_true", help="list available voices and exit")
    args = parser.parse_args()

    if args.voices:
        for voice in asyncio.run(edge_tts.list_voices()):
            print(f"{voice['ShortName']:<34} {voice['Gender']:<7} {voice['Locale']}")
        return 0

    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
