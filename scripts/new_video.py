#!/usr/bin/env python3
"""One command, topic to finished video.

    npm run new -- --topic "पुरानी हवेली का राज़" --minutes 10

Flow:

    topic + length
        -> beats.plan()            structure for free, no model needed
        -> narration               written by a model, or left as a TODO
        -> scripts/tts.py          Kokoro narration + Rhubarb mouth shapes
        -> remotion render         the finished MP4

Only step two costs tokens, and it is the cheap part: roughly 1.2 tokens per
narration word. Everything else is code.

Narration has three sources, tried in order:

  1. --narration FILE   use narration you already have
  2. an LLM             if one is reachable (see "Configuring a writer" below)
  3. a TODO file        otherwise, so an agent can fill it in and re-run

Configuring a writer
--------------------
Any OpenAI-compatible endpoint works. Set these environment variables:

    LLM_BASE_URL   e.g. https://api.openai.com/v1, https://openrouter.ai/api/v1
    LLM_API_KEY    your key (omit for a local server)
    LLM_MODEL      e.g. gpt-4o-mini, llama3.1, qwen2.5

A local Ollama server is detected automatically at http://localhost:11434/v1 and
needs no key, which makes this genuinely free end to end.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import beats  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = ROOT / "script.json"
GENERATED = ROOT / "src" / "generated" / "project.json"

OLLAMA_URL = "http://localhost:11434/v1"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass


# --------------------------------------------------------------------------
# The narration writer
# --------------------------------------------------------------------------


LANGUAGE_NAMES = {"hi": "Hindi (Devanagari script)", "en": "English"}


def build_prompt(planned: dict, voice_style: str) -> str:
    """The instruction sent to a model when it writes the narration.

    Deliberately compact. The structure is already decided, so the model is only
    asked for prose and headlines, which keeps both the input and the output
    small.
    """
    target = planned["target"]
    language = LANGUAGE_NAMES.get(planned["language"], planned["language"])
    lines = [
        f"You are writing narration for a {target['minutes']:g}-minute {language} "
        f"horror story video in the style of Hindi horror narration channels.",
        "",
        f"Topic: {planned['topic']}",
        f"Tone: {voice_style}",
        "",
        "The video is already storyboarded. Each scene below has a narrative role "
        "and a target length. Write ONLY the narration, never describing the "
        "visuals, never addressing the audience as a video.",
        "",
        "Scene list:",
    ]
    for index, scene in enumerate(planned["scenes"], start=1):
        lines.append(
            f"  scene {index}: role={scene['beat']}, "
            f"about {scene['_targetWords']} words"
        )

    lines += [
        "",
        "Rules:",
        f"- Write everything in {language}.",
        "- Match each scene's word count within 15 percent; the audio length "
        "depends on it, so short scenes make a short video.",
        "- Plain prose only. No markdown, no emoji, no stage directions, no "
        "sound-effect labels.",
        "- Write numbers as words in the target language, because the narration "
        "is read exactly as written.",
        "- Keep the horror restrained and specific. Dread comes from detail, "
        "not from shouting.",
        "",
        "Output format, exactly this and nothing else:",
        "",
        "TITLE: <a short title>",
        "###1|<short scene headline>",
        "<narration for scene 1>",
        "###2|<short scene headline>",
        "<narration for scene 2>",
        "...",
    ]
    return "\n".join(lines)


def candidates() -> list[tuple[str, str, str | None]]:
    """Reachable writers as (label, base_url, api_key)."""
    found: list[tuple[str, str, str | None]] = []

    # A local Ollama needs no key, so prefer it: it keeps the whole pipeline free.
    try:
        request = urllib.request.Request(f"{OLLAMA_URL}/models", method="GET")
        with urllib.request.urlopen(request, timeout=2) as response:
            if response.status == 200:
                found.append(("ollama (local)", OLLAMA_URL, None))
    except (urllib.error.URLError, OSError, ValueError):
        pass

    base = os.environ.get("LLM_BASE_URL")
    key = os.environ.get("LLM_API_KEY")
    if base:
        found.append((f"{base} (env)", base.rstrip("/"), key))

    return found


def ask_llm(base_url: str, api_key: str | None, model: str, prompt: str) -> str:
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You write narration for short horror videos. You follow "
                    "output formats exactly.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.85,
        }
    ).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(
        f"{base_url}/chat/completions", data=payload, headers=headers, method="POST"
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"]


def parse_written(text: str, scene_count: int) -> tuple[str, list[tuple[str, str]]]:
    """Parse `TITLE:` plus `###N|headline` blocks into (title, [(headline, body)])."""
    title = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("TITLE:"):
            title = stripped.split(":", 1)[1].strip()
            break

    blocks: list[tuple[str, str]] = []
    current_headline: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        if current_headline is not None:
            body = " ".join(" ".join(buffer).split())
            blocks.append((current_headline, body))

    for line in text.splitlines():
        match = re.match(r"^\s*#{2,3}\s*(\d+)\s*[|:]\s*(.*)$", line)
        if match:
            flush()
            buffer = []
            current_headline = match.group(2).strip()
            continue
        if current_headline is not None:
            buffer.append(line)
    flush()

    if not blocks:
        # The model ignored the format. Fall back to splitting on blank lines so
        # a usable result still comes out of a sloppy response.
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        blocks = [("", p) for p in paragraphs]

    return title, blocks[:scene_count]


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def caption_options(style: str, enabled: bool) -> dict:
    """On-screen text settings.

    `statement` shows a few large words in the middle of frame, which is the
    stronger look when the narration is the whole show and there are no
    characters. `subtitle` is the conventional readable band at the bottom.
    """
    if style == "statement":
        return {
            "enabled": enabled,
            "wordsPerCue": 3,
            "bottom": 0.08,
            "fontSize": 86,
            "uppercase": False,
            "highlight": True,
            "position": "center",
            "outline": 0,
            "panel": False,
            "maxWidth": 0.8,
            "rise": 30,
        }
    return {
        "enabled": enabled,
        "wordsPerCue": 5,
        "bottom": 0.07,
        "fontSize": 62,
        "uppercase": False,
        "highlight": True,
        "position": "bottom",
        "outline": 0,
        "panel": True,
        "maxWidth": 0.92,
        "rise": 16,
    }


def build_script(
    planned: dict,
    title: str,
    blocks: list[tuple[str, str]],
    *,
    voice: str,
    speed: float,
    captions: dict,
) -> tuple[dict, list[int]]:
    """Combine the planned structure with written narration into script.json."""
    scenes = []
    missing: list[int] = []

    for index, scene in enumerate(planned["scenes"]):
        headline, narration = ("", "")
        if index < len(blocks):
            headline, narration = blocks[index]
        if not narration.strip():
            missing.append(index + 1)
            # A silent scene still renders as a held shot, so the video stays
            # watchable instead of crashing partway through.
            narration = ""

        # Pass the planned scene through wholesale and override only what the
        # writer supplies. Listing keys by hand silently drops anything new the
        # planner starts emitting -- which is exactly how the acting block was
        # lost on the first pass.
        entry = {key: value for key, value in scene.items() if not key.startswith("_")}
        entry["headline"] = headline or scene["headline"]
        entry["lines"] = [{"character": None, "text": narration}] if narration else []
        scenes.append(entry)

    script = {
        "title": title or planned["topic"],
        "voice": voice,
        "speed": speed,
        "fps": 30,
        "width": 1920,
        "height": 1080,
        "sentencePauseSeconds": 0.32,
        "captions": captions,
        "theme": {
            "accent": "#c9524a",
            "text": "#f2ece4",
            "muted": "#a89f96",
            "panel": "rgba(4, 4, 6, 0.62)",
            "headlineFont": "Georgia, 'Times New Roman', serif",
            "bodyFont": "'Segoe UI', system-ui, -apple-system, sans-serif",
        },
        "characters": planned.get("characters", {}),
        "scenes": scenes,
    }
    return script, missing


def run_step(label: str, command: list[str], *, shell: bool = False) -> bool:
    print(f"\n--- {label} ---")
    result = subprocess.run(command, shell=shell, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"{label} failed with exit code {result.returncode}", file=sys.stderr)
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--topic", required=True, help="what the video is about")
    parser.add_argument("--minutes", type=float, required=True, help="target length")
    parser.add_argument("--voice", default="hm_omega", help="Kokoro voice id")
    parser.add_argument("--speed", type=float, default=0.94)
    parser.add_argument("--language", default="hi", help="hi or en")
    parser.add_argument("--wpm", type=int, default=beats.DEFAULT_WPM)
    parser.add_argument("--scene-seconds", type=int, default=beats.DEFAULT_SCENE_SECONDS)
    parser.add_argument("--shots", type=int, default=2, help="shots per scene")
    parser.add_argument(
        "--caption-style",
        choices=["statement", "subtitle"],
        default="statement",
        help="statement = large words in centre frame; subtitle = band at the bottom",
    )
    parser.add_argument(
        "--captions", dest="captions", action="store_true", default=True,
        help="(default) show the narration on screen, word-timed",
    )
    parser.add_argument(
        "--no-captions", dest="captions", action="store_false",
        help="turn on-screen text off entirely",
    )
    parser.add_argument(
        "--characters", action="store_true",
        help="cast a character; off by default, since these videos are text and atmosphere",
    )
    parser.add_argument("--narration", help="use narration from this file instead of a model")
    parser.add_argument("--out", default="out/video.mp4")
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL", "gpt-4o-mini"))
    parser.add_argument("--dry-run", action="store_true", help="show the plan and stop")
    parser.add_argument("--no-render", action="store_true", help="write script.json and stop")
    args = parser.parse_args()

    planned = beats.plan(
        args.topic,
        args.minutes,
        wpm=args.wpm,
        scene_seconds=args.scene_seconds,
        shots_per_scene=args.shots,
        language=args.language,
        with_characters=args.characters,
    )

    print("=" * 74)
    print("PLAN")
    print("=" * 74)
    print(beats.describe(planned))

    if args.dry_run:
        print("\n(dry run: nothing written)")
        return 0

    # ---- narration -------------------------------------------------------
    print()
    print("=" * 74)
    print("NARRATION")
    print("=" * 74)

    prompt = build_prompt(planned, "slow, restrained, building dread")

    title = ""
    blocks: list[tuple[str, str]] = []

    if args.narration:
        source = Path(args.narration)
        if not source.exists():
            print(f"error: {source} not found", file=sys.stderr)
            return 1
        title, blocks = parse_written(source.read_text(encoding="utf-8"), len(planned["scenes"]))
        print(f"  source: {source}")

    if not blocks:
        available = candidates()
        if available:
            label, base_url, api_key = available[0]
            print(f"  writer: {label}, model {args.model}")
            print(f"  prompt: ~{len(prompt) // 4} tokens in, "
                  f"~{planned['target']['totalWords']} words out")
            started = time.time()
            try:
                response = ask_llm(base_url, api_key, args.model, prompt)
                title, blocks = parse_written(response, len(planned["scenes"]))
                print(f"  wrote {len(blocks)} scenes in {time.time() - started:.0f}s")
            except Exception as exc:  # noqa: BLE001 - report and fall through
                print(f"  writer failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        else:
            print("  no writer available (no LLM reachable, no --narration)")

    if not blocks:
        todo = ROOT / "out" / "narration-todo.txt"
        todo.parent.mkdir(parents=True, exist_ok=True)
        todo.write_text(prompt, encoding="utf-8")
        print()
        print(f"  Wrote the narration brief to: {todo.relative_to(ROOT)}")
        print("  Give that to an agent (or set LLM_BASE_URL / start Ollama), then")
        print("  re-run this command. Structure is already decided, so the writer")
        print("  only has to supply prose.")
        return 0

    script, missing = build_script(
        planned,
        title,
        blocks,
        voice=args.voice,
        speed=args.speed,
        captions=caption_options(args.caption_style, args.captions),
    )

    SCRIPT_PATH.write_text(
        json.dumps(script, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    words = sum(len(line["text"].split()) for s in script["scenes"] for line in s["lines"])
    print(f"  wrote {SCRIPT_PATH.relative_to(ROOT)}: {len(script['scenes'])} scenes, {words} words")
    if missing:
        print(f"  WARNING: scenes {missing} came back empty and will hold as silent shots")

    if args.no_render:
        return 0

    # ---- audio and video -------------------------------------------------
    if not run_step(
        "narration and lip sync (Kokoro + Rhubarb)",
        [sys.executable, str(ROOT / "scripts" / "tts.py")],
    ):
        return 1

    if not run_step(
        "render (Remotion)",
        f'npx remotion render Main "{args.out}"',
        shell=True,
    ):
        return 1

    # ---- report ----------------------------------------------------------
    print()
    print("=" * 74)
    print("DONE")
    print("=" * 74)
    if GENERATED.exists():
        timing = json.loads(GENERATED.read_text(encoding="utf-8"))["timings"]
        actual = timing["totalDuration"]
        target_seconds = args.minutes * 60
        delta = actual - target_seconds
        print(f"  target : {args.minutes:g} min ({target_seconds:.0f}s)")
        print(f"  actual : {actual / 60:.2f} min ({actual:.0f}s)")
        print(f"  delta  : {delta:+.0f}s  ({delta / target_seconds * 100:+.1f}%)")
        print(f"  output : {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
