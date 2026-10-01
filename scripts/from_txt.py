#!/usr/bin/env python3
"""Convert a plain-text script into ``script.json``.

Writing JSON by hand is tedious, and an AI agent can generate either format.
If you would rather just have a script, write a ``.txt`` file with this shape
and let this script handle the plumbing:

    @title The Deep Sea
    @voice en-US-GuyNeural
    @rate +6%

    # The Deep Sea
    ! gradient: #04101c -> #0f3352
    > An introduction
    * Roughly 71% of the surface
    Two thirds of our planet is ocean, and almost all of it is dark.

    # Four Thousand Meters
    ! pattern: grid #08111f #1b3a58
    At four thousand metres the pressure would crush a steel submarine.

Directives:
    @key value          script-level setting (before the first `#`)
    # Heading           starts a new scene; the text becomes its headline
    ! background        one of:
                          gradient: #aabbcc -> #112233
                          color: #0b1220
                          pattern: grid|dots|diagonal BG LINE
                          image: images/sea.jpg
                          video: video/waves.mp4
    > Subtitle line     small muted text under the headline
    * Bullet            adds a bullet card
    // comment          ignored

Usage:
    py scripts/from_txt.py myscript.txt -o script.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SCRIPT_KEYS = {
    "title",
    "voice",
    "rate",
    "pitch",
    "volume",
    "fps",
    "width",
    "height",
    "tailPaddingSeconds",
}

HEADLINE_POSITIONS = {"top", "center", "bottom"}


def parse_background(value: str) -> dict:
    kind, _, rest = value.partition(":")
    kind = kind.strip().lower()
    rest = rest.strip()

    if kind == "gradient":
        from_color, _, to_color = rest.replace("->", "|").partition("|")
        return {
            "type": "gradient",
            "from": from_color.strip(),
            "to": to_color.strip() or from_color.strip(),
            "angle": 160,
        }

    if kind == "color":
        return {"type": "color", "color": rest}

    if kind == "pattern":
        parts = rest.split()
        pattern = parts[0] if parts else "grid"
        if pattern not in {"grid", "dots", "diagonal"}:
            raise ValueError(f"unknown pattern '{pattern}'")
        bg = parts[1] if len(parts) > 1 else "#0b1220"
        line = parts[2] if len(parts) > 2 else "#1d3b5c"
        return {"type": "pattern", "pattern": pattern, "bg": bg, "line": line}

    if kind == "image":
        return {"type": "image", "src": rest, "kenBurns": "in", "dim": 0.25}

    if kind == "video":
        return {"type": "video", "src": rest, "dim": 0.2}

    raise ValueError(f"unknown background type '{kind}'")


def new_scene(headline: str) -> dict:
    return {"headline": headline.strip(), "narration": "", "bullets": []}


def parse(text: str) -> dict:
    script: dict = {"scenes": []}
    scene: dict | None = None
    paragraphs: list[str] = []

    def flush() -> None:
        if scene is not None and paragraphs:
            joined = " ".join(" ".join(p.split()) for p in paragraphs).strip()
            existing = scene.get("narration", "")
            scene["narration"] = f"{existing} {joined}".strip()
            paragraphs.clear()

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line or line.startswith("//"):
            continue

        if line.startswith("@"):
            if scene is not None:
                raise ValueError(
                    f"settings must appear before the first scene heading: {line!r}"
                )
            body = line[1:]
            key, _, value = body.partition(" ")
            key = key.rstrip(":").strip()
            value = value.strip()
            if key in SCRIPT_KEYS:
                script[key] = value
                continue
            if key == "position":
                continue
            raise ValueError(f"unknown setting '@{key}'")

        if line.startswith("#"):
            flush()
            scene = new_scene(line.lstrip("#").strip())
            script["scenes"].append(scene)
            continue

        if scene is None:
            raise ValueError(f"content before the first '#' heading: {line!r}")

        if line.startswith("!"):
            scene["background"] = parse_background(line[1:].strip())
        elif line.startswith(">"):
            scene["subtext"] = line[1:].strip()
        elif line.startswith("*") or line.startswith("-"):
            scene["bullets"].append(line[1:].strip())
        elif line.startswith("@"):  # pragma: no cover - handled above
            pass
        else:
            paragraphs.append(line)

    flush()
    return script


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="plain-text script to convert")
    parser.add_argument(
        "-o", "--out", default=str(ROOT / "script.json"), help="output JSON path"
    )
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        print(f"error: {source} not found", file=sys.stderr)
        return 1

    script = parse(source.read_text(encoding="utf-8"))
    if not script["scenes"]:
        print("error: no scenes found (each scene starts with a '#' heading)", file=sys.stderr)
        return 1

    for i, scene in enumerate(script["scenes"]):
        scene.setdefault("id", f"scene-{i + 1}")
        scene.setdefault("headlinePosition", "center")

    out = Path(args.out)
    out.write_text(json.dumps(script, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    words = sum(len(s.get("narration", "").split()) for s in script["scenes"])
    print(f"Wrote {out} with {len(script['scenes'])} scenes and {words} narration words")
    print("Next: npm run voice")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
