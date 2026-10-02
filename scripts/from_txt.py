#!/usr/bin/env python3
"""Convert a plain-text script into ``script.json``.

Writing JSON by hand is tedious and an AI agent can generate either format, so
if you would rather just write a script here is a readable format:

    @title The Deep Sea
    @voice bm_george
    @speed 1.0

    @char maya af_heart bun
    @char sam  am_michael short glasses

    # The Deep Sea
    ! gradient: #04101c -> #0f3352
    ! cast maya 0.5 1.0
    > Two researchers, one very large ocean
    Two thirds of our planet is ocean, and almost all of it is completely dark.

    # Four Thousand Meters Down
    ! pattern: grid #061321 #153a5c
    ! cast maya 0.29 0.92 flip
    ! cast sam 0.71 0.92
    sam: At four thousand meters, the pressure would crush a submarine.
    maya: And yet something down there is making its own light.

Directives:
    @title / @voice / @speed / @fps   script-level settings
    @char ID VOICE [HAIRSTYLE] [ACCESSORY]
                                      define a character. HAIRSTYLE is one of
                                      short, bob, long, bun, bald; ACCESSORY is
                                      none or glasses.
    # Heading                         starts a scene; text becomes the headline
    ! background                      color / gradient / pattern / image / video
    ! cast ID X [SCALE] [flip]        place a character. X is 0 (left) to 1 (right)
    > Subtitle                        small muted line under the headline
    * Bullet                          adds a bullet card
    ID: dialogue text                 spoken by that character
    plain text                        narration in the default voice
    // comment                        ignored

Usage:
    .venv\\Scripts\\python.exe scripts\\from_txt.py myscript.txt -o script.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SCRIPT_KEYS = {
    "title": str,
    "voice": str,
    "speed": float,
    "fps": int,
    "width": int,
    "height": int,
    "tailPaddingSeconds": float,
    "sentencePauseSeconds": float,
}

HAIRSTYLES = {"short", "bob", "long", "bun", "bald"}
ACCESSORIES = {"none", "glasses"}

# Rotating palette so characters are visually distinct without listing colours.
PALETTES = [
    {
        "skin": "#f0c49a", "skinShade": "#dba87c", "hair": "#2f2a3a",
        "shirt": "#e0653f", "accent": "#ffd166", "outline": "#232a33",
    },
    {
        "skin": "#c98d63", "skinShade": "#ad7450", "hair": "#1f1b18",
        "shirt": "#3f6f8f", "accent": "#8ed1fc", "outline": "#232a33",
    },
    {
        "skin": "#f6d7b8", "skinShade": "#e2bb98", "hair": "#7a4a23",
        "shirt": "#5a8f6a", "accent": "#b8e6c1", "outline": "#232a33",
    },
    {
        "skin": "#a3703f", "skinShade": "#8a5c33", "hair": "#141210",
        "shirt": "#8a5fa8", "accent": "#d9b8f0", "outline": "#232a33",
    },
]


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
        return {
            "type": "pattern",
            "pattern": pattern,
            "bg": parts[1] if len(parts) > 1 else "#0b1220",
            "line": parts[2] if len(parts) > 2 else "#1d3b5c",
        }

    if kind == "image":
        return {"type": "image", "src": rest, "kenBurns": "in", "dim": 0.25}

    if kind == "video":
        return {"type": "video", "src": rest, "dim": 0.2}

    raise ValueError(f"unknown background type '{kind}'")


def parse_character(parts: list[str], existing: dict) -> tuple[str, dict]:
    if len(parts) < 2:
        raise ValueError("@char needs at least an id and a voice: @char maya af_heart")

    char_id = parts[0]
    palette = PALETTES[len(existing) % len(PALETTES)]
    hairstyle = "short"
    accessory = "none"

    for token in parts[2:]:
        if token in HAIRSTYLES:
            hairstyle = token
        elif token in ACCESSORIES:
            accessory = token
        else:
            raise ValueError(
                f"@char {char_id}: '{token}' is not a hairstyle ({', '.join(sorted(HAIRSTYLES))}) "
                f"or accessory ({', '.join(sorted(ACCESSORIES))})"
            )

    return char_id, {
        "name": char_id.capitalize(),
        "voice": parts[1],
        "palette": palette,
        "hairstyle": hairstyle,
        "accessory": accessory,
    }


def parse_cast(parts: list[str]) -> dict:
    if len(parts) < 2:
        raise ValueError("! cast needs an id and an x position: ! cast maya 0.5")

    member: dict = {"id": parts[0], "x": float(parts[1])}
    if len(parts) > 2:
        member["scale"] = float(parts[2])
    if "flip" in parts[3:] or "flip" in parts:
        member["flip"] = True
    return member


def parse(text: str) -> dict:
    script: dict = {"scenes": [], "characters": {}}
    characters: dict = script["characters"]
    scene: dict | None = None

    def new_scene(headline: str) -> dict:
        return {"headline": headline.strip(), "lines": [], "cast": []}

    def flush_narration() -> None:
        """Turns any accumulated prose into narration lines on the scene."""
        if scene is None or not pending:
            return
        joined = " ".join(" ".join(p.split()) for p in pending).strip()
        pending.clear()
        if not joined:
            return
        speaker = scene["cast"][0]["id"] if scene["cast"] and scene["cast"][0]["id"] in characters else None
        scene["lines"].append({"character": speaker, "text": joined})

    pending: list[str] = []

    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()

        if not line or line.startswith("//"):
            continue

        try:
            if line.startswith("@"):
                flush_narration()
                if scene is not None:
                    raise ValueError("settings must appear before the first '#' heading")
                body = line[1:]
                key, _, value = body.partition(" ")
                key = key.rstrip(":").strip()
                value = value.strip()

                if key == "char":
                    char_id, spec = parse_character(value.split(), characters)
                    characters[char_id] = spec
                elif key in SCRIPT_KEYS:
                    script[key] = SCRIPT_KEYS[key](value)
                else:
                    raise ValueError(f"unknown setting '@{key}'")
                continue

            if line.startswith("#"):
                flush_narration()
                scene = new_scene(line.lstrip("#").strip())
                script["scenes"].append(scene)
                continue

            if scene is None:
                raise ValueError(f"content before the first '#' heading: {line!r}")

            if line.startswith("!"):
                flush_narration()
                body = line[1:].strip()
                if body.lower().startswith("cast"):
                    scene["cast"].append(parse_cast(body[4:].split()))
                else:
                    scene["background"] = parse_background(body)
                continue

            if line.startswith(">"):
                flush_narration()
                scene["subtext"] = line[1:].strip()
                continue

            if line.startswith("*") or line.startswith("-"):
                flush_narration()
                scene["bullets"] = scene.get("bullets", []) + [line[1:].strip()]
                continue

            # `id: text` is dialogue when `id` names a defined character.
            if ":" in line:
                head, _, rest = line.partition(":")
                if head.strip() in characters:
                    flush_narration()
                    scene["lines"].append({"character": head.strip(), "text": rest.strip()})
                    continue

            pending.append(line)

        except ValueError as exc:
            print(f"error on line {lineno}: {exc}", file=sys.stderr)
            raise SystemExit(1)

    flush_narration()

    for index, item in enumerate(script["scenes"]):
        item.setdefault("id", f"scene-{index + 1}")
        item.setdefault("headlinePosition", "center")
        item.setdefault("bullets", [])
        item.setdefault("subtext", "")

    return script


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="plain-text script to convert")
    parser.add_argument("-o", "--out", default=str(ROOT / "script.json"))
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        print(f"error: {source} not found", file=sys.stderr)
        return 1

    script = parse(source.read_text(encoding="utf-8"))
    if not script["scenes"]:
        print("error: no scenes found (each scene starts with a '#' heading)", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.write_text(json.dumps(script, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = sum(len(s["lines"]) for s in script["scenes"])
    words = sum(len(line["text"].split()) for s in script["scenes"] for line in s["lines"])
    print(
        f"Wrote {out} with {len(script['scenes'])} scenes, "
        f"{len(script['characters'])} characters, {lines} lines, {words} words"
    )
    print("Next: npm run voice")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
