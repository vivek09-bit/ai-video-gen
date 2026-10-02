"""Verify the acting layer, measuring the eyes precisely.

The first attempt at this measured "dark pixels" across a wide window and got
~78,000 of them, which is obviously the dark background rather than pupils. It
also compared different background shots against each other, so the numbers were
mostly measuring scenery.

This version derives the eye positions from the actual geometry in
`project.json` (cast position, scale, head facing) and measures a tight box
around each eye. Inside that box, bright neutral pixels are sclera and very dark
pixels are pupil plus outline, so the comparison is meaningful.

Run with:  .venv\\Scripts\\python.exe scripts\\_verify_acting.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
GENERATED = ROOT / "src" / "generated" / "project.json"

WIDTH, HEIGHT = 1920, 1080
# Character SVG space: 400 x 640, character is 88% of frame height, eyes at y=158.
CHAR_FRAME_FRACTION = 0.88
EYE_LOCAL_X = (166, 234)
EYE_LOCAL_Y = 158
HEAD_CENTRE_LOCAL_X = 200
FACING_SHIFT = {"left": -7, "right": 7, "center": 0}
SHIRT = (61, 74, 92)  # #3d4a5c, a reliable presence signature for the character

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

if not GENERATED.exists():
    raise SystemExit("run `npm run voice` first")

project = json.loads(GENERATED.read_text(encoding="utf-8"))
script_scenes = {s["id"]: s for s in project["script"]["scenes"]}
timings = project["timings"]["scenes"]

print("=== acting as it reached the renderer ===")
for timing in timings:
    scene = script_scenes.get(timing["id"], {})
    acting = scene.get("acting")
    cast = scene.get("cast") or []
    print(
        f"  {timing['id']}  frames {int(timing['start'] * 30):>5}-"
        f"{int((timing['start'] + timing['duration']) * 30):>5}  "
        f"acting={acting}  cast={len(cast)}"
    )

if not any(script_scenes.get(t["id"], {}).get("acting") for t in timings):
    raise SystemExit("FAIL: no scene carries an `acting` block, so it is not reaching the renderer")


def eye_regions(scene: dict) -> list[tuple[int, int, int]]:
    """Screen-space (cx, cy, half-size) for each eye of the first cast member."""
    if not scene.get("cast"):
        return []
    member = scene["cast"][0]
    acting = member.get("acting") or scene.get("acting") or {}
    facing = acting.get("head", "center")

    char_height = HEIGHT * CHAR_FRAME_FRACTION * member.get("scale", 1)
    k = char_height / 640
    top = HEIGHT - char_height
    cx = member["x"] * WIDTH
    shift = FACING_SHIFT.get(facing, 0)

    regions = []
    for local_x in EYE_LOCAL_X:
        x = cx + (local_x - HEAD_CENTRE_LOCAL_X) * k + shift
        y = top + EYE_LOCAL_Y * k
        half = int(26 * k)
        regions.append((int(x), int(y), max(12, half)))
    return regions


def measure(image: Image.Image, regions: list[tuple[int, int, int]]) -> dict:
    white = dark = 0
    for cx, cy, half in regions:
        for y in range(max(0, cy - half), min(HEIGHT, cy + half)):
            for x in range(max(0, cx - half), min(WIDTH, cx + half)):
                r, g, b = image.getpixel((x, y))[:3]
                mean = (r + g + b) / 3
                if mean > 222 and abs(r - b) < 26:
                    white += 1
                elif mean < 95:
                    dark += 1
    return {"white": white, "dark": dark}


def presence(image: Image.Image, cx: float, half: int = 320) -> int:
    """Count shirt-coloured pixels, which only the character has."""
    count = 0
    for y in range(300, HEIGHT - 40, 3):
        for x in range(max(0, int(cx) - half), min(WIDTH, int(cx) + half), 3):
            r, g, b = image.getpixel((x, y))[:3]
            if abs(r - SHIRT[0]) < 26 and abs(g - SHIRT[1]) < 26 and abs(b - SHIRT[2]) < 26:
                count += 1
    return count


SAMPLES = []
for timing in timings:
    scene = script_scenes.get(timing["id"], {})
    if not scene.get("cast"):
        continue
    middle = timing["start"] + timing["duration"] * 0.5
    SAMPLES.append((int(middle * 30), timing["id"], (scene.get("acting") or {}).get("emotion")))

print()
print(f"{'frame':>6} {'scene':<12} {'emotion':<9} {'shirtPx':>8} {'eyeWhite':>9} {'eyeDark':>8}")
print("-" * 62)

results = {}
for frame, scene_id, emotion in SAMPLES:
    path = OUT / f"act-{frame}.png"
    if not path.exists():
        print(f"{frame:>6} {scene_id:<12} {emotion or '':<9} MISSING STILL")
        continue
    scene = script_scenes[scene_id]
    with Image.open(path) as raw:
        image = raw.convert("RGB")
        stats = measure(image, eye_regions(scene))
        stats["shirt"] = presence(image, scene["cast"][0]["x"] * WIDTH)
    results[frame] = (scene_id, emotion, stats)
    print(
        f"{frame:>6} {scene_id:<12} {emotion or '':<9} {stats['shirt']:>8} "
        f"{stats['white']:>9} {stats['dark']:>8}"
    )

print()
failures: list[str] = []

for frame, (scene_id, _e, stats) in results.items():
    if stats["shirt"] < 60:
        failures.append(f"{scene_id}: character not detected (shirt pixels {stats['shirt']})")

by_emotion: dict[str, list[dict]] = {}
for _f, (_s, emotion, stats) in results.items():
    by_emotion.setdefault(emotion or "none", []).append(stats)

if "worried" in by_emotion and "shock" in by_emotion:
    w, s = by_emotion["worried"], by_emotion["shock"]
    w_white = sum(x["white"] for x in w) / len(w)
    s_white = sum(x["white"] for x in s) / len(s)
    w_dark = sum(x["dark"] for x in w) / len(w)
    s_dark = sum(x["dark"] for x in s) / len(s)

    print("--- emotion comparison (tight eye boxes) ---")
    print(f"  worried: sclera {w_white:7.0f}   pupil {w_dark:7.0f}")
    print(f"  shock  : sclera {s_white:7.0f}   pupil {s_dark:7.0f}")
    print("  (shock uses eyeOpen 1.30 vs 1.00, and pupil 0.68 vs 1.06)")

    if s_white <= w_white:
        failures.append(f"shock did not widen the eyes (sclera {s_white:.0f} vs {w_white:.0f})")
    if s_dark >= w_dark:
        failures.append(f"shock did not shrink the pupil (pupil {s_dark:.0f} vs {w_dark:.0f})")
else:
    print(f"(emotions found: {sorted(by_emotion)})")

print()
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for item in failures:
        print(f"  - {item}")
    raise SystemExit(1)
print("All acting checks passed.")
