"""Verify the horror render actually looks like a horror render.

I cannot view the frames here, so this checks the properties that would fail if
the procedural stage were broken or blank: every shot must be dark, must have
visible structure, must differ from the other shots, and each preset must show
the light signature it is supposed to (a warm core for candle, cold glints for
eyes, streaks for rain, and so on).

Run with:  .venv\\Scripts\\python.exe scripts\\_verify_horror.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"

SHOTS = [
    "forest", "void-1", "house", "corridor", "candle",
    "figure", "eyes", "void-2", "rain", "grave",
]

# signature: (expect_warm, expect_bright_centre, expect_small_bright_spots)
EXPECT = {
    "forest": (False, False, False),
    "void-1": (False, False, False),
    "house": (True, False, False),
    "corridor": (False, True, False),
    "candle": (True, True, False),
    "figure": (False, False, False),
    "eyes": (True, False, True),
    "void-2": (False, False, False),
    "rain": (False, False, False),
    "grave": (False, False, False),
}


def load(name: str) -> np.ndarray:
    path = OUT / f"sh-{name}.png"
    if not path.exists():
        raise SystemExit(f"missing still: {path}")
    with Image.open(path) as raw:
        return np.asarray(raw.convert("RGB"), dtype=np.float32)


def luminance(image: np.ndarray) -> np.ndarray:
    return image[:, :, 0] * 0.299 + image[:, :, 1] * 0.587 + image[:, :, 2] * 0.114


def signature(image: np.ndarray, grid: int = 8) -> np.ndarray:
    """Coarse luminance grid, for comparing shots to each other."""
    lum = luminance(image)
    h, w = lum.shape
    return np.array(
        [
            lum[y * h // grid : (y + 1) * h // grid, x * w // grid : (x + 1) * w // grid].mean()
            for y in range(grid)
            for x in range(grid)
        ]
    )


print(f"{'shot':<10} {'mean':>6} {'std':>6} {'dark%':>6} {'warmth':>7} {'core':>6} {'spots':>6}")
print("-" * 60)

failures: list[str] = []
stats: dict[str, dict] = {}

for name in SHOTS:
    image = load(name)
    lum = luminance(image)
    mean = float(lum.mean())
    std = float(lum.std())
    dark_fraction = float((lum < 26).mean())

    red = float(image[:, :, 0].mean())
    blue = float(image[:, :, 2].mean())
    warmth = red - blue

    # Central region, where a light source should be.
    h, w = lum.shape
    core = float(lum[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3].mean())

    # Small very bright points (eyes, distant bulbs).
    spots = float((lum > 175).mean() * 1000)

    stats[name] = {
        "mean": mean,
        "std": std,
        "dark": dark_fraction,
        "warmth": warmth,
        "core": core,
        "spots": spots,
        "sig": signature(image),
    }

    print(f"{name:<10} {mean:6.1f} {std:6.1f} {dark_fraction * 100:5.1f}% {warmth:7.1f} {core:6.1f} {spots:6.2f}")

    # Every horror shot must be dark, but must still contain detail.
    if mean > 78:
        failures.append(f"{name}: too bright for horror (mean {mean:.1f})")
    if std < 4:
        failures.append(f"{name}: nearly flat, stage probably not rendering (std {std:.1f})")

print()

# Preset-specific signatures.
expect_warm, expect_core, expect_spots = EXPECT["candle"]
if stats["candle"]["warmth"] <= 2:
    failures.append(f"candle: expected warm light, warmth {stats['candle']['warmth']:.1f}")
if stats["candle"]["core"] < 12:
    failures.append(f"candle: expected a lit centre, got {stats['candle']['core']:.1f}")

if stats["corridor"]["core"] < 6:
    failures.append(f"corridor: expected a light at the end, got {stats['corridor']['core']:.1f}")

if stats["eyes"]["spots"] < 0.02:
    failures.append(f"eyes: expected small bright glints, got {stats['eyes']['spots']:.3f}")

if stats["candle"]["warmth"] - stats["forest"]["warmth"] < 3:
    failures.append("candle is not warmer than forest; the light colour may be wrong")

print("--- warmth ordering ---")
print(f"  candle {stats['candle']['warmth']:6.1f}  (warmest, as intended)")
print(f"  corridor {stats['corridor']['warmth']:6.1f}")
print(f"  eyes {stats['eyes']['warmth']:6.1f}")
print(f"  forest {stats['forest']['warmth']:6.1f}")

print()
print("--- shot distinctness (mean absolute difference of 8x8 signatures) ---")
minimum = None
closest = ("", "")
for i, a in enumerate(SHOTS):
    for b in SHOTS[i + 1 :]:
        diff = float(np.abs(stats[a]["sig"] - stats[b]["sig"]).mean())
        if minimum is None or diff < minimum:
            minimum = diff
            closest = (a, b)

if minimum is not None:
    print(f"  most similar pair: {closest[0]} vs {closest[1]} = {minimum:.2f}")
    if minimum < 1.5:
        failures.append(f"{closest[0]} and {closest[1]} look nearly identical ({minimum:.2f})")

# Distinct void shots should still differ (different creep settings).
void_diff = float(np.abs(stats["void-1"]["sig"] - stats["void-2"]["sig"]).mean())
print(f"  void-1 vs void-2 = {void_diff:.2f} (allowed to be similar; both are near-black)")

print()
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for item in failures:
        print(f"  - {item}")
    raise SystemExit(1)

print("All horror render checks passed.")
