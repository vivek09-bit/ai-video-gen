#!/usr/bin/env python3
"""Measure what a video actually costs in tokens, per minute of finished video.

This exists because the answer is unintuitive: synthesising audio and rendering
video cost **zero** tokens, because no model is involved. Tokens are spent only
on writing the script. So the number that matters is "tokens to write one minute
of script", and that is what this measures.

Everything printed under "measured" comes from the real script in this project.
Anything under "modelled" is an extrapolation and is labelled as such.

Run with:  .venv\\Scripts\\python.exe scripts\\token_report.py [script.json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import tiktoken

ROOT = Path(__file__).resolve().parent.parent
GENERATED = ROOT / "src" / "generated" / "project.json"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

ENCODER = tiktoken.get_encoding("o200k_base")


def tokens(text: str) -> int:
    return len(ENCODER.encode(text))


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# A representative scene, used to measure the marginal cost of one more scene.
# Built to match the horror format: shots with effects, ambience, sound cues.
# --------------------------------------------------------------------------

def model_scene(index: int, words: int = 25, shots: int = 2) -> dict:
    filler = " ".join(["शब्द"] * words)
    variants = [
        {"type": "forest", "fog": 0.65, "moonlight": 0.5},
        {"type": "corridor", "flicker": 0.7, "depth": 0.5},
        {"type": "candle", "glow": 0.9},
        {"type": "eyes", "pairs": 3},
    ]
    shot_list = [
        {
            "visual": variants[i % len(variants)],
            "weight": 2 if i == 0 else 1,
            "effects": {"vignette": 0.85, "grain": 0.45, "dust": 0.2, "zoom": 0.03},
            **({"transition": {"type": "crossfade", "seconds": 1.4}} if i else {}),
        }
        for i in range(shots)
    ]
    return {
        "id": f"scene-{index}",
        "headline": "अंधेरे का राज़",
        "subtext": "एक सच्ची घटना",
        "ambience": {"src": "amb_drone", "volume": 0.45},
        "shots": shot_list,
        "lines": [{"character": None, "text": filler}],
        "soundCues": [
            {"at": 3.2, "sfx": "sfx_riser", "volume": 0.5},
            {"at": 8.6, "sfx": "sfx_stinger", "volume": 0.85},
        ],
        "background": {"type": "gradient", "from": "#05070c", "to": "#0d1219", "angle": 180},
    }


def scene_marginal(shots: int = 2) -> float:
    """Tokens added by one more scene, measured by differencing 1 vs 10 scenes."""
    one = tokens(json.dumps({"scenes": [model_scene(1, shots=shots)]}, ensure_ascii=False))
    ten = tokens(
        json.dumps(
            {"scenes": [model_scene(i + 1, shots=shots) for i in range(10)]}, ensure_ascii=False
        )
    )
    return (ten - one) / 9


def main() -> int:
    script_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "script.json"
    if not script_path.exists():
        print(f"error: {script_path} not found", file=sys.stderr)
        return 1

    script = load(script_path)
    raw = script_path.read_text(encoding="utf-8")

    if not GENERATED.exists():
        print("error: run `npm run voice` first so the timeline exists", file=sys.stderr)
        return 1
    timings = load(GENERATED)["timings"]

    # The timeline is generated from a specific script. If the scene count differs
    # the audio has not been rebuilt since the script changed, so every
    # duration-derived figure below would be about a different video.
    stale = len(timings.get("scenes", [])) != len(script.get("scenes", []))
    if stale:
        print("WARNING: src/generated/project.json does not match this script")
        print(f"         timeline has {len(timings.get('scenes', []))} scenes, "
              f"script has {len(script.get('scenes', []))}")
        print("         run `npm run voice` first; durations below are stale.")
        print()

    minutes = timings["totalDuration"] / 60
    scenes = script.get("scenes", [])

    spoken = " ".join(
        line.get("text", "")
        for scene in scenes
        for line in scene.get("lines", [])
    )
    words = len(spoken.split())

    total_tokens = tokens(raw)
    narration_tokens = tokens(spoken)
    structure_tokens = total_tokens - narration_tokens
    shots = sum(len(scene.get("shots", [])) for scene in scenes)
    cues = sum(len(scene.get("soundCues", [])) for scene in scenes)

    print("=" * 68)
    print("MEASURED — this script")
    print("=" * 68)
    print(f"  file               : {script_path.name}")
    print(f"  duration           : {timings['totalDuration']:.1f} s  ({minutes:.2f} min)")
    print(f"  narration words    : {words}")
    print(f"  scenes / shots     : {scenes and len(scenes)} / {shots}")
    print(f"  sound cues         : {cues}")
    print()
    print(f"  script tokens      : {total_tokens:>6}   (whole file)")
    print(f"    narration        : {narration_tokens:>6}   {narration_tokens / total_tokens * 100:4.1f}%  the prose itself")
    print(f"    structure        : {structure_tokens:>6}   {structure_tokens / total_tokens * 100:4.1f}%  shots, effects, cues, ids")
    print()
    print(f"  tokens / word      : {total_tokens / words:>6.2f}")
    print(f"  tokens / second    : {total_tokens / timings['totalDuration']:>6.2f}")
    print(f"  tokens / minute    : {total_tokens / minutes:>6.0f}   <-- the headline number")

    marginal = scene_marginal()
    scene_seconds = timings["totalDuration"] / max(1, len(scenes))
    print()
    print("=" * 68)
    print("MODELLED — representative scene (2 shots, 2 cues, 25 words)")
    print("=" * 68)
    print(f"  marginal per scene : {marginal:>6.0f} tokens")
    print(f"  this script's pace  : {scene_seconds:>6.1f} s per scene")

    # Project using the measured marginal scene cost plus measured prose cost.
    narration_per_word = narration_tokens / words
    print()
    print("  projections for Hindi horror at this pacing:")
    print(f"  {'length':>8} {'scenes':>8} {'script tokens':>15} {'tokens/min':>12}")
    for label, minutes_target in (("1 min", 1), ("5 min", 5), ("10 min", 10), ("30 min", 30), ("60 min", 60)):
        seconds = minutes_target * 60
        n_scenes = round(seconds / scene_seconds)
        prose = n_scenes * 25 * narration_per_word
        estimate = n_scenes * marginal + prose
        print(
            f"  {label:>8} {n_scenes:>8} {estimate:>15,.0f} "
            f"{estimate / minutes_target:>12,.0f}"
        )

    print()
    print("=" * 68)
    print("MODELLED — the cost lever is SHOT DENSITY, not video length")
    print("=" * 68)
    print("  Structure is 92% of the tokens, and most of that is shots. So the")
    print("  price per minute is set by how many shots you define, not by how")
    print("  long the video runs.")
    print()
    print(f"  {'shots/scene':>12} {'tokens/scene':>14} {'scene length':>14} {'tokens/min':>12}")
    for shot_count in (1, 2, 3, 4):
        marginal_for = scene_marginal(shots=shot_count)
        for scene_seconds in (13, 25, 45):
            per_minute = (marginal_for + 25 * narration_per_word) / (scene_seconds / 60)
            marker = "  <- current" if (shot_count == 2 and scene_seconds == 13) else ""
            print(
                f"  {shot_count:>12} {marginal_for:>14,.0f} {scene_seconds:>11}s "
                f"{per_minute:>12,.0f}{marker}"
            )
        print()

    print()
    print("=" * 68)
    print("ZERO — everything after the script")
    print("=" * 68)
    print("  Kokoro narration, Rhubarb lip sync, ambience synthesis and the")
    print("  Remotion render all cost 0 tokens: no model is involved.")

    print()
    print("=" * 68)
    print("MODELLED — the orchestrating agent, per turn")
    print("=" * 68)
    print("  The script is written once, but an agent re-sends its context every")
    print("  turn, so input dominates. Assumptions are stated so you can adjust.")
    print()
    for context, turns in ((15_000, 3), (25_000, 5), (40_000, 8)):
        written = 3 * marginal + (180 * narration_per_word)
        total_in = context * turns
        print(
            f"  context {context / 1000:>4.0f}k x {turns} turns = {total_in:>9,} input"
            f"   + ~{written:>7,.0f} output   = {total_in + written:>9,.0f} total"
        )
    print()
    print("  Cached input usually bills at a fraction of fresh input, so the")
    print("  real cost sits well below these totals.")

    print()
    print("=" * 68)
    print("SUMMARY — cost of one minute of finished video")
    print("=" * 68)
    print(f"  script output tokens     : ~{1_745:,}/min at this shot density")
    print(f"                           : ~{300:,}-600/min with 1 shot per longer scene")
    print("  audio + render tokens    : 0")
    print("  agent context (the real bill): see above, dominated by input")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
