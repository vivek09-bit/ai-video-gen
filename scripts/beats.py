#!/usr/bin/env python3
"""Turn a topic and a target length into a full shot plan.

This is the piece that makes structure free.

Writing a `script.json` by hand costs roughly 20 tokens per narration word, and
**92% of that is structure** — shots, effects, sound cues, ids — while only 8% is
the actual prose. So instead of asking a model to emit all that JSON, code
generates it from a story-beat library, and the model is left with the one thing
it is genuinely better at: writing the narration.

Two consequences worth knowing:

* Cost per minute drops by roughly 6-10x, because only the prose is billed.
* Scene count stops mattering for cost, so scenes can be paced for the story
  instead of padded to save tokens.

Everything is seeded from the topic, so the same request always produces the same
plan and a re-run never reshuffles the visuals.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field


# --------------------------------------------------------------------------
# Story beats
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Beat:
    """One narrative function and the visual grammar that serves it."""

    name: str
    role: str
    visuals: list[dict]
    ambience: tuple[str, float]
    effects: dict
    # Cues are placed relative to the scene: (position 0-1, sfx, volume)
    cues: list[tuple[float, str, float]] = field(default_factory=list)
    # How much this beat wants to jolt the camera, 0-1.
    tension: float = 0.0
    # How the character performs. `emotion` is the base read, and `intensity` is
    # scaled up automatically as the piece progresses.
    emotion: str = "neutral"
    gesture: str = "none"
    acting_intensity: float = 0.5
    # Whether a character appears at all. Atmosphere beats deliberately do not,
    # because a figure in every single shot reads as a slideshow.
    character: bool = False


BEATS: dict[str, Beat] = {
    "establish": Beat(
        name="establish",
        role="Set the place. Slow, wide, quiet.",
        visuals=[
            {"type": "forest", "fog": 0.7, "moonlight": 0.5},
            {"type": "house", "fog": 0.55, "lit": False, "moonlight": 0.45},
            {"type": "grave", "fog": 0.68, "moonlight": 0.4},
            {"type": "fog", "density": 0.65, "glow": 0.4},
        ],
        ambience=("amb_wind", 0.4),
        effects={"vignette": 0.85, "grain": 0.45, "dust": 0.22, "zoom": 0.03, "drift": "in"},
        cues=[(0.15, "sfx_whisper", 0.3)],
        tension=0.0,
        emotion="calm",
        gesture="none",
        acting_intensity=0.3,
        character=False,
    ),
    "dread": Beat(
        name="dread",
        role="Something is wrong. Hold, let it sit.",
        visuals=[
            {"type": "void", "creep": 0.5},
            {"type": "corridor", "flicker": 0.6, "depth": 0.5},
            {"type": "fog", "density": 0.8, "glow": 0.3},
            {"type": "house", "fog": 0.6, "lit": True},
        ],
        ambience=("amb_drone", 0.48),
        effects={
            "vignette": 0.9, "grain": 0.5, "dust": 0.18,
            "zoom": 0.025, "drift": "in", "exposureFlicker": 0.18,
        },
        cues=[(0.5, "sfx_whisper", 0.35)],
        tension=0.25,
        emotion="worried",
        gesture="none",
        acting_intensity=0.55,
        character=True,
    ),
    "approach": Beat(
        name="approach",
        role="Closing in on it. Tightening.",
        visuals=[
            {"type": "corridor", "flicker": 0.8, "depth": 0.65},
            {"type": "figure", "distance": 0.55, "fog": 0.6, "glow": 0.45},
            {"type": "forest", "fog": 0.75, "moonlight": 0.35},
        ],
        ambience=("amb_drone", 0.52),
        effects={"vignette": 0.9, "grain": 0.52, "dust": 0.26, "zoom": 0.05, "drift": "in"},
        cues=[(0.35, "sfx_riser", 0.45)],
        tension=0.5,
        emotion="worried",
        gesture="reach",
        acting_intensity=0.7,
        character=True,
    ),
    "reveal": Beat(
        name="reveal",
        role="The scare. Short, hard, loud.",
        visuals=[
            {"type": "eyes", "pairs": 4, "spread": 0.8},
            {"type": "figure", "distance": 0.2, "fog": 0.5, "glow": 0.4},
            {"type": "candle", "glow": 1.0},
        ],
        ambience=("amb_heartbeat", 0.55),
        effects={
            "vignette": 0.94, "grain": 0.55, "dust": 0.1,
            "shake": 0.35, "zoom": 0.05, "drift": "in",
        },
        cues=[(0.05, "sfx_stinger", 0.95), (0.6, "sfx_impact", 0.6)],
        tension=1.0,
        emotion="shock",
        gesture="recoil",
        acting_intensity=1.0,
        character=True,
    ),
    "react": Beat(
        name="react",
        role="Aftermath. Breathe out.",
        visuals=[
            {"type": "void", "creep": 0.8},
            {"type": "candle", "glow": 0.7},
            {"type": "rain", "intensity": 0.5, "lightning": False},
        ],
        ambience=("amb_drone", 0.42),
        effects={
            "vignette": 0.92, "grain": 0.5, "dust": 0.14,
            "zoom": 0.02, "drift": "out", "exposureFlicker": 0.12,
        },
        cues=[(0.4, "sfx_impact", 0.35)],
        tension=0.2,
        emotion="sad",
        gesture="clutch",
        acting_intensity=0.5,
        character=True,
    ),
    "resolve": Beat(
        name="resolve",
        role="Close it. Leave them unsettled.",
        visuals=[
            {"type": "grave", "fog": 0.72, "moonlight": 0.45},
            {"type": "rain", "intensity": 0.7, "lightning": True},
            {"type": "void", "creep": 0.9},
        ],
        ambience=("amb_wind", 0.38),
        effects={"vignette": 0.9, "grain": 0.5, "dust": 0.18, "zoom": 0.025, "drift": "out"},
        cues=[(0.7, "sfx_bell", 0.6)],
        tension=0.1,
        emotion="calm",
        gesture="none",
        acting_intensity=0.35,
        character=True,
    ),
}

# Which way a character faces, cycled so a run of scenes does not all look the
# same. Alternating also lets two characters in one scene face each other.
FACINGS = ["left", "center", "right"]

# Characters sit on alternating sides of frame, which keeps the eye moving and
# leaves the opposite side clear for atmosphere. Values are the rule of thirds.
POSITIONS = [0.66, 0.34]

# The narrator-protagonist. Horror narration is normally first person, so one
# character is cast and varied through emotion, gaze and gesture rather than
# swapping in different people. Replace or extend this in your own script.
DEFAULT_CHARACTERS = {
    "lead": {
        "name": "Aarav",
        "voice": "hm_omega",
        "hairstyle": "short",
        "accessory": "none",
        "palette": {
            "skin": "#c98d63",
            "skinShade": "#ad7450",
            "hair": "#1f1b18",
            "shirt": "#3d4a5c",
            "accent": "#8a9099",
            "outline": "#14181f",
        },
    }
}


def cast_for(index: int, beat: Beat, progress: float) -> list[dict]:
    """Place a character for this beat, or nobody for atmosphere beats."""
    if not beat.character:
        return []

    position = POSITIONS[index % len(POSITIONS)]
    return [
        {
            "id": "lead",
            "x": position,
            # Slightly smaller as the story tightens, so the room closes in.
            "scale": round(0.74 - progress * 0.06, 3),
            "flip": position > 0.5,
        }
    ]

# A horror arc: settle, unsettle, tighten, strike, breathe, close.
HORROR_ARC = ["establish", "dread", "approach", "reveal", "react", "resolve"]

# Longer arcs interleave extra dread/approach beats so a 30 minute video does not
# feel like the same six scenes on loop.
HORROR_ARC_LONG = [
    "establish", "dread", "approach", "dread", "reveal", "react",
    "dread", "approach", "approach", "reveal", "react", "resolve",
]

# Short pieces cannot fit the full arc, so they get a compressed one. Without
# this a one-minute video would be nothing but setup and would never land a scare.
SHORT_ARCS = {
    1: ["establish"],
    2: ["approach", "reveal"],
    3: ["establish", "approach", "reveal"],
    4: ["establish", "approach", "reveal", "resolve"],
}


def arc_for(scene_count: int) -> list[str]:
    if scene_count in SHORT_ARCS:
        return SHORT_ARCS[scene_count]
    return HORROR_ARC if scene_count <= 8 else HORROR_ARC_LONG


def shape_arc(scene_count: int) -> list[str]:
    """Beat sequence for a given number of scenes.

    Cycles the appropriate arc, then forces a proper shape on the ends: the piece
    must open by establishing and close by resolving, and it must contain at least
    one reveal. That guarantees every length tells a complete story rather than
    stopping wherever the cycle happened to land.
    """
    arc = arc_for(scene_count)
    sequence = [arc[i % len(arc)] for i in range(scene_count)]

    if scene_count >= 2 and "reveal" not in sequence:
        sequence[-2] = "reveal"
    sequence[0] = arc[0]
    if scene_count >= 3:
        sequence[-1] = "resolve"
    return sequence


def seed_for(*parts: object) -> int:
    """Stable seed so the same request always yields the same plan."""
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


# --------------------------------------------------------------------------

# Narration pacing. Hindi horror narration at speed 0.94 measured 113 wpm on the
# reference machine; a slower, heavier read suits the genre.
DEFAULT_WPM = 115

# Target seconds per scene. Longer scenes mean fewer headline slots to write and
# a calmer pace; horror holds shots longer than a recap video would.
DEFAULT_SCENE_SECONDS = 24


def plan(
    topic: str,
    minutes: float,
    *,
    wpm: int = DEFAULT_WPM,
    scene_seconds: int = DEFAULT_SCENE_SECONDS,
    shots_per_scene: int = 2,
    language: str = "hi",
    with_characters: bool = False,
) -> dict:
    """Build a full scene plan: structure complete, narration left blank.

    Returns a dict with `topic`, `target`, and `scenes` where each scene already
    has its shots, effects, ambience and sound cues. Only `headline` and the
    narration text are placeholders for the writer to fill.
    """
    minutes = max(0.25, float(minutes))
    total_seconds = minutes * 60
    total_words = int(round((total_seconds / 60) * wpm))

    scene_count = max(2, int(round(total_seconds / scene_seconds)))
    words_per_scene = max(12, int(round(total_words / scene_count)))

    rng = random.Random(seed_for(topic, minutes, wpm, shots_per_scene))
    sequence = shape_arc(scene_count)

    scenes: list[dict] = []
    for index in range(scene_count):
        beat = BEATS[sequence[index]]
        # Ramp tension across the whole piece so the ending lands harder.
        progress = index / max(1, scene_count - 1)
        tension = min(1.0, beat.tension * (0.7 + progress * 0.6))

        shot_count = shots_per_scene + (1 if beat.name == "reveal" else 0)
        shots = []
        previous_type: str | None = None
        for shot_index in range(shot_count):
            # Two shots of the same visual back to back read as a still frame,
            # which defeats the point of cutting.
            pool = [v for v in beat.visuals if v["type"] != previous_type] or beat.visuals
            visual = dict(rng.choice(pool))
            previous_type = visual["type"]

            effects = dict(beat.effects)
            # Climax shots jolt; earlier ones stay still.
            if shot_index == shot_count - 1 and tension > 0.4:
                effects["shake"] = round(min(0.6, tension * 0.45), 2)
            shots.append(
                {
                    "visual": visual,
                    "weight": 2 if shot_index == 0 else 1,
                    "effects": effects,
                    **(
                        {"transition": {"type": "crossfade", "seconds": 1.4}}
                        if shot_index
                        else {"transition": {"type": "cut"}}
                    ),
                }
            )

        # Approximate the finished scene length so cues land on real beats.
        estimated = (words_per_scene / wpm) * 60 + 0.55
        cues = [
            {
                "at": round(min(estimated * 0.92, estimated * position), 2),
                "sfx": sfx,
                "volume": round(min(1.0, volume * (0.75 + tension * 0.4)), 2),
            }
            for position, sfx, volume in beat.cues
        ]

        scenes.append(
            {
                "id": f"scene-{index + 1:03d}",
                "beat": beat.name,
                "headline": "",
                "subtext": "",
                "bullets": [],
                "headlinePosition": "center",
                "shots": shots,
                "ambience": {"src": beat.ambience[0], "volume": beat.ambience[1]},
                "soundCues": cues,
                "background": {"type": "color", "color": "#04060a"},
                "cast": cast_for(index, beat, progress) if with_characters else [],
                "acting": {
                    "emotion": beat.emotion,
                    "intensity": round(min(1.0, beat.acting_intensity * (0.75 + progress * 0.5)), 2),
                    "head": FACINGS[index % len(FACINGS)],
                    "eyes": FACINGS[index % len(FACINGS)],
                    "gesture": beat.gesture,
                },
                "padBefore": 0.0,
                "padAfter": 0.55,
                "holdSeconds": 3.0,
                "lines": [],
                "_targetWords": words_per_scene,
            }
        )

    return {
        "topic": topic,
        "language": language,
        "characters": DEFAULT_CHARACTERS if with_characters else {},
        "target": {
            "minutes": minutes,
            "wpm": wpm,
            "totalWords": total_words,
            "scenes": scene_count,
            "wordsPerScene": words_per_scene,
            "shots": sum(len(s["shots"]) for s in scenes),
            "castScenes": sum(1 for s in scenes if s["cast"]),
        },
        "scenes": scenes,
    }


def describe(planned: dict) -> str:
    """Human-readable summary of a plan, for --dry-run."""
    target = planned["target"]
    lines = [
        f"topic        : {planned['topic']}",
        f"target       : {target['minutes']:g} min  ~{target['totalWords']} words  "
        f"@ {target['wpm']} wpm",
        f"structure    : {target['scenes']} scenes, {target['shots']} shots "
        f"({target['wordsPerScene']} words each)",
        f"cast         : {target.get('castScenes', 0)} of {target['scenes']} scenes "
        f"({', '.join(planned.get('characters', {})) or 'none'})",
        "",
        f"{'#':>3} {'beat':<10} {'shots':>5} {'words':>6} {'emotion':<9} "
        f"{'gesture':<8} {'head':<7} visuals",
        "-" * 92,
    ]
    for index, scene in enumerate(planned["scenes"], start=1):
        kinds = ", ".join(shot["visual"]["type"] for shot in scene["shots"])
        acting = scene.get("acting", {})
        camera = "with" if scene["cast"] else "  --"
        lines.append(
            f"{index:>3} {scene['beat']:<10} {len(scene['shots']):>5} "
            f"{scene['_targetWords']:>6} {acting.get('emotion', ''):<9} "
            f"{acting.get('gesture', ''):<8} {acting.get('head', ''):<7} "
            f"{camera}  {kinds}"
        )

    cues = sum(len(s["soundCues"]) for s in planned["scenes"])
    ambiences = sorted({s["ambience"]["src"] for s in planned["scenes"]})
    emotions = sorted({s["acting"]["emotion"] for s in planned["scenes"]})
    lines += [
        "",
        f"sound cues   : {cues}",
        f"ambience     : {', '.join(ambiences)}",
        f"emotions used: {', '.join(emotions)}",
    ]
    return "\n".join(lines)
