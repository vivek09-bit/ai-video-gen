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
import re
import sys
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


# Default base beats (horror & suspense)
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

# --------------------------------------------------------------------------
# Story structure
# --------------------------------------------------------------------------
#
# Beats are visual grammar, not a story. Cycling beats produces atmosphere, which
# is why a longer video used to loop the same six shots and never conclude.
#
# A story is a shape, and that shape has to hold at every length. So runtime is
# apportioned across narrative movements by proportion and the whole set of
# movements is always present: a one-minute piece gets shorter scenes, not half
# a story.


@dataclass(frozen=True)
class Movement:
    """One narrative movement and what has to happen inside it."""

    key: str
    duty: str
    beat: str
    weight: float


# --------------------------------------------------------------------------
# Story structure & Movements
# --------------------------------------------------------------------------
#
# Beats are visual grammar, not a story. Cycling beats produces atmosphere, which
# is why a longer video used to loop the same six shots and never conclude.
#
# A story is a shape, and that shape has to hold at every length. So runtime is
# apportioned across narrative movements by proportion and the whole set of
# movements is always present: a one-minute piece gets shorter scenes, not half
# a story.


@dataclass(frozen=True)
class Movement:
    """One narrative movement and what has to happen inside it."""

    key: str
    duty: str
    beat: str
    weight: float


MOVEMENTS: dict[str, Movement] = {
    "hook": Movement(
        key="hook",
        duty="Open with a line that makes them stay: a warning, a question, a promise.",
        beat="establish",
        weight=0.05,
    ),
    "setup": Movement(
        key="setup",
        duty="Say who this is about and where they are, before anything goes wrong.",
        beat="establish",
        weight=0.13,
    ),
    "inciting": Movement(
        key="inciting",
        duty="The disturbance. Something arrives, breaks, or is found that cannot be undone.",
        beat="dread",
        weight=0.12,
    ),
    "rising": Movement(
        key="rising",
        duty="Complications. Each attempt makes it worse. Raise the stakes in every scene.",
        beat="approach",
        weight=0.32,
    ),
    "climax": Movement(
        key="climax",
        duty="The confrontation. The worst moment, faced directly rather than avoided.",
        beat="reveal",
        weight=0.16,
    ),
    "resolution": Movement(
        key="resolution",
        duty="What it cost, and what changed. Close the loop you opened in the setup.",
        beat="react",
        weight=0.14,
    ),
    "button": Movement(
        key="button",
        duty="One last line that reframes everything they just heard. Land it and stop.",
        beat="resolve",
        weight=0.08,
    ),
}

# Which movements fit in a given number of scenes.
# For short-form / 4-6 scene videos, hook is ALWAYS included because the first 3 seconds
# determine watch time and viral algorithm distribution.
MOVEMENT_SETS: list[tuple[int, list[str]]] = [
    (9, ["hook", "setup", "inciting", "rising", "climax", "resolution", "button"]),
    (7, ["hook", "setup", "inciting", "rising", "climax", "resolution", "button"]),
    (5, ["hook", "inciting", "rising", "climax", "resolution"]),
    (4, ["hook", "inciting", "climax", "resolution"]),
    (3, ["hook", "climax", "resolution"]),
    (2, ["hook", "climax"]),
    (1, ["hook"]),
    (0, ["hook"]),
]

# Absolute minimum scenes supported by the engine
MIN_SCENES = 1

# Rising action alternates beats so a long middle does not look like one shot
# held for six scenes.
RISING_BEATS = ["approach", "dread"]


# --------------------------------------------------------------------------
# Multi-Genre & Dynamic Theme Architecture
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ThemeProfile:
    """Color palette and typography for captions, headlines, overlays, and stage backdrops."""

    key: str
    name: str
    aliases: list[str]
    accent: str
    text: str
    muted: str
    panel: str
    headlineFont: str
    bodyFont: str
    bg_color: str = "#04060a"
    genres: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, str]:
        return {
            "key": self.key,
            "name": self.name,
            "accent": self.accent,
            "text": self.text,
            "muted": self.muted,
            "panel": self.panel,
            "headlineFont": self.headlineFont,
            "bodyFont": self.bodyFont,
            "bg_color": self.bg_color,
        }


THEMES: dict[str, ThemeProfile] = {
    "blood_red": ThemeProfile(
        key="blood_red",
        name="Blood Red",
        aliases=["blood", "red", "crimson", "scarlet", "horror_red", "vampire"],
        accent="#ef4444",
        text="#fef2f2",
        muted="#fca5a5",
        panel="rgba(45, 10, 10, 0.75)",
        headlineFont="Georgia, 'Times New Roman', serif",
        bodyFont="'Segoe UI', system-ui, -apple-system, sans-serif",
        bg_color="#0c0404",
        genres=["horror", "drama"],
    ),
    "amber_ember": ThemeProfile(
        key="amber_ember",
        name="Amber Ember",
        aliases=["amber", "ember", "fire", "orange", "inferno", "flame"],
        accent="#f97316",
        text="#fff7ed",
        muted="#fdba74",
        panel="rgba(35, 18, 10, 0.75)",
        headlineFont="Georgia, 'Times New Roman', serif",
        bodyFont="'Segoe UI', system-ui, -apple-system, sans-serif",
        bg_color="#0a0502",
        genres=["horror", "mythology", "motivational"],
    ),
    "cyber_neon": ThemeProfile(
        key="cyber_neon",
        name="Cyber Neon",
        aliases=["cyber", "neon", "cyan", "matrix", "hologram", "future", "scifi"],
        accent="#06b6d4",
        text="#ecfeff",
        muted="#67e8f9",
        panel="rgba(8, 24, 36, 0.78)",
        headlineFont="'Segoe UI', system-ui, sans-serif",
        bodyFont="'Segoe UI', system-ui, -apple-system, sans-serif",
        bg_color="#020b12",
        genres=["scifi", "documentary"],
    ),
    "golden_radiance": ThemeProfile(
        key="golden_radiance",
        name="Golden Radiance",
        aliases=["gold", "golden", "saffron", "radiance", "divine", "vedic", "sacred"],
        accent="#ff9933",
        text="#fffdf5",
        muted="#fed7aa",
        panel="rgba(38, 20, 8, 0.75)",
        headlineFont="Georgia, 'Times New Roman', serif",
        bodyFont="'Segoe UI', system-ui, -apple-system, sans-serif",
        bg_color="#100703",
        genres=["mythology", "fantasy", "motivational"],
    ),
    "amethyst_dream": ThemeProfile(
        key="amethyst_dream",
        name="Amethyst Dream",
        aliases=["amethyst", "purple", "violet", "mystic", "enchanted", "lavender", "fantasy"],
        accent="#a855f7",
        text="#faf5ff",
        muted="#d8b4fe",
        panel="rgba(46, 16, 101, 0.72)",
        headlineFont="Georgia, 'Times New Roman', serif",
        bodyFont="'Segoe UI', system-ui, -apple-system, sans-serif",
        bg_color="#090412",
        genres=["fantasy", "drama"],
    ),
    "emerald_forest": ThemeProfile(
        key="emerald_forest",
        name="Emerald Forest",
        aliases=["emerald", "forest", "green", "jade", "nature", "enchanted_forest"],
        accent="#10b981",
        text="#ecfdf5",
        muted="#a7f3d0",
        panel="rgba(6, 44, 30, 0.75)",
        headlineFont="Georgia, 'Times New Roman', serif",
        bodyFont="'Segoe UI', system-ui, -apple-system, sans-serif",
        bg_color="#02120b",
        genres=["fantasy", "documentary"],
    ),
    "midnight_blue": ThemeProfile(
        key="midnight_blue",
        name="Midnight Blue",
        aliases=["midnight", "blue", "sapphire", "ocean", "deep_blue", "abyss", "docu"],
        accent="#38bdf8",
        text="#f8fafc",
        muted="#94a3b8",
        panel="rgba(15, 23, 42, 0.75)",
        headlineFont="Georgia, 'Times New Roman', serif",
        bodyFont="'Segoe UI', system-ui, -apple-system, sans-serif",
        bg_color="#050c18",
        genres=["documentary", "horror", "scifi"],
    ),
    "sunset_glow": ThemeProfile(
        key="sunset_glow",
        name="Sunset Glow",
        aliases=["sunset", "rose", "coral", "dusk", "romance", "pink"],
        accent="#f43f5e",
        text="#fff1f2",
        muted="#fecdd3",
        panel="rgba(40, 10, 20, 0.72)",
        headlineFont="Georgia, 'Times New Roman', serif",
        bodyFont="'Segoe UI', system-ui, -apple-system, sans-serif",
        bg_color="#120408",
        genres=["drama", "fantasy", "comedy"],
    ),
    "monochrome_noir": ThemeProfile(
        key="monochrome_noir",
        name="Monochrome Noir",
        aliases=["monochrome", "noir", "black_and_white", "bw", "silver", "minimal", "slate"],
        accent="#e2e8f0",
        text="#ffffff",
        muted="#94a3b8",
        panel="rgba(20, 20, 24, 0.82)",
        headlineFont="Georgia, 'Times New Roman', serif",
        bodyFont="'Segoe UI', system-ui, -apple-system, sans-serif",
        bg_color="#050507",
        genres=["horror", "documentary", "drama"],
    ),
    "toxic_lime": ThemeProfile(
        key="toxic_lime",
        name="Toxic Lime",
        aliases=["toxic", "lime", "radioactive", "acid", "neon_green"],
        accent="#84cc16",
        text="#f7fee7",
        muted="#bef264",
        panel="rgba(20, 30, 10, 0.75)",
        headlineFont="'Segoe UI', system-ui, sans-serif",
        bodyFont="'Segoe UI', system-ui, -apple-system, sans-serif",
        bg_color="#060b03",
        genres=["horror", "scifi", "comedy"],
    ),
    "solar_flare": ThemeProfile(
        key="solar_flare",
        name="Solar Flare",
        aliases=["solar", "sun", "sunburst", "yellow", "electric_yellow", "supernova"],
        accent="#eab308",
        text="#ffffff",
        muted="#cbd5e1",
        panel="rgba(25, 20, 5, 0.75)",
        headlineFont="'Segoe UI', system-ui, sans-serif",
        bodyFont="'Segoe UI', system-ui, -apple-system, sans-serif",
        bg_color="#0a0802",
        genres=["motivational", "scifi"],
    ),
    "synthwave": ThemeProfile(
        key="synthwave",
        name="Synthwave",
        aliases=["synthwave", "retrowave", "magenta", "cyberpunk_pink", "neon_pink"],
        accent="#ec4899",
        text="#fdf2f8",
        muted="#fbcfe8",
        panel="rgba(40, 10, 45, 0.75)",
        headlineFont="'Segoe UI', system-ui, sans-serif",
        bodyFont="'Segoe UI', system-ui, -apple-system, sans-serif",
        bg_color="#0f0312",
        genres=["scifi", "comedy"],
    ),
    "ice_glacier": ThemeProfile(
        key="ice_glacier",
        name="Ice Glacier",
        aliases=["ice", "glacier", "frost", "arctic", "crystal", "frozen"],
        accent="#7dd3fc",
        text="#f0f9ff",
        muted="#bae6fd",
        panel="rgba(12, 30, 45, 0.75)",
        headlineFont="Georgia, 'Times New Roman', serif",
        bodyFont="'Segoe UI', system-ui, -apple-system, sans-serif",
        bg_color="#030c14",
        genres=["fantasy", "documentary", "scifi"],
    ),
    "royal_velvet": ThemeProfile(
        key="royal_velvet",
        name="Royal Velvet",
        aliases=["royal", "velvet", "imperial", "regal", "ruby_gold", "luxury"],
        accent="#fbbf24",
        text="#fffbeb",
        muted="#fde68a",
        panel="rgba(45, 12, 28, 0.75)",
        headlineFont="Georgia, 'Times New Roman', serif",
        bodyFont="'Segoe UI', system-ui, -apple-system, sans-serif",
        bg_color="#0e0208",
        genres=["mythology", "fantasy", "drama"],
    ),
    "sepia_vintage": ThemeProfile(
        key="sepia_vintage",
        name="Sepia Vintage",
        aliases=["sepia", "vintage", "antique", "parchment", "history_bronze", "aged"],
        accent="#d97706",
        text="#fef3c7",
        muted="#fde68a",
        panel="rgba(35, 25, 15, 0.75)",
        headlineFont="Georgia, 'Times New Roman', serif",
        bodyFont="'Segoe UI', system-ui, -apple-system, sans-serif",
        bg_color="#0d0804",
        genres=["documentary", "mythology"],
    ),
    "pastel_whimsy": ThemeProfile(
        key="pastel_whimsy",
        name="Pastel Whimsy",
        aliases=["pastel", "whimsy", "playful", "candy", "cheerful", "cartoon"],
        accent="#fb7185",
        text="#fff7ed",
        muted="#fed7aa",
        panel="rgba(35, 20, 25, 0.72)",
        headlineFont="'Segoe UI', system-ui, sans-serif",
        bodyFont="'Segoe UI', system-ui, -apple-system, sans-serif",
        bg_color="#0c0608",
        genres=["comedy"],
    ),
    "matrix_green": ThemeProfile(
        key="matrix_green",
        name="Matrix Green",
        aliases=["matrix_green", "terminal", "hacker", "code", "neon_terminal"],
        accent="#22c55e",
        text="#f0fdf4",
        muted="#86efac",
        panel="rgba(5, 30, 15, 0.78)",
        headlineFont="'Segoe UI', system-ui, sans-serif",
        bodyFont="'Segoe UI', system-ui, -apple-system, sans-serif",
        bg_color="#020d05",
        genres=["scifi"],
    ),
    "desert_dusk": ThemeProfile(
        key="desert_dusk",
        name="Desert Dusk",
        aliases=["desert", "terracotta", "sandstorm", "saffron_dusk", "warm_earth"],
        accent="#f97316",
        text="#fff7ed",
        muted="#ffedd5",
        panel="rgba(40, 22, 10, 0.75)",
        headlineFont="Georgia, 'Times New Roman', serif",
        bodyFont="'Segoe UI', system-ui, -apple-system, sans-serif",
        bg_color="#0e0602",
        genres=["mythology", "drama", "documentary"],
    ),
    "crimson_gothic": ThemeProfile(
        key="crimson_gothic",
        name="Crimson Gothic",
        aliases=["crimson_gothic", "gothic", "dark_red", "blood_moon"],
        accent="#c9524a",
        text="#f2ece4",
        muted="#a89f96",
        panel="rgba(4, 4, 6, 0.65)",
        headlineFont="Georgia, 'Times New Roman', serif",
        bodyFont="'Segoe UI', system-ui, -apple-system, sans-serif",
        bg_color="#04060a",
        genres=["horror"],
    ),
    "citrus_burst": ThemeProfile(
        key="citrus_burst",
        name="Citrus Burst",
        aliases=["citrus", "burst", "lemon", "lime_orange", "vibrant"],
        accent="#f59e0b",
        text="#fffbeb",
        muted="#fde68a",
        panel="rgba(35, 25, 10, 0.70)",
        headlineFont="'Segoe UI', system-ui, sans-serif",
        bodyFont="'Segoe UI', system-ui, -apple-system, sans-serif",
        bg_color="#0a0803",
        genres=["comedy", "motivational"],
    ),
    "nordic_frost": ThemeProfile(
        key="nordic_frost",
        name="Nordic Frost",
        aliases=["nordic", "frost", "arctic_frost", "slate_frost"],
        accent="#38bdf8",
        text="#f1f5f9",
        muted="#94a3b8",
        panel="rgba(15, 23, 42, 0.72)",
        headlineFont="Georgia, 'Times New Roman', serif",
        bodyFont="'Segoe UI', system-ui, -apple-system, sans-serif",
        bg_color="#040810",
        genres=["documentary", "scifi", "fantasy"],
    ),
}

GENRE_THEME_POOLS: dict[str, list[str]] = {
    "anime": [
        "solar_flare",
        "cyber_neon",
        "synthwave",
        "amber_ember",
        "blood_red",
        "golden_radiance",
        "amethyst_dream",
    ],
    "fantasy": [
        "amethyst_dream",
        "emerald_forest",
        "golden_radiance",
        "royal_velvet",
        "ice_glacier",
        "sunset_glow",
    ],
    "scifi": [
        "cyber_neon",
        "synthwave",
        "solar_flare",
        "matrix_green",
        "ice_glacier",
        "midnight_blue",
    ],
    "documentary": [
        "midnight_blue",
        "sepia_vintage",
        "nordic_frost",
        "monochrome_noir",
        "emerald_forest",
        "desert_dusk",
    ],
    "mythology": [
        "golden_radiance",
        "royal_velvet",
        "desert_dusk",
        "amber_ember",
        "amethyst_dream",
    ],
    "motivational": [
        "solar_flare",
        "golden_radiance",
        "amber_ember",
        "cyber_neon",
        "royal_velvet",
        "citrus_burst",
    ],
    "drama": [
        "sunset_glow",
        "amethyst_dream",
        "desert_dusk",
        "royal_velvet",
        "monochrome_noir",
        "blood_red",
    ],
    "comedy": [
        "pastel_whimsy",
        "citrus_burst",
        "sunset_glow",
        "amber_ember",
        "toxic_lime",
        "synthwave",
    ],
    "explainer": [
        "cyber_neon",
        "solar_flare",
        "matrix_green",
        "nordic_frost",
        "midnight_blue",
        "synthwave",
    ],
    "horror": [
        "blood_red",
        "amber_ember",
        "crimson_gothic",
        "midnight_blue",
        "monochrome_noir",
        "toxic_lime",
    ],
}


def get_theme(name_or_alias: str | None, default_key: str = "blood_red") -> ThemeProfile:
    """Look up a ThemeProfile by name or alias, falling back to default."""
    if not name_or_alias:
        return THEMES.get(default_key, THEMES["blood_red"])
    norm = name_or_alias.lower().strip().replace("-", "_").replace(" ", "_")
    if norm in THEMES:
        return THEMES[norm]
    for th in THEMES.values():
        if norm in th.aliases:
            return th
    return THEMES.get(default_key, THEMES["blood_red"])


def _make_custom_hex_theme(hex_str: str, genre_key: str = "horror") -> ThemeProfile:
    """Generate a cohesive ThemeProfile from any custom hex color code."""
    hex_clean = hex_str.lstrip("#")
    if len(hex_clean) == 3:
        hex_clean = "".join(c * 2 for c in hex_clean)
    try:
        r = int(hex_clean[0:2], 16)
        g = int(hex_clean[2:4], 16)
        b = int(hex_clean[4:6], 16)
    except ValueError:
        r, g, b = 239, 68, 68

    panel_r = max(4, int(r * 0.15))
    panel_g = max(4, int(g * 0.15))
    panel_b = max(4, int(b * 0.15))
    bg_r = max(2, int(r * 0.05))
    bg_g = max(2, int(g * 0.05))
    bg_b = max(2, int(b * 0.05))

    is_serif = genre_key in ("horror", "fantasy", "mythology", "documentary", "drama")

    return ThemeProfile(
        key=f"hex_{hex_clean.lower()}",
        name=f"Custom (#{hex_clean})",
        aliases=[],
        accent=f"#{hex_clean}",
        text="#f8fafc",
        muted="#cbd5e1",
        panel=f"rgba({panel_r}, {panel_g}, {panel_b}, 0.75)",
        headlineFont="Georgia, 'Times New Roman', serif" if is_serif else "'Segoe UI', system-ui, sans-serif",
        bodyFont="'Segoe UI', system-ui, -apple-system, sans-serif",
        bg_color=f"#{bg_r:02x}{bg_g:02x}{bg_b:02x}",
        genres=[genre_key],
    )


def resolve_theme(
    theme_input: str | dict | None = None,
    genre: str | None = None,
    topic: str = "",
    seed: int | None = None,
) -> ThemeProfile:
    """Resolve theme input dynamically to a concrete ThemeProfile.

    Modes:
      - None / "auto": Dynamically selects a genre-matching theme using deterministic hash of topic
      - "random" / "shuffle" / "rand": Picks a random theme from the entire 20+ theme catalog
      - "random_genre" / "genre_random": Picks a random theme from the current genre's pool
      - Specific theme key or alias (e.g. "blood_red", "cyber_neon", "golden_radiance", "amethyst", "emerald", "sunset", "noir", "toxic", "synthwave", "ice", "royal", "sepia", "pastel", "matrix", "desert", etc.)
      - Custom hex color: e.g. "#ef4444" -> Generates a harmonious palette
      - Custom dictionary with theme keys
    """
    if isinstance(theme_input, dict):
        return ThemeProfile(
            key=theme_input.get("key", "custom"),
            name=theme_input.get("name", "Custom Theme"),
            aliases=[],
            accent=theme_input.get("accent", "#f59e0b"),
            text=theme_input.get("text", "#f8fafc"),
            muted=theme_input.get("muted", "#94a3b8"),
            panel=theme_input.get("panel", "rgba(15, 23, 42, 0.72)"),
            headlineFont=theme_input.get("headlineFont", "Georgia, 'Times New Roman', serif"),
            bodyFont=theme_input.get("bodyFont", "'Segoe UI', system-ui, -apple-system, sans-serif"),
            bg_color=theme_input.get("bg_color", "#04060a"),
            genres=[genre] if genre else [],
        )

    t_str = str(theme_input).strip() if theme_input else ""
    genre_key = genre.lower().strip() if genre else "horror"

    # 1. Custom hex string
    if t_str.startswith("#") and len(t_str) in (4, 7):
        return _make_custom_hex_theme(t_str, genre_key)

    # 2. Random theme across whole catalog
    if t_str.lower() in ("random", "shuffle", "rand"):
        rng = random.Random()
        if seed is not None:
            rng = random.Random(seed)
        chosen_key = rng.choice(list(THEMES.keys()))
        return THEMES[chosen_key]

    # 3. Random theme within genre
    if t_str.lower() in ("random_genre", "genre_random", "rand_genre"):
        pool = GENRE_THEME_POOLS.get(genre_key, list(THEMES.keys()))
        rng = random.Random()
        if seed is not None:
            rng = random.Random(seed)
        chosen_key = rng.choice(pool)
        return THEMES.get(chosen_key, THEMES["blood_red"])

    # 4. Explicit named theme or alias
    if t_str and t_str.lower() not in ("auto", "default", "none"):
        found = get_theme(t_str)
        if found:
            return found

    # 5. Dynamic Auto Selection:
    # Deterministically maps (topic, genre) to a curated genre-specific theme
    pool = GENRE_THEME_POOLS.get(genre_key, ["blood_red", "amber_ember", "midnight_blue"])
    if not pool:
        pool = list(THEMES.keys())

    h_seed = seed if seed is not None else seed_for(topic, genre_key, "theme_auto")
    chosen_key = pool[h_seed % len(pool)]
    return THEMES.get(chosen_key, THEMES["blood_red"])


def list_themes() -> list[dict]:
    """Return all available visual themes with metadata and color info."""
    return [
        {
            "key": th.key,
            "name": th.name,
            "aliases": th.aliases,
            "accent": th.accent,
            "text": th.text,
            "panel": th.panel,
            "headlineFont": th.headlineFont,
            "bodyFont": th.bodyFont,
            "bg_color": th.bg_color,
            "genres": th.genres,
        }
        for th in THEMES.values()
    ]


@dataclass(frozen=True)
class GenreProfile:
    """Full creative DNA for a content genre: story beats, tone, prompting, visuals, theme."""

    key: str
    name: str
    aliases: list[str]
    tone: str
    writer_instructions: str
    system_prompt: str
    visual_style_prefix: str
    visual_style_suffix: str
    search_keywords_map: dict[str, str]
    default_keywords: str
    theme: dict[str, str]
    movements: dict[str, Movement]
    beats: dict[str, Beat]
    rising_beats: list[str]
    default_wpm: int = 115
    default_speed: float = 0.94
    default_voice: str = "hm_omega"
    bg_color: str = "#04060a"

    def voice_for_language(self, language: str = "hi") -> str:
        """Return the optimal default voice for this genre and language."""
        lang = (language or "hi").lower().strip()
        if lang in ("hi", "hindi", "hinglish"):
            return "hm_omega"
        elif lang in ("en", "en-us", "en-gb", "english"):
            if self.key == "documentary":
                return "bm_george"
            elif self.key == "horror":
                return "am_onyx"
            elif self.key == "motivational":
                return "am_adam"
            elif self.key in ("anime", "comedy"):
                return "af_bella"
            elif self.key in ("fantasy", "drama"):
                return "af_heart"
            return "am_michael"
        elif lang in ("ja", "japanese"):
            return "jf_nezumi" if self.key == "anime" else "jm_kumo"
        return self.default_voice


GENRE_AUDIO_POOLS: dict[str, dict[str, list[str]]] = {
    "ancient_history": {
        "ambience_setup": ["amb_ancient_sanctum", "amb_temple", "amb_vedic_resonance", "amb_quarry_echo"],
        "ambience_rising": ["amb_ancient_sanctum", "amb_drone"],
        "ambience_climax": ["amb_vedic_resonance", "amb_drone"],
        "ambience_resolution": ["amb_vedic_resonance", "amb_ancient_sanctum", "amb_temple", "amb_warm"],
        "sfx_setup": ["sfx_temple_gong", "sfx_chisel_tap", "sfx_bell", "sfx_chisel_echo"],
        "sfx_rising": ["sfx_ancient_mechanism", "sfx_riser"],
        "sfx_climax": ["sfx_stone_impact", "sfx_cinematic_hit", "sfx_mystic_reveal", "sfx_boom", "sfx_impact"],
        "sfx_resolution": ["sfx_temple_gong", "sfx_bell", "sfx_mystic_reveal"],
    },
    "anime": {
        "ambience_setup": ["amb_cyber", "amb_wind", "amb_warm"],
        "ambience_rising": ["amb_tension", "amb_drone", "amb_cyber"],
        "ambience_climax": ["amb_tension", "amb_heartbeat", "amb_cyber"],
        "ambience_resolution": ["amb_uplift", "amb_warm", "amb_wind"],
        "sfx_setup": ["sfx_whoosh", "sfx_dark_whoosh", "sfx_glitch", "sfx_sparkle"],
        "sfx_rising": ["sfx_whoosh", "sfx_riser", "sfx_glitch", "sfx_swell"],
        "sfx_climax": ["sfx_cinematic_hit", "sfx_boom", "sfx_impact", "sfx_stinger"],
        "sfx_resolution": ["sfx_swell", "sfx_bell", "sfx_chime", "sfx_sparkle"],
    },
    "comedy": {
        "ambience_setup": ["amb_uplift", "amb_warm", "amb_hum"],
        "ambience_rising": ["amb_hum", "amb_uplift"],
        "ambience_climax": ["amb_uplift", "amb_warm"],
        "ambience_resolution": ["amb_uplift", "amb_warm"],
        "sfx_setup": ["sfx_pop", "sfx_whoosh", "sfx_chime"],
        "sfx_rising": ["sfx_pop", "sfx_whoosh", "sfx_sparkle"],
        "sfx_climax": ["sfx_pop", "sfx_cinematic_hit", "sfx_sparkle"],
        "sfx_resolution": ["sfx_pop", "sfx_chime", "sfx_sparkle"],
    },
    "documentary": {
        "ambience_setup": ["amb_wind", "amb_hum", "amb_warm"],
        "ambience_rising": ["amb_tension", "amb_drone", "amb_wind"],
        "ambience_climax": ["amb_tension", "amb_drone"],
        "ambience_resolution": ["amb_warm", "amb_wind", "amb_hum"],
        "sfx_setup": ["sfx_clock_tick", "sfx_whoosh", "sfx_bell"],
        "sfx_rising": ["sfx_riser", "sfx_clock_tick", "sfx_whoosh"],
        "sfx_climax": ["sfx_cinematic_hit", "sfx_boom", "sfx_impact"],
        "sfx_resolution": ["sfx_bell", "sfx_chime", "sfx_swell"],
    },
    "drama": {
        "ambience_setup": ["amb_warm", "amb_rain", "amb_wind"],
        "ambience_rising": ["amb_tension", "amb_rain", "amb_drone"],
        "ambience_climax": ["amb_tension", "amb_drone"],
        "ambience_resolution": ["amb_warm", "amb_wind", "amb_rain"],
        "sfx_setup": ["sfx_bell", "sfx_clock_tick", "sfx_whisper"],
        "sfx_rising": ["sfx_riser", "sfx_whisper", "sfx_clock_tick"],
        "sfx_climax": ["sfx_impact", "sfx_swell", "sfx_cinematic_hit"],
        "sfx_resolution": ["sfx_swell", "sfx_bell", "sfx_whisper"],
    },
    "explainer": {
        "ambience_setup": ["amb_hum", "amb_warm", "amb_cyber"],
        "ambience_rising": ["amb_drone", "amb_tension", "amb_cyber"],
        "ambience_climax": ["amb_tension", "amb_uplift", "amb_cyber"],
        "ambience_resolution": ["amb_uplift", "amb_warm", "amb_hum"],
        "sfx_setup": ["sfx_whoosh", "sfx_glitch", "sfx_clock_tick", "sfx_pop"],
        "sfx_rising": ["sfx_whoosh", "sfx_riser", "sfx_chime", "sfx_pop"],
        "sfx_climax": ["sfx_cinematic_hit", "sfx_swell", "sfx_impact"],
        "sfx_resolution": ["sfx_chime", "sfx_bell", "sfx_sparkle", "sfx_pop"],
    },
    "fantasy": {
        "ambience_setup": ["amb_ethereal", "amb_warm", "amb_wind"],
        "ambience_rising": ["amb_ethereal", "amb_temple", "amb_tension"],
        "ambience_climax": ["amb_ethereal", "amb_tension"],
        "ambience_resolution": ["amb_warm", "amb_wind", "amb_ethereal"],
        "sfx_setup": ["sfx_sparkle", "sfx_chime", "sfx_bell", "sfx_swell"],
        "sfx_rising": ["sfx_sparkle", "sfx_whoosh", "sfx_riser", "sfx_chime"],
        "sfx_climax": ["sfx_mystic_reveal", "sfx_cinematic_hit", "sfx_boom", "sfx_swell"],
        "sfx_resolution": ["sfx_chime", "sfx_bell", "sfx_sparkle", "sfx_swell"],
    },
    "horror": {
        "ambience_setup": ["amb_wind", "amb_drone", "amb_whispers", "amb_rain"],
        "ambience_rising": ["amb_drone", "amb_rain", "amb_tension", "amb_whispers", "amb_ghost_voices"],
        "ambience_climax": ["amb_heartbeat", "amb_drone", "amb_ghost_voices", "amb_tension"],
        "ambience_resolution": ["amb_wind", "amb_drone", "amb_rain", "amb_whispers"],
        "sfx_setup": [
            "sfx_dark_whoosh",
            "sfx_whisper",
            "sfx_creak",
            "sfx_eerie_drone_swell",
            "sfx_creepy_music_box",
            "sfx_haunted_breathing",
            "sfx_footstep_wood",
        ],
        "sfx_rising": [
            "sfx_scare_string",
            "sfx_riser",
            "sfx_demonic_whisper",
            "sfx_ghost_crying",
            "sfx_ghostly_groan",
            "sfx_thunder_rumble",
            "sfx_heartbeat_single",
        ],
        "sfx_climax": [
            "sfx_jumpscare",
            "sfx_phantom_shriek",
            "sfx_cinematic_hit",
            "sfx_monster_growl",
            "sfx_boom",
            "sfx_stinger",
            "sfx_impact",
        ],
        "sfx_resolution": [
            "sfx_bell",
            "sfx_swell",
            "sfx_whisper",
            "sfx_creepy_music_box",
            "sfx_eerie_drone_swell",
        ],
    },
    "motivational": {
        "ambience_setup": ["amb_warm", "amb_wind"],
        "ambience_rising": ["amb_tension", "amb_drone", "amb_warm"],
        "ambience_climax": ["amb_uplift", "amb_warm"],
        "ambience_resolution": ["amb_uplift", "amb_warm", "amb_wind"],
        "sfx_setup": ["sfx_whoosh", "sfx_clock_tick", "sfx_swell"],
        "sfx_rising": ["sfx_riser", "sfx_whoosh", "sfx_clock_tick"],
        "sfx_climax": ["sfx_cinematic_hit", "sfx_swell", "sfx_boom"],
        "sfx_resolution": ["sfx_chime", "sfx_bell", "sfx_swell"],
    },
    "mythology": {
        "ambience_setup": ["amb_temple", "amb_ethereal", "amb_warm"],
        "ambience_rising": ["amb_temple", "amb_drone", "amb_tension"],
        "ambience_climax": ["amb_temple", "amb_drone", "amb_tension"],
        "ambience_resolution": ["amb_temple", "amb_warm", "amb_wind", "amb_ethereal"],
        "sfx_setup": ["sfx_bell", "sfx_horn", "sfx_conch_call", "sfx_whoosh"],
        "sfx_rising": ["sfx_riser", "sfx_swell", "sfx_horn", "sfx_conch_call"],
        "sfx_climax": ["sfx_mystic_reveal", "sfx_cinematic_hit", "sfx_boom", "sfx_impact"],
        "sfx_resolution": ["sfx_bell", "sfx_chime", "sfx_swell", "sfx_conch_call"],
    },
    "scifi": {
        "ambience_setup": ["amb_cyber", "amb_hum", "amb_wind"],
        "ambience_rising": ["amb_cyber", "amb_tension", "amb_drone", "amb_radio_voices"],
        "ambience_climax": ["amb_cyber", "amb_tension", "amb_heartbeat"],
        "ambience_resolution": ["amb_hum", "amb_cyber", "amb_wind"],
        "sfx_setup": ["sfx_dark_whoosh", "sfx_whoosh", "sfx_glitch", "sfx_clock_tick"],
        "sfx_rising": ["sfx_glitch", "sfx_riser", "sfx_whoosh"],
        "sfx_climax": ["sfx_cinematic_hit", "sfx_boom", "sfx_glitch", "sfx_impact"],
        "sfx_resolution": ["sfx_chime", "sfx_bell", "sfx_glitch"],
    },
}


def resolve_dynamic_audio(
    genre_key: str,
    movement_key: str,
    beat: Beat,
    scene_index: int,
    scene_count: int,
    seed: int,
    estimated_seconds: float,
    tension: float,
) -> tuple[dict, list[dict]]:
    """Determine dynamic ambience bed and optimal, tasteful sound cues strictly from the genre audio pool."""
    pool = GENRE_AUDIO_POOLS.get(genre_key, GENRE_AUDIO_POOLS["horror"])
    rng = random.Random(seed + scene_index * 37)

    # 1. Subtle, non-intrusive Ambience volume (0.16 - 0.24)
    if movement_key in ("hook", "setup"):
        amb_options = pool.get("ambience_setup") or pool.get("ambience_resolution", ["amb_wind"])
        default_vol = 0.18
    elif movement_key in ("inciting", "rising"):
        amb_options = pool.get("ambience_rising") or pool.get("ambience_setup", ["amb_drone"])
        default_vol = 0.20
    elif movement_key == "climax":
        amb_options = pool.get("ambience_climax") or pool.get("ambience_rising", ["amb_tension"])
        default_vol = 0.24
    else:  # resolution, button
        amb_options = pool.get("ambience_resolution") or pool.get("ambience_setup", ["amb_warm"])
        default_vol = 0.18

    chosen_amb = rng.choice(amb_options) if amb_options else "amb_drone"
    amb_volume = round(min(0.25, max(0.14, default_vol * (0.85 + tension * 0.2))), 2)

    # 2. Optimal, tasteful sound cues (sparse & high-impact: only at key dramatic moments)
    sound_cues: list[dict] = []

    # Only place SFX at significant story beats, not after every single audio track:
    # - Opening hook (scene 0)
    # - Inciting incident / disturbance
    # - Climax / peak confrontation
    # - Final closing scene / button
    # - High tension peaks in rising action (every 3rd rising scene or tension >= 0.8)
    should_have_sfx = (
        scene_index == 0
        or movement_key in ("hook", "inciting", "climax", "button")
        or (movement_key == "resolution" and scene_index == scene_count - 1)
        or (movement_key == "rising" and (scene_index % 3 == 0 or tension >= 0.8))
    )

    if should_have_sfx:
        # Choose primary scene entry cue from genre pool
        if movement_key == "climax":
            primary_pool = pool.get("sfx_climax") or pool.get("sfx_rising", ["sfx_cinematic_hit"])
            primary_vol = 0.24
        elif movement_key in ("hook", "setup") or scene_index == 0:
            primary_pool = pool.get("sfx_setup") or pool.get("sfx_rising", ["sfx_whoosh"])
            primary_vol = 0.20
        elif movement_key in ("inciting", "rising"):
            primary_pool = pool.get("sfx_rising") or pool.get("sfx_setup", ["sfx_whoosh"])
            primary_vol = 0.22
        else:  # resolution, button
            primary_pool = pool.get("sfx_resolution") or pool.get("sfx_setup", ["sfx_bell"])
            primary_vol = 0.18

        cue1_sfx = rng.choice(primary_pool)
        cue1_at = 0.0 if scene_index == 0 else round(min(0.15, max(0.02, estimated_seconds * 0.02)), 2)
        cue1_vol = round(min(0.26, max(0.14, primary_vol * (0.8 + tension * 0.25))), 2)
        sound_cues.append({"at": cue1_at, "sfx": cue1_sfx, "volume": cue1_vol})

        # Optional 2nd accent cue ONLY on climax scenes with sufficient duration (>= 4.0s)
        if estimated_seconds >= 4.0 and movement_key == "climax":
            accent_pool = pool.get("sfx_climax") or pool.get("sfx_rising", ["sfx_boom"])
            accent_vol = 0.24
            accent_candidates = [s for s in accent_pool if s != cue1_sfx]
            cue2_sfx = rng.choice(accent_candidates) if accent_candidates else rng.choice(accent_pool)
            cue2_at = round(min(estimated_seconds - 0.5, max(1.0, estimated_seconds * 0.55)), 2)
            cue2_vol = round(min(0.25, max(0.14, accent_vol * (0.8 + tension * 0.25))), 2)
            sound_cues.append({"at": cue2_at, "sfx": cue2_sfx, "volume": cue2_vol})

    sound_cues.sort(key=lambda c: c["at"])
    return {"src": chosen_amb, "volume": amb_volume}, sound_cues


def _make_anime_genre() -> GenreProfile:
    beats = {
        "establish": Beat(
            name="establish",
            role="Introduce the warriors, battlefield grounds, and the impending legendary clash.",
            visuals=[
                {"type": "forest", "fog": 0.3, "moonlight": 0.85},
                {"type": "corridor", "flicker": 0.4, "depth": 0.7},
                {"type": "figure", "distance": 0.65, "fog": 0.3, "glow": 0.6},
            ],
            ambience=("amb_tension", 0.4),
            effects={"vignette": 0.65, "grain": 0.2, "dust": 0.25, "zoom": 0.03, "drift": "in"},
            cues=[(0.2, "sfx_whoosh", 0.5)],
            tension=0.2,
            emotion="calm",
            gesture="none",
            acting_intensity=0.5,
            character=True,
        ),
        "dread": Beat(
            name="dread",
            role="Power auras ignite as the rivals take their fighting stances.",
            visuals=[
                {"type": "candle", "glow": 1.0},
                {"type": "figure", "distance": 0.45, "fog": 0.25, "glow": 0.85},
                {"type": "eyes", "pairs": 2, "spread": 0.5},
            ],
            ambience=("amb_tension", 0.46),
            effects={"vignette": 0.7, "grain": 0.22, "dust": 0.3, "zoom": 0.04, "drift": "in"},
            cues=[(0.35, "sfx_riser", 0.55)],
            tension=0.5,
            emotion="worried",
            gesture="reach",
            acting_intensity=0.75,
            character=True,
        ),
        "approach": Beat(
            name="approach",
            role="High-speed combat flurry, shockwaves, and rising power escalation.",
            visuals=[
                {"type": "figure", "distance": 0.35, "fog": 0.2, "glow": 0.95},
                {"type": "rain", "intensity": 0.6, "lightning": True},
                {"type": "corridor", "flicker": 0.6, "depth": 0.8},
            ],
            ambience=("amb_drone", 0.48),
            effects={"vignette": 0.75, "grain": 0.25, "dust": 0.35, "zoom": 0.05, "shake": 0.2, "drift": "in"},
            cues=[(0.15, "sfx_whoosh", 0.6), (0.6, "sfx_cinematic_hit", 0.7)],
            tension=0.75,
            emotion="shock",
            gesture="reach",
            acting_intensity=0.88,
            character=True,
        ),
        "reveal": Beat(
            name="reveal",
            role="Ultimate signature move unleashed with catastrophic explosive impact.",
            visuals=[
                {"type": "figure", "distance": 0.2, "fog": 0.15, "glow": 1.0},
                {"type": "candle", "glow": 1.0},
                {"type": "eyes", "pairs": 3, "spread": 0.7},
            ],
            ambience=("amb_tension", 0.55),
            effects={"vignette": 0.82, "grain": 0.28, "dust": 0.4, "shake": 0.45, "zoom": 0.06, "drift": "in"},
            cues=[(0.05, "sfx_boom", 0.95), (0.55, "sfx_cinematic_hit", 0.85)],
            tension=1.0,
            emotion="shock",
            gesture="recoil",
            acting_intensity=1.0,
            character=True,
        ),
        "react": Beat(
            name="react",
            role="Dust clears over the crater, revealing the shattered ground and aftermath.",
            visuals=[
                {"type": "fog", "density": 0.5, "glow": 0.6},
                {"type": "forest", "fog": 0.4, "moonlight": 0.8},
            ],
            ambience=("amb_wind", 0.35),
            effects={"vignette": 0.68, "grain": 0.2, "dust": 0.25, "zoom": 0.02, "drift": "out"},
            cues=[(0.4, "sfx_swell", 0.5)],
            tension=0.3,
            emotion="calm",
            gesture="clutch",
            acting_intensity=0.5,
            character=True,
        ),
        "resolve": Beat(
            name="resolve",
            role="Mutual respect, triumphant hero stance, and looking toward future battles.",
            visuals=[
                {"type": "forest", "fog": 0.25, "moonlight": 0.9},
                {"type": "house", "fog": 0.2, "lit": True},
            ],
            ambience=("amb_uplift", 0.32),
            effects={"vignette": 0.62, "grain": 0.18, "dust": 0.2, "zoom": 0.02, "drift": "out"},
            cues=[(0.7, "sfx_bell", 0.65)],
            tension=0.05,
            emotion="calm",
            gesture="none",
            acting_intensity=0.35,
            character=True,
        ),
    }
    movements = {
        "hook": Movement("hook", "Open with a legendary clash of titans, intense energy aura flare, or battle challenge.", "establish", 0.05),
        "setup": Movement("setup", "Establish the legendary warriors, their motivations, fighting styles, and battle stakes.", "establish", 0.13),
        "inciting": Movement("inciting", "The confrontation begins: stances taken, energy charging, first blinding exchange.", "dread", 0.12),
        "rising": Movement("rising", "Escalating combat flurry: rapid counterattacks, power transformations, and terrain-shattering strikes.", "approach", 0.32),
        "climax": Movement("climax", "The ultimate finishing technique: catastrophic explosion of energy, decisive impact, and shockwaves.", "reveal", 0.16),
        "resolution": Movement("resolution", "The battlefield settles: smoke clears, showing the aftermath, sheer power, and mutual warrior respect.", "react", 0.14),
        "button": Movement("button", "A legendary closing declaration and triumphant gaze toward the next grand battle horizon.", "resolve", 0.08),
    }
    return GenreProfile(
        key="anime",
        name="Anime & Action",
        aliases=[
            "anime", "action", "fight", "battle", "shonen", "manga", "warrior", "superhero",
            "martial_arts", "dragonball", "naruto", "goku", "dbz", "shinobi", "ninja", "samurai",
            "epic_battle", "showdown", "duel"
        ],
        tone="energetic, explosive, epic, high-stakes, intense, and heroic",
        writer_instructions=(
            "You are writing narration for an epic, high-octane anime and action battle video. "
            "Deliver fast-paced energy, explosive combat choreography, rising power auras, legendary rivalries, "
            "signature special techniques, and triumphant climaxes with cinematic anime flair."
        ),
        system_prompt="You write narration for intense, epic, and cinematic anime/action stories. You deliver high-stakes battles, power awakenings, and legendary showdowns.",
        visual_style_prefix="Masterpiece dynamic anime illustration, intense high-budget anime action key visual, Studio Ufotable & Mappa style, sharp glowing energy auras, cinematic composition: ",
        visual_style_suffix=", dynamic camera angle, vibrant celestial lighting, sharp cel-shaded outlines, 4k anime wallpaper.",
        search_keywords_map={
            "forest": "dense bamboo anime battle forest misty cherry blossoms swirling energy",
            "house": "traditional japanese dojo martial arts training hall destroyed roof",
            "grave": "shattered battlefield crater smoke lightning residual ki energy aura",
            "corridor": "futuristic cyber anime corridor glowing neon energy trails",
            "candle": "raging blue fire flame aura spiritual energy burst",
            "figure": "fierce anime warrior charging energy stance heroic silhouette glowing aura",
            "eyes": "intense glowing anime eyes closeup glowing sparks determination",
            "rain": "torrential anime rainstorm lightning striking shattered rocky ground",
            "fog": "thick battlefield smoke dust explosion swirling vortex aura",
            "void": "cosmic dimensional rift energy explosion supernova power clash",
        },
        default_keywords="epic anime battle energy aura warriors clash explosion landscape",
        theme={
            "accent": "#f59e0b",
            "text": "#ffffff",
            "muted": "#fdba74",
            "panel": "rgba(20, 10, 35, 0.75)",
            "headlineFont": "'Impact', 'Montserrat', system-ui, sans-serif",
            "bodyFont": "'Segoe UI', system-ui, -apple-system, sans-serif",
        },
        movements=movements,
        beats=beats,
        rising_beats=["approach", "dread"],
        default_wpm=125,
        default_speed=0.96,
        bg_color="#0c0714",
    )


def _make_fantasy_genre() -> GenreProfile:
    beats = {
        "establish": Beat(
            name="establish",
            role="Reveal the wondrous realm and magical tranquility.",
            visuals=[
                {"type": "forest", "fog": 0.35, "moonlight": 0.8},
                {"type": "house", "fog": 0.3, "lit": True, "moonlight": 0.7},
                {"type": "fog", "density": 0.45, "glow": 0.75},
            ],
            ambience=("amb_wind", 0.35),
            effects={"vignette": 0.7, "grain": 0.25, "dust": 0.35, "zoom": 0.03, "drift": "in"},
            cues=[(0.15, "sfx_bell", 0.45)],
            tension=0.0,
            emotion="calm",
            gesture="none",
            acting_intensity=0.4,
            character=False,
        ),
        "dread": Beat(
            name="dread",
            role="The magical discovery or enchanted call.",
            visuals=[
                {"type": "candle", "glow": 0.95},
                {"type": "forest", "fog": 0.45, "moonlight": 0.85},
                {"type": "figure", "distance": 0.55, "fog": 0.4, "glow": 0.7},
            ],
            ambience=("amb_wind", 0.38),
            effects={"vignette": 0.75, "grain": 0.3, "dust": 0.4, "zoom": 0.03, "drift": "in"},
            cues=[(0.4, "sfx_bell", 0.55)],
            tension=0.2,
            emotion="worried",
            gesture="none",
            acting_intensity=0.55,
            character=True,
        ),
        "approach": Beat(
            name="approach",
            role="Venturing deeper into the magical adventure.",
            visuals=[
                {"type": "figure", "distance": 0.45, "fog": 0.35, "glow": 0.75},
                {"type": "forest", "fog": 0.5, "moonlight": 0.8},
                {"type": "corridor", "flicker": 0.3, "depth": 0.6},
            ],
            ambience=("amb_drone", 0.38),
            effects={"vignette": 0.75, "grain": 0.3, "dust": 0.45, "zoom": 0.04, "drift": "in"},
            cues=[(0.35, "sfx_riser", 0.4)],
            tension=0.45,
            emotion="calm",
            gesture="reach",
            acting_intensity=0.65,
            character=True,
        ),
        "reveal": Beat(
            name="reveal",
            role="The peak magical wonder, enchantment, or triumph.",
            visuals=[
                {"type": "candle", "glow": 1.0},
                {"type": "figure", "distance": 0.2, "fog": 0.3, "glow": 0.95},
                {"type": "eyes", "pairs": 3, "spread": 0.6},
            ],
            ambience=("amb_wind", 0.48),
            effects={"vignette": 0.8, "grain": 0.35, "dust": 0.5, "shake": 0.2, "zoom": 0.05, "drift": "in"},
            cues=[(0.05, "sfx_impact", 0.75), (0.6, "sfx_bell", 0.85)],
            tension=0.85,
            emotion="shock",
            gesture="recoil",
            acting_intensity=0.9,
            character=True,
        ),
        "react": Beat(
            name="react",
            role="Joyful wonder and relief in the aftermath.",
            visuals=[
                {"type": "candle", "glow": 0.85},
                {"type": "fog", "density": 0.4, "glow": 0.7},
                {"type": "forest", "fog": 0.3, "moonlight": 0.85},
            ],
            ambience=("amb_wind", 0.35),
            effects={"vignette": 0.72, "grain": 0.28, "dust": 0.35, "zoom": 0.02, "drift": "out"},
            cues=[(0.4, "sfx_bell", 0.5)],
            tension=0.2,
            emotion="calm",
            gesture="clutch",
            acting_intensity=0.5,
            character=True,
        ),
        "resolve": Beat(
            name="resolve",
            role="Magical harmony, wishes fulfilled, and peace.",
            visuals=[
                {"type": "forest", "fog": 0.25, "moonlight": 0.9},
                {"type": "house", "fog": 0.2, "lit": True, "moonlight": 0.8},
                {"type": "fog", "density": 0.3, "glow": 0.8},
            ],
            ambience=("amb_wind", 0.3),
            effects={"vignette": 0.7, "grain": 0.25, "dust": 0.3, "zoom": 0.025, "drift": "out"},
            cues=[(0.7, "sfx_bell", 0.65)],
            tension=0.1,
            emotion="calm",
            gesture="none",
            acting_intensity=0.35,
            character=True,
        ),
    }
    movements = {
        "hook": Movement("hook", "Open with a captivating glimpse of wonder, magic, or a legendary fairy tale secret.", "establish", 0.05),
        "setup": Movement("setup", "Introduce the wondrous realm, the pure-hearted protagonist, and their innocent wish.", "establish", 0.13),
        "inciting": Movement("inciting", "The magical call: a fairy guide arrives, an enchanted artifact appears, or a magical realm opens.", "dread", 0.12),
        "rising": Movement("rising", "Enchanting trials and magical adventures. Courage grows with every wondrous step.", "approach", 0.32),
        "climax": Movement("climax", "The grand magical test or triumph of kindness and bravery against dark sorcery.", "reveal", 0.16),
        "resolution": Movement("resolution", "Wishes fulfilled, harmony restored across the enchanted kingdom, and blessings for all.", "react", 0.14),
        "button": Movement("button", "A heartwarming parting thought on innocence, wonder, and the magic inside pure hearts.", "resolve", 0.08),
    }
    return GenreProfile(
        key="fantasy",
        name="Fantasy & Fairy Tale",
        aliases=["fantasy", "fairytale", "fairy_tale", "magic", "sonpari", "son_pari", "jadui", "pari", "folklore", "mythical", "fable"],
        tone="enchanting, magical, wondrous, heartwarming, and full of courage",
        writer_instructions=(
            "You are writing narration for an enchanting fantasy and fairy tale story video. "
            "Blend wonder, magical adventures, courage, vivid imagination, and heartwarming lessons. "
            "Describe magical realms, mystical artifacts, celestial beings, and heroic pure hearts."
        ),
        system_prompt="You write narration for magical fantasy, fairy tale, and wondrous adventure videos. You craft vivid, uplifting, engaging stories with warmth and wonder.",
        visual_style_prefix="Enchanting 2D digital fairy tale concept art, vibrant magical colors, whimsical lighting, glowing particles, celestial wonder: ",
        visual_style_suffix=", storybook illustration style, highly detailed fantasy background, cinematic composition, breathtaking color palette.",
        search_keywords_map={
            "forest": "enchanted magical glowing forest fairy woods",
            "house": "magical fairy tale cottage fantasy castle",
            "grave": "ancient mystical ruins glowing sanctuary",
            "corridor": "glowing crystal palace hall fantasy corridor",
            "candle": "magical glowing lantern fairy lights",
            "figure": "celestial fairy guardian magical spirit glowing wings",
            "eyes": "twinkling starlight magical sparkles",
            "rain": "magical shimmering starlight rain celestial sky",
            "fog": "glowing rainbow mist enchanted valley",
            "void": "celestial starry cosmos magical galaxy",
        },
        default_keywords="enchanted fairy tale magical realm glowing fantasy landscape",
        theme={
            "accent": "#f59e0b",
            "text": "#fffdf5",
            "muted": "#d8b4fe",
            "panel": "rgba(30, 20, 50, 0.68)",
            "headlineFont": "Georgia, 'Times New Roman', serif",
            "bodyFont": "'Segoe UI', system-ui, -apple-system, sans-serif",
        },
        movements=movements,
        beats=beats,
        rising_beats=["approach", "dread"],
        bg_color="#0d0818",
    )


def _make_scifi_genre() -> GenreProfile:
    beats = {
        "establish": Beat(
            name="establish",
            role="Establish futuristic setting, technology, and status quo.",
            visuals=[
                {"type": "corridor", "flicker": 0.4, "depth": 0.7},
                {"type": "fog", "density": 0.5, "glow": 0.6},
                {"type": "forest", "fog": 0.6, "moonlight": 0.7},
            ],
            ambience=("amb_hum", 0.45),
            effects={"vignette": 0.8, "grain": 0.35, "dust": 0.2, "zoom": 0.03, "drift": "in"},
            cues=[(0.2, "sfx_riser", 0.35)],
            tension=0.0,
            emotion="calm",
            gesture="none",
            acting_intensity=0.35,
            character=False,
        ),
        "dread": Beat(
            name="dread",
            role="System anomaly, AI awakening, or deep space disturbance.",
            visuals=[
                {"type": "corridor", "flicker": 0.7, "depth": 0.6},
                {"type": "void", "creep": 0.6},
                {"type": "fog", "density": 0.7, "glow": 0.5},
            ],
            ambience=("amb_hum", 0.5),
            effects={"vignette": 0.85, "grain": 0.4, "dust": 0.22, "zoom": 0.03, "drift": "in", "exposureFlicker": 0.15},
            cues=[(0.4, "sfx_riser", 0.45)],
            tension=0.3,
            emotion="worried",
            gesture="none",
            acting_intensity=0.6,
            character=True,
        ),
        "approach": Beat(
            name="approach",
            role="Racing against the clock into high-tech peril.",
            visuals=[
                {"type": "corridor", "flicker": 0.85, "depth": 0.8},
                {"type": "figure", "distance": 0.4, "fog": 0.4, "glow": 0.6},
            ],
            ambience=("amb_drone", 0.5),
            effects={"vignette": 0.88, "grain": 0.45, "dust": 0.25, "zoom": 0.05, "drift": "in"},
            cues=[(0.35, "sfx_riser", 0.5)],
            tension=0.55,
            emotion="worried",
            gesture="reach",
            acting_intensity=0.75,
            character=True,
        ),
        "reveal": Beat(
            name="reveal",
            role="The technological singularity, revelation, or cosmic encounter.",
            visuals=[
                {"type": "eyes", "pairs": 4, "spread": 0.7},
                {"type": "figure", "distance": 0.2, "fog": 0.3, "glow": 0.8},
                {"type": "candle", "glow": 1.0},
            ],
            ambience=("amb_drone", 0.55),
            effects={"vignette": 0.9, "grain": 0.5, "dust": 0.2, "shake": 0.3, "zoom": 0.05, "drift": "in"},
            cues=[(0.05, "sfx_stinger", 0.85), (0.6, "sfx_impact", 0.7)],
            tension=0.95,
            emotion="shock",
            gesture="recoil",
            acting_intensity=0.95,
            character=True,
        ),
        "react": Beat(
            name="react",
            role="The shift in human destiny and cybernetic aftermath.",
            visuals=[
                {"type": "void", "creep": 0.7},
                {"type": "corridor", "flicker": 0.4, "depth": 0.5},
            ],
            ambience=("amb_hum", 0.42),
            effects={"vignette": 0.85, "grain": 0.4, "dust": 0.2, "zoom": 0.02, "drift": "out"},
            cues=[(0.4, "sfx_impact", 0.4)],
            tension=0.25,
            emotion="calm",
            gesture="clutch",
            acting_intensity=0.5,
            character=True,
        ),
        "resolve": Beat(
            name="resolve",
            role="Visionary outlook on humanity's technological horizon.",
            visuals=[
                {"type": "fog", "density": 0.5, "glow": 0.7},
                {"type": "corridor", "flicker": 0.2, "depth": 0.6},
            ],
            ambience=("amb_wind", 0.35),
            effects={"vignette": 0.8, "grain": 0.35, "dust": 0.2, "zoom": 0.025, "drift": "out"},
            cues=[(0.7, "sfx_bell", 0.55)],
            tension=0.1,
            emotion="calm",
            gesture="none",
            acting_intensity=0.35,
            character=True,
        ),
    }
    movements = {
        "hook": Movement("hook", "Open with a provocative question or chilling prophecy about future tech, space, or AI.", "establish", 0.05),
        "setup": Movement("setup", "Establish the futuristic world, advanced technologies, and the status quo before the anomaly.", "establish", 0.13),
        "inciting": Movement("inciting", "The technological disturbance: an unexpected AI behavior, deep space signal, or quantum anomaly.", "dread", 0.12),
        "rising": Movement("rising", "Escalating technological stakes, cascading systems, and relentless investigation into the unknown.", "approach", 0.32),
        "climax": Movement("climax", "The high-stakes breakthrough or decisive confrontation between human consciousness and machine reality.", "reveal", 0.16),
        "resolution": Movement("resolution", "The aftermath for humanity, showing how the world or protagonist is transformed forever.", "react", 0.14),
        "button": Movement("button", "A visionary final thought on humanity's technological future and cosmic place.", "resolve", 0.08),
    }
    return GenreProfile(
        key="scifi",
        name="Sci-Fi & Cyberpunk",
        aliases=["scifi", "sci-fi", "cyberpunk", "futuristic", "space", "tech", "ai", "robot", "cyber", "galaxy", "future"],
        tone="futuristic, high-stakes, analytical, awe-inspiring, and visionary",
        writer_instructions=(
            "You are writing narration for a thrilling sci-fi and cyberpunk story video. "
            "Explore cutting-edge technology, artificial intelligence, deep space exploration, and futuristic human destiny with high-stakes tension."
        ),
        system_prompt="You write narration for sci-fi, cyberpunk, futuristic tech, and space exploration videos. You deliver thought-provoking, sleek, high-tech narratives.",
        visual_style_prefix="Futuristic cyberpunk and sci-fi digital concept art, glowing neon blue and cyan accents, sleek holographic interfaces, hyper-technological aesthetic: ",
        visual_style_suffix=", cinematic sci-fi background, Unreal Engine 5 render style, dramatic volumetric lighting.",
        search_keywords_map={
            "forest": "bioluminescent alien cyber forest futuristic planet",
            "house": "cyberpunk futuristic megacity neon skyscraper lab",
            "grave": "abandoned space station derelict starship ruins",
            "corridor": "high-tech starship corridor neon cyber hallway",
            "candle": "glowing holographic data terminal quantum core",
            "figure": "futuristic cyborg android cybernetic human silhouette",
            "eyes": "neon cyber optic sensors robotic vision",
            "rain": "neon drenched cyberpunk city street rain night",
            "fog": "steam rising from futuristic cyber metropolis vents",
            "void": "deep space nebula cosmos starfield galaxy",
        },
        default_keywords="futuristic cyberpunk neon city space station sci-fi landscape",
        theme={
            "accent": "#00e5ff",
            "text": "#f0f6fc",
            "muted": "#8b949e",
            "panel": "rgba(10, 16, 28, 0.75)",
            "headlineFont": "'Segoe UI', system-ui, sans-serif",
            "bodyFont": "'Segoe UI', system-ui, -apple-system, sans-serif",
        },
        movements=movements,
        beats=beats,
        rising_beats=["approach", "dread"],
        bg_color="#050a14",
    )


def _make_documentary_genre() -> GenreProfile:
    beats = {
        "establish": Beat(
            name="establish",
            role="Set the historical/scientific foundation with clarity.",
            visuals=[
                {"type": "forest", "fog": 0.4, "moonlight": 0.6},
                {"type": "house", "fog": 0.3, "lit": True, "moonlight": 0.5},
            ],
            ambience=("amb_wind", 0.35),
            effects={"vignette": 0.7, "grain": 0.25, "dust": 0.2, "zoom": 0.02, "drift": "in"},
            cues=[(0.2, "sfx_bell", 0.4)],
            tension=0.0,
            emotion="calm",
            gesture="none",
            acting_intensity=0.3,
            character=False,
        ),
        "dread": Beat(
            name="dread",
            role="The intriguing puzzle, question, or historical turning point.",
            visuals=[
                {"type": "candle", "glow": 0.85},
                {"type": "corridor", "flicker": 0.3, "depth": 0.5},
            ],
            ambience=("amb_drone", 0.35),
            effects={"vignette": 0.75, "grain": 0.3, "dust": 0.25, "zoom": 0.025, "drift": "in"},
            cues=[(0.4, "sfx_bell", 0.45)],
            tension=0.2,
            emotion="calm",
            gesture="none",
            acting_intensity=0.45,
            character=True,
        ),
        "approach": Beat(
            name="approach",
            role="Deep investigation, evidence analysis, and historical journey.",
            visuals=[
                {"type": "figure", "distance": 0.5, "fog": 0.4, "glow": 0.5},
                {"type": "corridor", "flicker": 0.4, "depth": 0.6},
            ],
            ambience=("amb_drone", 0.4),
            effects={"vignette": 0.75, "grain": 0.35, "dust": 0.25, "zoom": 0.035, "drift": "in"},
            cues=[(0.35, "sfx_riser", 0.35)],
            tension=0.4,
            emotion="calm",
            gesture="explain",
            acting_intensity=0.6,
            character=True,
        ),
        "reveal": Beat(
            name="reveal",
            role="The monumental discovery or conclusive revelation.",
            visuals=[
                {"type": "candle", "glow": 1.0},
                {"type": "figure", "distance": 0.3, "fog": 0.3, "glow": 0.7},
            ],
            ambience=("amb_wind", 0.45),
            effects={"vignette": 0.8, "grain": 0.35, "dust": 0.25, "shake": 0.15, "zoom": 0.04, "drift": "in"},
            cues=[(0.05, "sfx_impact", 0.7), (0.6, "sfx_bell", 0.75)],
            tension=0.7,
            emotion="calm",
            gesture="explain",
            acting_intensity=0.8,
            character=True,
        ),
        "react": Beat(
            name="react",
            role="Examining the profound impact and historical legacy.",
            visuals=[
                {"type": "fog", "density": 0.4, "glow": 0.6},
                {"type": "house", "fog": 0.3, "lit": True},
            ],
            ambience=("amb_wind", 0.35),
            effects={"vignette": 0.7, "grain": 0.25, "dust": 0.2, "zoom": 0.02, "drift": "out"},
            cues=[(0.4, "sfx_bell", 0.45)],
            tension=0.15,
            emotion="calm",
            gesture="none",
            acting_intensity=0.4,
            character=True,
        ),
        "resolve": Beat(
            name="resolve",
            role="The enduring truth and modern significance.",
            visuals=[
                {"type": "forest", "fog": 0.3, "moonlight": 0.7},
                {"type": "house", "fog": 0.2, "lit": True},
            ],
            ambience=("amb_wind", 0.3),
            effects={"vignette": 0.68, "grain": 0.22, "dust": 0.18, "zoom": 0.02, "drift": "out"},
            cues=[(0.7, "sfx_bell", 0.6)],
            tension=0.05,
            emotion="calm",
            gesture="none",
            acting_intensity=0.3,
            character=True,
        ),
    }
    movements = {
        "hook": Movement("hook", "Open with a mind-bending fact, historical puzzle, or compelling question that demands an explanation.", "establish", 0.05),
        "setup": Movement("setup", "Lay the foundation: the historical context, geographic setting, or scientific baseline.", "establish", 0.13),
        "inciting": Movement("inciting", "The discovery or turning point that challenged what people believed or started an unprecedented chain of events.", "dread", 0.12),
        "rising": Movement("rising", "Unfolding evidence, key milestones, competing theories, and dramatic breakthroughs.", "approach", 0.32),
        "climax": Movement("climax", "The ultimate revelation, pivotal event, or scientific explanation that ties everything together.", "reveal", 0.16),
        "resolution": Movement("resolution", "The lasting impact, historical significance, and real-world implications today.", "react", 0.14),
        "button": Movement("button", "A powerful concluding insight that reframes how the viewer sees history, science, or the world.", "resolve", 0.08),
    }
    return GenreProfile(
        key="documentary",
        name="Documentary & Educational",
        aliases=["documentary", "docu", "history", "historic", "science", "facts", "educational", "biography", "case_study", "nature", "curiosity"],
        tone="engaging, authoritative, captivating, informative, and insightful",
        writer_instructions=(
            "You are writing narration for an engaging documentary and educational video. "
            "Present fascinating facts, historical depth, mysteries, and compelling insights with clear, immersive storytelling."
        ),
        system_prompt="You write narration for educational, historical, scientific, and investigative documentary videos. You present rich facts and gripping insights.",
        visual_style_prefix="Cinematic documentary photography and digital painting, National Geographic aesthetic, elegant historical realism: ",
        visual_style_suffix=", wide cinematic shot, museum quality lighting, rich natural colors, highly detailed.",
        search_keywords_map={
            "forest": "pristine ancient wilderness nature landscape documentary",
            "house": "historical ancient monument archaeological landmark",
            "grave": "ancient tomb archaeological excavation history",
            "corridor": "historic library archive museum exhibition gallery",
            "candle": "vintage manuscript magnifying glass historical research",
            "figure": "historical explorer researcher scientist silhouette",
            "eyes": "telescope observatory deep sky celestial view",
            "rain": "dramatic weather phenomenon stormy clouds nature",
            "fog": "mysterious misty historical valley landscape",
            "void": "deep ocean trench cosmic universe astronomical discovery",
        },
        default_keywords="cinematic documentary landscape historical discovery scientific phenomenon",
        theme={
            "accent": "#38bdf8",
            "text": "#f8fafc",
            "muted": "#94a3b8",
            "panel": "rgba(15, 23, 42, 0.72)",
            "headlineFont": "Georgia, 'Times New Roman', serif",
            "bodyFont": "'Segoe UI', system-ui, -apple-system, sans-serif",
        },
        movements=movements,
        beats=beats,
        rising_beats=["approach", "dread"],
        bg_color="#080c14",
    )


def _make_ancient_history_genre() -> GenreProfile:
    beats = {
        "establish": Beat(
            name="establish",
            role="Set the ancient grandeur, monolithic scale, and historical enigma.",
            visuals=[
                {"type": "house", "fog": 0.25, "lit": True, "glow": 0.8},
                {"type": "corridor", "flicker": 0.2, "depth": 0.6},
            ],
            ambience=("amb_ancient_sanctum", 0.35),
            effects={"vignette": 0.72, "grain": 0.22, "dust": 0.3, "zoom": 0.025, "drift": "in"},
            cues=[(0.1, "sfx_temple_gong", 0.55), (0.6, "sfx_bronze_bell_ring", 0.45)],
            tension=0.0,
            emotion="calm",
            gesture="none",
            acting_intensity=0.35,
            character=False,
        ),
        "dread": Beat(
            name="dread",
            role="The architectural impossibility, lost tool, or mathematical paradox.",
            visuals=[
                {"type": "corridor", "flicker": 0.35, "depth": 0.7},
                {"type": "candle", "glow": 0.95},
            ],
            ambience=("amb_quarry_echo", 0.38),
            effects={"vignette": 0.78, "grain": 0.28, "dust": 0.32, "zoom": 0.03, "drift": "in"},
            cues=[(0.2, "sfx_monolith_shift", 0.55), (0.75, "sfx_chisel_echo", 0.45)],
            tension=0.25,
            emotion="calm",
            gesture="none",
            acting_intensity=0.5,
            character=True,
        ),
        "approach": Beat(
            name="approach",
            role="Deep architectural analysis, toolmarks, monolithic carving, and spatial mechanics.",
            visuals=[
                {"type": "figure", "distance": 0.45, "fog": 0.3, "glow": 0.6},
                {"type": "corridor", "flicker": 0.3, "depth": 0.8},
            ],
            ambience=("amb_subterranean_chamber", 0.40),
            effects={"vignette": 0.75, "grain": 0.25, "dust": 0.28, "zoom": 0.035, "drift": "in"},
            cues=[(0.25, "sfx_hydraulic_stone_gate", 0.5), (0.7, "sfx_ancient_mechanism", 0.45)],
            tension=0.45,
            emotion="calm",
            gesture="explain",
            acting_intensity=0.65,
            character=True,
        ),
        "reveal": Beat(
            name="reveal",
            role="The profound structural, astronomical, or metallurgical breakthrough of the monument.",
            visuals=[
                {"type": "house", "fog": 0.2, "lit": True, "glow": 1.0},
                {"type": "candle", "glow": 1.0},
            ],
            ambience=("amb_vedic_resonance", 0.42),
            effects={"vignette": 0.8, "grain": 0.3, "dust": 0.25, "shake": 0.12, "zoom": 0.04, "drift": "in"},
            cues=[(0.05, "sfx_stone_impact", 0.65), (0.55, "sfx_mystic_reveal", 0.7)],
            tension=0.75,
            emotion="calm",
            gesture="explain",
            acting_intensity=0.85,
            character=True,
        ),
        "react": Beat(
            name="react",
            role="Contemplating the astonishing intellect, precision, and legacy of ancient master builders.",
            visuals=[
                {"type": "fog", "density": 0.35, "glow": 0.7},
                {"type": "house", "fog": 0.25, "lit": True},
            ],
            ambience=("amb_ancient_sanctum", 0.35),
            effects={"vignette": 0.7, "grain": 0.22, "dust": 0.2, "zoom": 0.02, "drift": "out"},
            cues=[(0.3, "sfx_conch_call", 0.5), (0.8, "sfx_temple_gong", 0.45)],
            tension=0.2,
            emotion="calm",
            gesture="none",
            acting_intensity=0.45,
            character=True,
        ),
        "resolve": Beat(
            name="resolve",
            role="The timeless stone testament standing tall across millennia as an immortal wonder.",
            visuals=[
                {"type": "forest", "fog": 0.25, "moonlight": 0.8},
                {"type": "house", "fog": 0.2, "lit": True},
            ],
            ambience=("amb_vedic_resonance", 0.30),
            effects={"vignette": 0.68, "grain": 0.2, "dust": 0.18, "zoom": 0.02, "drift": "out"},
            cues=[(0.5, "sfx_bronze_bell_ring", 0.55), (0.9, "sfx_temple_gong", 0.5)],
            tension=0.05,
            emotion="calm",
            gesture="none",
            acting_intensity=0.35,
            character=True,
        ),
    }
    movements = {
        "hook": Movement("hook", "Open with a shocking physical impossibility, massive tonnage anomaly, or lost sound code that shatters modern construction logic.", "establish", 0.06),
        "setup": Movement("setup", "Establish the sheer monolithic scale, unbroken granite cliffside, and centuries of preserved majesty.", "establish", 0.12),
        "inciting": Movement("inciting", "The architectural paradox: missing quarry debris, zero-mortar interlocking joints, sub-millimeter precision, or underground acoustic resonance.", "dread", 0.12),
        "rising": Movement("rising", "Mechanical & Acoustic Breakdown: dissecting vertical top-down carving, 136.1 Hz Om resonance, astronomical alignments, and failed modern demolition attempts.", "approach", 0.32),
        "climax": Movement("climax", "The Master Revelation: how ancient architects encoded sacred geometry, acoustic physics, and eternal science into living stone.", "reveal", 0.18),
        "resolution": Movement("resolution", "The Indestructible Testament: standing unchanged across millennia as proof of a forgotten high-technology civilization.", "react", 0.12),
        "button": Movement("button", "An infinite-loop closing revelation that circles back directly to the opening hook question.", "resolve", 0.08),
    }
    return GenreProfile(
        key="ancient_history",
        name="Ancient Civilizations & Temple Engineering",
        aliases=[
            "ancient_history",
            "temple_engineering",
            "ancient_civilizations",
            "ancient_temples",
            "indian_history",
            "temple_architecture",
            "ancient_monuments",
            "ancient_india",
            "kailasa",
            "ancient_engineering",
            "archaeology",
            "vedic_architecture",
            "fictional_ancient",
            "regional_story",
            "regional_folklore",
            "ancient_lore",
            "cursed_village",
            "regional_accent_story",
            "temple_lore",
        ],
        tone="authoritative, mesmerizing, awe-inspiring, historically precise, and enigmatic",
        writer_instructions=(
            "You are writing narration for a viral, high-retention documentary video on ancient civilizations, megalithic engineering, and mysterious temple architecture. "
            "Deliver authoritative, evocative, and high-velocity storytelling in natural spoken Hindi (हिंदी लिपि). "
            "Master the 'Impossible Engineering' storytelling framework: "
            "1. Start MID-ACTION with a mind-boggling physical contradiction or mathematical impossibility (e.g. 400,000 tons of rock excavated top-down, zero mortar, acoustic resonance at 136.1 Hz). "
            "2. Reveal concrete archaeological enigmas (missing quarry waste, sub-millimeter precision cuts in granite, seismic interlocking joints, subterranean acoustic chambers). "
            "3. Disclose the failure of modern technology or brute force to replicate or destroy the monument. "
            "4. Conclude with a mind-bending cognitive paradox and an infinite replay loop connecting the closing sentence directly back into the opening hook. "
            "Punctuate with precision (use ',' for micro-breath, '।' for firm pauses, '...' for atmospheric suspense) to pace neural TTS delivery."
        ),
        system_prompt="You write authoritative, viral documentary scripts on ancient civilizations, megalithic architecture, and temple engineering in conversational Hindi. You turn mathematical anomalies, acoustic resonance, and impossible stone construction into high-retention viral stories.",
        visual_style_prefix="Cinematic national geographic documentary still, hyper-detailed archaeological photography, dramatic ancient architecture: ",
        visual_style_suffix=", monolithic stone carving, intricately detailed granite pillars, ancient Indian temple engineering, matte painting, hyper-detailed, ray tracing, cinematic atmosphere, 8k resolution, volumetric golden hour lighting.",
        search_keywords_map={
            "forest": "ancient sacred grove dense jungle overgrown ancient ruins",
            "house": "monolithic ancient temple carved granite architectural wonder",
            "grave": "ancient subterranean archaeological excavation crypt stone vault",
            "corridor": "ancient carved stone temple corridor granite pillars bas-reliefs",
            "candle": "sacred brass oil lamp flickering illumination ancient sanctum sanctorum",
            "figure": "ancient master architect visionary sculptor silhouette stone carving",
            "eyes": "ancient stone deity face intricate carving mystical gaze",
            "rain": "monsoon rain cascading down ancient temple stone spires dramatic",
            "fog": "mystical morning mist rising over ancient temple complex sacred valley",
            "void": "deep subterranean cavern beneath ancient stone temple secrets",
        },
        default_keywords="ancient indian temple monolithic architecture granite carving archaeological mystery",
        theme={
            "accent": "#d97706",
            "text": "#fef3c7",
            "muted": "#fbbf24",
            "panel": "rgba(24, 16, 7, 0.78)",
            "headlineFont": "Georgia, 'Times New Roman', serif",
            "bodyFont": "'Segoe UI', system-ui, -apple-system, sans-serif",
        },
        movements=movements,
        beats=beats,
        rising_beats=["approach", "dread"],
        default_voice="hm_omega",
        default_wpm=112,
        default_speed=0.94,
        bg_color="#0c0906",
    )


def _make_mythology_genre() -> GenreProfile:
    beats = {
        "establish": Beat(
            name="establish",
            role="Invoke the sacred era, divine balance, and majestic temple realm.",
            visuals=[
                {"type": "forest", "fog": 0.35, "moonlight": 0.85},
                {"type": "house", "fog": 0.3, "lit": True, "moonlight": 0.75},
            ],
            ambience=("amb_wind", 0.38),
            effects={"vignette": 0.75, "grain": 0.3, "dust": 0.35, "zoom": 0.03, "drift": "in"},
            cues=[(0.2, "sfx_bell", 0.55)],
            tension=0.0,
            emotion="calm",
            gesture="none",
            acting_intensity=0.4,
            character=False,
        ),
        "dread": Beat(
            name="dread",
            role="The challenge to dharma, divine prophecy, or gathering storm.",
            visuals=[
                {"type": "candle", "glow": 0.9},
                {"type": "fog", "density": 0.6, "glow": 0.65},
            ],
            ambience=("amb_drone", 0.45),
            effects={"vignette": 0.8, "grain": 0.35, "dust": 0.3, "zoom": 0.03, "drift": "in"},
            cues=[(0.4, "sfx_bell", 0.6)],
            tension=0.25,
            emotion="worried",
            gesture="none",
            acting_intensity=0.6,
            character=True,
        ),
        "approach": Beat(
            name="approach",
            role="Heroic devotion, gathering celestial weapons, and epic quest.",
            visuals=[
                {"type": "figure", "distance": 0.45, "fog": 0.35, "glow": 0.8},
                {"type": "forest", "fog": 0.4, "moonlight": 0.8},
            ],
            ambience=("amb_drone", 0.48),
            effects={"vignette": 0.82, "grain": 0.4, "dust": 0.35, "zoom": 0.045, "drift": "in"},
            cues=[(0.35, "sfx_riser", 0.45)],
            tension=0.5,
            emotion="calm",
            gesture="reach",
            acting_intensity=0.75,
            character=True,
        ),
        "reveal": Beat(
            name="reveal",
            role="Monumental clash of dharma, divine intervention, and cosmic power.",
            visuals=[
                {"type": "candle", "glow": 1.0},
                {"type": "figure", "distance": 0.2, "fog": 0.25, "glow": 1.0},
                {"type": "eyes", "pairs": 3, "spread": 0.65},
            ],
            ambience=("amb_drone", 0.55),
            effects={"vignette": 0.88, "grain": 0.45, "dust": 0.4, "shake": 0.28, "zoom": 0.05, "drift": "in"},
            cues=[(0.05, "sfx_impact", 0.9), (0.6, "sfx_bell", 0.85)],
            tension=0.9,
            emotion="shock",
            gesture="recoil",
            acting_intensity=1.0,
            character=True,
        ),
        "react": Beat(
            name="react",
            role="Sacred awe, divine blessing, and restoration of dharma.",
            visuals=[
                {"type": "candle", "glow": 0.85},
                {"type": "forest", "fog": 0.3, "moonlight": 0.85},
            ],
            ambience=("amb_wind", 0.38),
            effects={"vignette": 0.76, "grain": 0.3, "dust": 0.3, "zoom": 0.025, "drift": "out"},
            cues=[(0.4, "sfx_bell", 0.6)],
            tension=0.2,
            emotion="calm",
            gesture="clutch",
            acting_intensity=0.5,
            character=True,
        ),
        "resolve": Beat(
            name="resolve",
            role="Eternal celebration, temple chimes, and timeless wisdom.",
            visuals=[
                {"type": "house", "fog": 0.2, "lit": True, "moonlight": 0.85},
                {"type": "forest", "fog": 0.25, "moonlight": 0.9},
            ],
            ambience=("amb_wind", 0.32),
            effects={"vignette": 0.72, "grain": 0.25, "dust": 0.25, "zoom": 0.025, "drift": "out"},
            cues=[(0.7, "sfx_bell", 0.75)],
            tension=0.08,
            emotion="calm",
            gesture="none",
            acting_intensity=0.35,
            character=True,
        ),
    }
    movements = {
        "hook": Movement("hook", "Open with a timeless verse, divine prophecy, or proclamation of eternal dharma.", "establish", 0.05),
        "setup": Movement("setup", "Introduce the sacred era, the holy kingdom, the revered sage/warrior, and the prevailing balance.", "establish", 0.13),
        "inciting": Movement("inciting", "The grand challenge to dharma or cosmic order: a boon, a curse, an unstoppable adversary, or divine decree.", "dread", 0.12),
        "rising": Movement("rising", "Heroic devotion, epic quests, gathering of celestial power, and escalating battles of virtue.", "approach", 0.32),
        "climax": Movement("climax", "The monumental clash of destiny and divine power, where dharma is upheld against impossible odds.", "reveal", 0.16),
        "resolution": Movement("resolution", "The restoration of cosmic balance, celebration of divine justice, and blessing upon the world.", "react", 0.14),
        "button": Movement("button", "An eternal spiritual lesson and salute to the timeless legend.", "resolve", 0.08),
    }
    return GenreProfile(
        key="mythology",
        name="Mythology & Epic Legends",
        aliases=["mythology", "mythological", "epic", "legend", "puran", "vedic", "ramayan", "mahabharat", "deva", "avatar", "warrior", "hindu_mythology", "myth"],
        tone="epic, majestic, revered, heroic, and profound",
        writer_instructions=(
            "You are writing narration for an epic mythological and legendary story video. "
            "Invoke timeless grandeur, heroic battles of dharma, divine interventions, and ancient wisdom. Bring sacred legends to life with majestic reverence."
        ),
        system_prompt="You write narration for grand mythological epics, ancient legends, divine chronicles, and heroic folklore videos.",
        visual_style_prefix="Grand epic mythological painting, ancient Indian temple art style, divine golden aura, dramatic clouds, majestic celestial glory: ",
        visual_style_suffix=", masterpiece epic fantasy art, opulent divine lighting, sacred atmosphere.",
        search_keywords_map={
            "forest": "sacred ashram himalayan pine forest ancient rishi",
            "house": "grand ancient golden temple sanctum vedic kingdom",
            "grave": "ancient battlefield kurukshetra timeless monument",
            "corridor": "majestic royal palace pillars golden temple hall",
            "candle": "sacred diya lamp temple aarti golden flame",
            "figure": "heroic ancient warrior divine avatar glowing aura",
            "eyes": "divine celestial radiance golden third eye beam",
            "rain": "divine monsoon over sacred mountains ganga river",
            "fog": "sacred mist himalayan kailash peaks morning dawn",
            "void": "cosmic vishwaroop celestial universe mahavishnu cosmos",
        },
        default_keywords="ancient golden temple sacred himalaya epic mythological painting",
        theme={
            "accent": "#ff9933",
            "text": "#fffdf5",
            "muted": "#fed7aa",
            "panel": "rgba(38, 20, 8, 0.72)",
            "headlineFont": "Georgia, 'Times New Roman', serif",
            "bodyFont": "'Segoe UI', system-ui, -apple-system, sans-serif",
        },
        movements=movements,
        beats=beats,
        rising_beats=["approach", "dread"],
        bg_color="#120804",
    )


def _make_motivational_genre() -> GenreProfile:
    beats = {
        "establish": Beat(
            name="establish",
            role="Set the arena of struggle and quiet determination.",
            visuals=[
                {"type": "forest", "fog": 0.4, "moonlight": 0.6},
                {"type": "house", "fog": 0.3, "lit": True, "moonlight": 0.5},
            ],
            ambience=("amb_wind", 0.35),
            effects={"vignette": 0.72, "grain": 0.28, "dust": 0.2, "zoom": 0.03, "drift": "in"},
            cues=[(0.2, "sfx_bell", 0.4)],
            tension=0.0,
            emotion="calm",
            gesture="none",
            acting_intensity=0.35,
            character=False,
        ),
        "dread": Beat(
            name="dread",
            role="The defining crossroad and commitment to transform.",
            visuals=[
                {"type": "candle", "glow": 0.9},
                {"type": "fog", "density": 0.5, "glow": 0.6},
            ],
            ambience=("amb_drone", 0.4),
            effects={"vignette": 0.78, "grain": 0.32, "dust": 0.25, "zoom": 0.03, "drift": "in"},
            cues=[(0.4, "sfx_bell", 0.5)],
            tension=0.2,
            emotion="worried",
            gesture="none",
            acting_intensity=0.55,
            character=True,
        ),
        "approach": Beat(
            name="approach",
            role="Relentless discipline and grinding through adversity.",
            visuals=[
                {"type": "figure", "distance": 0.45, "fog": 0.35, "glow": 0.6},
                {"type": "corridor", "flicker": 0.4, "depth": 0.6},
            ],
            ambience=("amb_drone", 0.45),
            effects={"vignette": 0.8, "grain": 0.38, "dust": 0.3, "zoom": 0.045, "drift": "in"},
            cues=[(0.35, "sfx_riser", 0.45)],
            tension=0.5,
            emotion="calm",
            gesture="reach",
            acting_intensity=0.75,
            character=True,
        ),
        "reveal": Beat(
            name="reveal",
            role="The summit breakthrough and supreme triumph over doubt.",
            visuals=[
                {"type": "candle", "glow": 1.0},
                {"type": "figure", "distance": 0.2, "fog": 0.2, "glow": 0.9},
            ],
            ambience=("amb_drone", 0.5),
            effects={"vignette": 0.85, "grain": 0.4, "dust": 0.35, "shake": 0.2, "zoom": 0.05, "drift": "in"},
            cues=[(0.05, "sfx_impact", 0.85), (0.6, "sfx_bell", 0.8)],
            tension=0.85,
            emotion="calm",
            gesture="reach",
            acting_intensity=0.95,
            character=True,
        ),
        "react": Beat(
            name="react",
            role="Internal strength, earned clarity, and deep confidence.",
            visuals=[
                {"type": "candle", "glow": 0.85},
                {"type": "forest", "fog": 0.3, "moonlight": 0.8},
            ],
            ambience=("amb_wind", 0.35),
            effects={"vignette": 0.74, "grain": 0.28, "dust": 0.25, "zoom": 0.02, "drift": "out"},
            cues=[(0.4, "sfx_bell", 0.55)],
            tension=0.15,
            emotion="calm",
            gesture="clutch",
            acting_intensity=0.5,
            character=True,
        ),
        "resolve": Beat(
            name="resolve",
            role="Empowered path forward and continuous mastery.",
            visuals=[
                {"type": "house", "fog": 0.2, "lit": True, "moonlight": 0.85},
                {"type": "forest", "fog": 0.2, "moonlight": 0.9},
            ],
            ambience=("amb_wind", 0.3),
            effects={"vignette": 0.7, "grain": 0.25, "dust": 0.2, "zoom": 0.025, "drift": "out"},
            cues=[(0.7, "sfx_bell", 0.7)],
            tension=0.08,
            emotion="calm",
            gesture="none",
            acting_intensity=0.35,
            character=True,
        ),
    }
    movements = {
        "hook": Movement("hook", "Open with a punchy, challenging truth about potential, adversity, and personal limits.", "establish", 0.05),
        "setup": Movement("setup", "Describe the everyday struggle, the weight of doubt, or the ordinary beginning before change began.", "establish", 0.13),
        "inciting": Movement("inciting", "The awakening: the moment of decision when settling is no longer an option and commitment takes hold.", "dread", 0.12),
        "rising": Movement("rising", "The crucible of discipline: overcoming setbacks, pushing through fatigue, and silencing inner excuses.", "approach", 0.32),
        "climax": Movement("climax", "The ultimate breakthrough: proving resilience when everything tested the resolve to give up.", "reveal", 0.16),
        "resolution": Movement("resolution", "The transformed self: internal peace, earned strength, and the clarity of continuous growth.", "react", 0.14),
        "button": Movement("button", "A direct, powerful call to action that inspires immediate courage in the listener.", "resolve", 0.08),
    }
    return GenreProfile(
        key="motivational",
        name="Motivational & Self-Improvement",
        aliases=["motivational", "motivation", "inspiration", "inspirational", "life_lessons", "parable", "success", "discipline", "mindset", "growth", "habits"],
        tone="powerful, uplifting, motivational, reflective, and empowering",
        writer_instructions=(
            "You are writing narration for an inspiring motivational and self-improvement video. "
            "Focus on overcoming struggle, building inner strength, discipline, resilience, and personal transformation with punchy, energizing prose."
        ),
        system_prompt="You write narration for inspiring motivational videos, life parables, and self-improvement stories that empower the listener.",
        visual_style_prefix="Inspirational cinematic photography, golden hour sunrise, breathtaking mountain landscape, epic dramatic lighting, triumph: ",
        visual_style_suffix=", cinematic 8k quality, motivational masterpiece, warm radiant glow.",
        search_keywords_map={
            "forest": "solitary trail sunrise through tall trees discipline",
            "house": "quiet study room morning sunlight workspace books",
            "grave": "stepping stones across river overcoming obstacles",
            "corridor": "long modern illuminated pathway tunnel to light",
            "candle": "single bright candle flame in darkness hope focus",
            "figure": "determined runner athlete standing on mountain peak",
            "eyes": "focused determined gaze looking toward horizon",
            "rain": "training through heavy storm relentless dedication",
            "fog": "mountain path clearing at dawn sunrise breakthrough",
            "void": "golden sunrise vast open sky unlimited potential",
        },
        default_keywords="mountain sunrise athlete training golden hour summit inspirational",
        theme={
            "accent": "#eab308",
            "text": "#ffffff",
            "muted": "#cbd5e1",
            "panel": "rgba(15, 23, 42, 0.70)",
            "headlineFont": "'Segoe UI', system-ui, sans-serif",
            "bodyFont": "'Segoe UI', system-ui, -apple-system, sans-serif",
        },
        movements=movements,
        beats=beats,
        rising_beats=["approach", "dread"],
        default_wpm=130,
        default_speed=1.05,
        bg_color="#090d16",
    )


def _make_drama_genre() -> GenreProfile:
    beats = {
        "establish": Beat(
            name="establish",
            role="Set the intimate, heartfelt atmosphere and emotional bonds.",
            visuals=[
                {"type": "forest", "fog": 0.35, "moonlight": 0.7},
                {"type": "house", "fog": 0.3, "lit": True, "moonlight": 0.6},
            ],
            ambience=("amb_wind", 0.32),
            effects={"vignette": 0.7, "grain": 0.25, "dust": 0.2, "zoom": 0.025, "drift": "in"},
            cues=[(0.2, "sfx_bell", 0.4)],
            tension=0.0,
            emotion="calm",
            gesture="none",
            acting_intensity=0.35,
            character=False,
        ),
        "dread": Beat(
            name="dread",
            role="The emotional test, painful realization, or impending separation.",
            visuals=[
                {"type": "candle", "glow": 0.85},
                {"type": "fog", "density": 0.5, "glow": 0.5},
            ],
            ambience=("amb_drone", 0.35),
            effects={"vignette": 0.76, "grain": 0.3, "dust": 0.25, "zoom": 0.025, "drift": "in"},
            cues=[(0.4, "sfx_impact", 0.35)],
            tension=0.2,
            emotion="sad",
            gesture="none",
            acting_intensity=0.55,
            character=True,
        ),
        "approach": Beat(
            name="approach",
            role="Deepening sacrifices, unspoken longing, and emotional trials.",
            visuals=[
                {"type": "figure", "distance": 0.5, "fog": 0.35, "glow": 0.6},
                {"type": "forest", "fog": 0.4, "moonlight": 0.7},
            ],
            ambience=("amb_drone", 0.38),
            effects={"vignette": 0.78, "grain": 0.35, "dust": 0.3, "zoom": 0.035, "drift": "in"},
            cues=[(0.35, "sfx_riser", 0.35)],
            tension=0.4,
            emotion="sad",
            gesture="clutch",
            acting_intensity=0.7,
            character=True,
        ),
        "reveal": Beat(
            name="reveal",
            role="Heartfelt confrontation, tearful sacrifice, or touching reconciliation.",
            visuals=[
                {"type": "candle", "glow": 1.0},
                {"type": "figure", "distance": 0.25, "fog": 0.25, "glow": 0.8},
            ],
            ambience=("amb_drone", 0.45),
            effects={"vignette": 0.82, "grain": 0.38, "dust": 0.3, "shake": 0.15, "zoom": 0.04, "drift": "in"},
            cues=[(0.05, "sfx_impact", 0.65), (0.6, "sfx_bell", 0.7)],
            tension=0.75,
            emotion="sad",
            gesture="clutch",
            acting_intensity=0.9,
            character=True,
        ),
        "react": Beat(
            name="react",
            role="Tender closure, shared forgiveness, and emotional warmth.",
            visuals=[
                {"type": "candle", "glow": 0.8},
                {"type": "forest", "fog": 0.3, "moonlight": 0.75},
            ],
            ambience=("amb_wind", 0.32),
            effects={"vignette": 0.72, "grain": 0.25, "dust": 0.2, "zoom": 0.02, "drift": "out"},
            cues=[(0.4, "sfx_bell", 0.45)],
            tension=0.15,
            emotion="calm",
            gesture="clutch",
            acting_intensity=0.45,
            character=True,
        ),
        "resolve": Beat(
            name="resolve",
            role="Peaceful closure and love enduring through time.",
            visuals=[
                {"type": "house", "fog": 0.2, "lit": True, "moonlight": 0.8},
                {"type": "forest", "fog": 0.25, "moonlight": 0.8},
            ],
            ambience=("amb_wind", 0.28),
            effects={"vignette": 0.68, "grain": 0.22, "dust": 0.18, "zoom": 0.02, "drift": "out"},
            cues=[(0.7, "sfx_bell", 0.6)],
            tension=0.08,
            emotion="calm",
            gesture="none",
            acting_intensity=0.3,
            character=True,
        ),
    }
    movements = {
        "hook": Movement("hook", "Open with a poignant reflection on time, memory, love, or the fragility of human relationships.", "establish", 0.05),
        "setup": Movement("setup", "Establish the emotional bond, the cherished routine, and the heartfelt world of the characters.", "establish", 0.13),
        "inciting": Movement("inciting", "The emotional rift or unforeseen test: separation, misunderstanding, illness, or an unavoidable choice.", "dread", 0.12),
        "rising": Movement("rising", "Deepening emotional stakes, heartfelt sacrifices, unspoken longing, and trials of dedication.", "approach", 0.32),
        "climax": Movement("climax", "The poignant confrontation or emotional reconciliation where truth and love prevail.", "reveal", 0.16),
        "resolution": Movement("resolution", "The heartfelt closure: lessons learned, healing embraced, and love enduring beyond circumstance.", "react", 0.14),
        "button": Movement("button", "A tender parting thought that leaves a warm, lasting impression in the listener's heart.", "resolve", 0.08),
    }
    return GenreProfile(
        key="drama",
        name="Emotional Storytelling & Drama",
        aliases=["drama", "emotional", "heartfelt", "love", "family", "friendship", "sacrifice", "sad", "nostalgia", "story", "romance", "human_interest"],
        tone="deeply emotional, heartfelt, touching, intimate, and poignant",
        writer_instructions=(
            "You are writing narration for a heartfelt and emotional human story video. "
            "Focus on meaningful relationships, empathy, personal sacrifices, love, nostalgia, and bittersweet moments that touch the soul."
        ),
        system_prompt="You write narration for deeply emotional, heartwarming, and poignant human story videos.",
        visual_style_prefix="Cinematic emotional drama digital art, soft warm lighting, sentimental atmosphere, beautiful painterly realism: ",
        visual_style_suffix=", cinematic composition, touching depth of field, delicate pastel and warm tones.",
        search_keywords_map={
            "forest": "peaceful autumn woods golden leaves gentle sunlight",
            "house": "warm cozy vintage family home living room lamp",
            "grave": "quiet bench in autumn park memories nostalgia",
            "corridor": "old train station platform vintage nostalgia",
            "candle": "warm candle glow old letters vintage photograph",
            "figure": "two friends talking together gentle silhouette warmth",
            "eyes": "tender emotional loving eyes heartfelt look",
            "rain": "gentle rain on window pane cozy nostalgic evening",
            "fog": "soft morning mist over quiet peaceful village",
            "void": "sunset over quiet ocean horizon peaceful evening",
        },
        default_keywords="cozy nostalgic sunset vintage home emotional reunion cinematic",
        theme={
            "accent": "#f43f5e",
            "text": "#fff1f2",
            "muted": "#fda4af",
            "panel": "rgba(30, 10, 20, 0.70)",
            "headlineFont": "Georgia, 'Times New Roman', serif",
            "bodyFont": "'Segoe UI', system-ui, -apple-system, sans-serif",
        },
        movements=movements,
        beats=beats,
        rising_beats=["approach", "dread"],
        bg_color="#10060c",
    )


def _make_comedy_genre() -> GenreProfile:
    beats = {
        "establish": Beat(
            name="establish",
            role="Set up the cheerful normal setting and quirky setup.",
            visuals=[
                {"type": "forest", "fog": 0.2, "moonlight": 0.9},
                {"type": "house", "fog": 0.15, "lit": True, "moonlight": 0.85},
            ],
            ambience=("amb_wind", 0.25),
            effects={"vignette": 0.55, "grain": 0.15, "dust": 0.15, "zoom": 0.02, "drift": "in"},
            cues=[(0.2, "sfx_bell", 0.4)],
            tension=0.0,
            emotion="calm",
            gesture="none",
            acting_intensity=0.4,
            character=False,
        ),
        "dread": Beat(
            name="dread",
            role="The initial comical mishap or funny misunderstanding.",
            visuals=[
                {"type": "corridor", "flicker": 0.2, "depth": 0.4},
                {"type": "figure", "distance": 0.6, "fog": 0.2, "glow": 0.5},
            ],
            ambience=("amb_hum", 0.28),
            effects={"vignette": 0.6, "grain": 0.2, "dust": 0.15, "zoom": 0.025, "drift": "in"},
            cues=[(0.4, "sfx_stinger", 0.35)],
            tension=0.2,
            emotion="worried",
            gesture="none",
            acting_intensity=0.6,
            character=True,
        ),
        "approach": Beat(
            name="approach",
            role="Comical escalation and clumsy attempts to fix the mess.",
            visuals=[
                {"type": "figure", "distance": 0.4, "fog": 0.2, "glow": 0.6},
                {"type": "corridor", "flicker": 0.3, "depth": 0.5},
            ],
            ambience=("amb_drone", 0.3),
            effects={"vignette": 0.62, "grain": 0.2, "dust": 0.2, "zoom": 0.04, "drift": "in"},
            cues=[(0.35, "sfx_riser", 0.35)],
            tension=0.45,
            emotion="worried",
            gesture="reach",
            acting_intensity=0.75,
            character=True,
        ),
        "reveal": Beat(
            name="reveal",
            role="Peak hilarious absurdity and chaotic punchline showdown.",
            visuals=[
                {"type": "candle", "glow": 0.95},
                {"type": "figure", "distance": 0.25, "fog": 0.2, "glow": 0.8},
            ],
            ambience=("amb_wind", 0.35),
            effects={"vignette": 0.68, "grain": 0.22, "dust": 0.2, "shake": 0.25, "zoom": 0.045, "drift": "in"},
            cues=[(0.05, "sfx_impact", 0.8), (0.6, "sfx_bell", 0.7)],
            tension=0.8,
            emotion="shock",
            gesture="recoil",
            acting_intensity=0.9,
            character=True,
        ),
        "react": Beat(
            name="react",
            role="Laughing out loud at the ridiculous resolution.",
            visuals=[
                {"type": "fog", "density": 0.3, "glow": 0.7},
                {"type": "house", "fog": 0.2, "lit": True},
            ],
            ambience=("amb_wind", 0.25),
            effects={"vignette": 0.58, "grain": 0.15, "dust": 0.15, "zoom": 0.02, "drift": "out"},
            cues=[(0.4, "sfx_bell", 0.45)],
            tension=0.15,
            emotion="calm",
            gesture="clutch",
            acting_intensity=0.45,
            character=True,
        ),
        "resolve": Beat(
            name="resolve",
            role="Cheerful parting wink and final comedic landing.",
            visuals=[
                {"type": "house", "fog": 0.15, "lit": True, "moonlight": 0.9},
                {"type": "forest", "fog": 0.2, "moonlight": 0.9},
            ],
            ambience=("amb_wind", 0.22),
            effects={"vignette": 0.55, "grain": 0.15, "dust": 0.12, "zoom": 0.02, "drift": "out"},
            cues=[(0.7, "sfx_bell", 0.6)],
            tension=0.05,
            emotion="calm",
            gesture="none",
            acting_intensity=0.3,
            character=True,
        ),
    }
    movements = {
        "hook": Movement("hook", "Open with an absurd observation, hilarious dilemma, or funny universal truth.", "establish", 0.05),
        "setup": Movement("setup", "Introduce the quirky character, their overly ambitious simple plan, and the ordinary setting.", "establish", 0.13),
        "inciting": Movement("inciting", "The initial comical blunder: a small mistake or funny misunderstanding that starts rolling downhill.", "dread", 0.12),
        "rising": Movement("rising", "Escalating comedy of errors: every desperate attempt to fix the blunder makes the chaos twice as funny.", "approach", 0.32),
        "climax": Movement("climax", "The peak hilarious catastrophe where everything collides in the most ridiculous way imaginable.", "reveal", 0.16),
        "resolution": Movement("resolution", "The dust settles with a funny realization, sheepish grins, and unexpected comical silver linings.", "react", 0.14),
        "button": Movement("button", "A sharp, witty punchline that delivers the final laugh.", "resolve", 0.08),
    }
    return GenreProfile(
        key="comedy",
        name="Comedy & Satire",
        aliases=["comedy", "funny", "humor", "humour", "satire", "joke", "parody", "hilarious", "fun", "prank", "mischief"],
        tone="humorous, witty, lively, lighthearted, and entertaining",
        writer_instructions=(
            "You are writing narration for a funny, witty, and lighthearted comedy video. "
            "Craft humorous character dynamics, ironic situations, comical blunders, and entertaining twists with sharp comedic timing."
        ),
        system_prompt="You write narration for humorous, witty, and entertaining comedy videos.",
        visual_style_prefix="Vibrant and colorful 2D animated illustration style, expressive, bright sunny lighting, humorous cartoon realism: ",
        visual_style_suffix=", clean vibrant aesthetic, cheerful energetic color palette.",
        search_keywords_map={
            "forest": "sunny vibrant playful cartoon forest cheerful animals",
            "house": "quirky colorful messy cozy modern kitchen room",
            "grave": "comical garden with wacky misplaced garden gnomes",
            "corridor": "bustling funny office hallway slapstick chaos",
            "candle": "birthday cake with exploding sparklers funny party",
            "figure": "expressive funny comical character scratching head confused",
            "eyes": "wide funny surprised cartoon eyes funny face",
            "rain": "funny guy holding tiny broken umbrella in downpour",
            "fog": "smoke coming out of funny failed science kitchen experiment",
            "void": "bright colorful pop art burst cheerful confetti",
        },
        default_keywords="colorful marketplace funny mishap vibrant cheerful illustration",
        theme={
            "accent": "#10b981",
            "text": "#f0fdf4",
            "muted": "#86efac",
            "panel": "rgba(6, 44, 28, 0.70)",
            "headlineFont": "'Segoe UI', system-ui, sans-serif",
            "bodyFont": "'Segoe UI', system-ui, -apple-system, sans-serif",
        },
        movements=movements,
        beats=beats,
        rising_beats=["approach", "dread"],
        bg_color="#04140c",
    )


def _make_explainer_genre() -> GenreProfile:
    beats = {
        "establish": Beat(
            name="establish",
            role="Pose the compelling mystery, scientific phenomenon, or central question.",
            visuals=[
                {"type": "corridor", "flicker": 0.1, "depth": 0.6},
                {"type": "figure", "distance": 0.5, "fog": 0.2, "glow": 0.75},
            ],
            ambience=("amb_hum", 0.32),
            effects={"vignette": 0.55, "grain": 0.15, "dust": 0.18, "zoom": 0.025, "drift": "in"},
            cues=[(0.15, "sfx_whoosh", 0.45)],
            tension=0.15,
            emotion="calm",
            gesture="none",
            acting_intensity=0.45,
            character=False,
        ),
        "dread": Beat(
            name="dread",
            role="The counter-intuitive anomaly: why standard intuition fails.",
            visuals=[
                {"type": "candle", "glow": 0.85},
                {"type": "corridor", "flicker": 0.2, "depth": 0.5},
            ],
            ambience=("amb_drone", 0.35),
            effects={"vignette": 0.6, "grain": 0.18, "dust": 0.2, "zoom": 0.03, "drift": "in"},
            cues=[(0.3, "sfx_glitch", 0.4)],
            tension=0.35,
            emotion="calm",
            gesture="explain",
            acting_intensity=0.55,
            character=True,
        ),
        "approach": Beat(
            name="approach",
            role="Step-by-step breakdown: presenting evidence, data layers, theories, and experiments.",
            visuals=[
                {"type": "figure", "distance": 0.4, "fog": 0.2, "glow": 0.65},
                {"type": "corridor", "flicker": 0.25, "depth": 0.7},
            ],
            ambience=("amb_hum", 0.4),
            effects={"vignette": 0.65, "grain": 0.2, "dust": 0.22, "zoom": 0.035, "drift": "in"},
            cues=[(0.25, "sfx_riser", 0.4)],
            tension=0.55,
            emotion="calm",
            gesture="explain",
            acting_intensity=0.7,
            character=True,
        ),
        "reveal": Beat(
            name="reveal",
            role="The breakthrough reveal: the underlying scientific truth or shocking realization.",
            visuals=[
                {"type": "candle", "glow": 1.0},
                {"type": "figure", "distance": 0.3, "fog": 0.15, "glow": 0.85},
            ],
            ambience=("amb_hum", 0.45),
            effects={"vignette": 0.7, "grain": 0.2, "dust": 0.2, "shake": 0.1, "zoom": 0.045, "drift": "in"},
            cues=[(0.05, "sfx_cinematic_hit", 0.7), (0.55, "sfx_bell", 0.65)],
            tension=0.8,
            emotion="calm",
            gesture="explain",
            acting_intensity=0.85,
            character=True,
        ),
        "react": Beat(
            name="react",
            role="Real-world implications: how this changes technology, understanding, and the future.",
            visuals=[
                {"type": "fog", "density": 0.3, "glow": 0.7},
                {"type": "house", "fog": 0.2, "lit": True},
            ],
            ambience=("amb_wind", 0.32),
            effects={"vignette": 0.58, "grain": 0.15, "dust": 0.15, "zoom": 0.02, "drift": "out"},
            cues=[(0.35, "sfx_whoosh", 0.4)],
            tension=0.25,
            emotion="calm",
            gesture="explain",
            acting_intensity=0.5,
            character=True,
        ),
        "resolve": Beat(
            name="resolve",
            role="The final memorable takeaway or mind-expanding insight.",
            visuals=[
                {"type": "forest", "fog": 0.2, "moonlight": 0.8},
                {"type": "house", "fog": 0.15, "lit": True},
            ],
            ambience=("amb_warm", 0.28),
            effects={"vignette": 0.55, "grain": 0.12, "dust": 0.12, "zoom": 0.02, "drift": "out"},
            cues=[(0.65, "sfx_chime", 0.55)],
            tension=0.05,
            emotion="calm",
            gesture="none",
            acting_intensity=0.35,
            character=True,
        ),
    }
    movements = {
        "hook": Movement("hook", "Open with a shocking fact, counter-intuitive question, or mind-bending premise that immediately grips the audience.", "establish", 0.06),
        "setup": Movement("setup", "Establish the core mechanism, background context, or foundational baseline.", "establish", 0.12),
        "inciting": Movement("inciting", "The anomaly: why common explanations fail and what makes this phenomenon strange.", "dread", 0.12),
        "rising": Movement("rising", "Step-by-step breakdown: presenting evidence, data layers, mechanics, and experiments.", "approach", 0.32),
        "climax": Movement("climax", "The breakthrough reveal: the underlying scientific principle, mechanism, or hidden truth.", "reveal", 0.16),
        "resolution": Movement("resolution", "Real-world impact: how this changes our understanding, daily life, or future technology.", "react", 0.14),
        "button": Movement("button", "A thought-provoking concluding insight that leaves the audience pondering.", "resolve", 0.08),
    }
    return GenreProfile(
        key="explainer",
        name="Explainer & Deep Dive Breakdown",
        aliases=["explainer", "educational", "science_breakdown", "tech_breakdown", "facts_explainer", "case_study", "how_it_works", "infotainment", "deep_dive", "analysis", "science_explainer", "guide"],
        tone="clear, fascinating, authoritative, high-energy, intellectually engaging, and inquisitive",
        writer_instructions=(
            "You are writing narration for an engaging, high-retention educational and science explainer video. "
            "Break down complex ideas into crisp, fascinating revelations. Use compelling hooks, clear cause-and-effect explanations, vivid analogies, and mind-expanding insights."
        ),
        system_prompt="You write narration for captivating science, technology, case-study, and educational explainer videos with crystal-clear logic and viral hooks.",
        visual_style_prefix="Masterpiece 3D isometric infographic concept art, futuristic scientific visualization, clean high-tech lighting, detailed digital illustration: ",
        visual_style_suffix=", 8k render, octane render aesthetic, sleek modern technical composition, volumetric glow, vibrant accents, cinematic depth of field.",
        search_keywords_map={
            "forest": "pristine futuristic biosphere ecosystem scientific visualization",
            "house": "high-tech modern laboratory research facility architectural cutaway",
            "grave": "ancient archaeological artifact deep scanning blueprint",
            "corridor": "futuristic particle accelerator data server room holographic tunnel",
            "candle": "glowing quantum energy core floating holographic diagram",
            "figure": "visionary scientist engineer silhouette looking at holographic data",
            "eyes": "futuristic neural network biometric eye scanning cosmic visualization",
            "rain": "digital data stream matrix cybernetic weather simulation",
            "fog": "volumetric luminous quantum particle cloud nebula",
            "void": "deep space singularity event horizon astrophysics visualization",
        },
        default_keywords="futuristic 3D isometric scientific breakdown concept art high tech data visualization",
        theme={
            "accent": "#06b6d4",
            "text": "#f8fafc",
            "muted": "#94a3b8",
            "panel": "rgba(8, 20, 32, 0.78)",
            "headlineFont": "'Segoe UI', system-ui, sans-serif",
            "bodyFont": "'Segoe UI', system-ui, -apple-system, sans-serif",
        },
        movements=movements,
        beats=beats,
        rising_beats=["approach", "dread"],
        default_voice="hindi_documentary",
        default_wpm=125,
        default_speed=1.05,
        bg_color="#030712",
    )


def _make_horror_genre() -> GenreProfile:
    beats = {
        "establish": Beat(
            name="establish",
            role="Establish the ominous setting, eerie silence, or disturbing baseline normal before darkness descends.",
            visuals=[
                {"type": "forest", "fog": 0.8, "moonlight": 0.25},
                {"type": "house", "fog": 0.7, "lit": False, "moonlight": 0.3},
                {"type": "corridor", "flicker": 0.4, "depth": 0.6},
            ],
            ambience=("amb_drone", 0.42),
            effects={"vignette": 0.88, "grain": 0.42, "dust": 0.28, "zoom": 0.025, "drift": "in", "exposureFlicker": 0.12},
            cues=[(0.3, "sfx_whisper", 0.4), (0.75, "sfx_creak", 0.35)],
            tension=0.25,
            emotion="worried",
            gesture="none",
            acting_intensity=0.45,
            character=True,
        ),
        "dread": Beat(
            name="dread",
            role="The chilling anomaly: a whispered voice, cold breath, a silhouette where nobody should be.",
            visuals=[
                {"type": "void", "creep": 0.65},
                {"type": "corridor", "flicker": 0.75, "depth": 0.7},
                {"type": "fog", "density": 0.85, "glow": 0.25},
                {"type": "candle", "glow": 0.6},
            ],
            ambience=("amb_drone", 0.52),
            effects={"vignette": 0.92, "grain": 0.48, "dust": 0.3, "zoom": 0.03, "drift": "in", "exposureFlicker": 0.2},
            cues=[(0.25, "sfx_whisper", 0.55), (0.6, "sfx_heartbeat", 0.5)],
            tension=0.55,
            emotion="worried",
            gesture="reach",
            acting_intensity=0.65,
            character=True,
        ),
        "approach": Beat(
            name="approach",
            role="Tightening psychological trap: closing in on the terrifying source, heart pounding, footsteps echoing.",
            visuals=[
                {"type": "corridor", "flicker": 0.85, "depth": 0.8},
                {"type": "figure", "distance": 0.45, "fog": 0.65, "glow": 0.35},
                {"type": "grave", "fog": 0.8, "moonlight": 0.2},
            ],
            ambience=("amb_heartbeat", 0.58),
            effects={"vignette": 0.94, "grain": 0.52, "dust": 0.35, "zoom": 0.045, "drift": "in", "exposureFlicker": 0.25},
            cues=[(0.2, "sfx_riser", 0.6), (0.7, "sfx_creak", 0.55)],
            tension=0.75,
            emotion="shock",
            gesture="recoil",
            acting_intensity=0.85,
            character=True,
        ),
        "reveal": Beat(
            name="reveal",
            role="The horrific revelation / jumpscare: the demon unmasked, the sudden strike, bloodcurdling terror.",
            visuals=[
                {"type": "eyes", "pairs": 5, "spread": 0.85},
                {"type": "figure", "distance": 0.15, "fog": 0.4, "glow": 0.5},
                {"type": "void", "creep": 0.9},
            ],
            ambience=("amb_heartbeat", 0.65),
            effects={"vignette": 0.96, "grain": 0.58, "dust": 0.15, "shake": 0.4, "zoom": 0.06, "drift": "in", "exposureFlicker": 0.35},
            cues=[(0.05, "sfx_stinger", 1.0), (0.45, "sfx_impact", 0.75), (0.8, "sfx_screech", 0.65)],
            tension=1.0,
            emotion="shock",
            gesture="recoil",
            acting_intensity=1.0,
            character=True,
        ),
        "react": Beat(
            name="react",
            role="Paralyzed aftermath: gasping for breath in dark shock, realizing the horrifying truth or permanent curse.",
            visuals=[
                {"type": "void", "creep": 0.85},
                {"type": "candle", "glow": 0.45},
                {"type": "rain", "intensity": 0.6, "lightning": True},
            ],
            ambience=("amb_drone", 0.46),
            effects={"vignette": 0.94, "grain": 0.5, "dust": 0.2, "zoom": 0.02, "drift": "out", "exposureFlicker": 0.15},
            cues=[(0.3, "sfx_impact", 0.45), (0.7, "sfx_whisper", 0.4)],
            tension=0.35,
            emotion="sad",
            gesture="clutch",
            acting_intensity=0.6,
            character=True,
        ),
        "resolve": Beat(
            name="resolve",
            role="Chilling, unsettling lingering dread: the curse remains alive, the door slowly creaks open again.",
            visuals=[
                {"type": "grave", "fog": 0.85, "moonlight": 0.3},
                {"type": "rain", "intensity": 0.8, "lightning": True},
                {"type": "void", "creep": 0.95},
            ],
            ambience=("amb_wind", 0.42),
            effects={"vignette": 0.92, "grain": 0.48, "dust": 0.22, "zoom": 0.025, "drift": "out"},
            cues=[(0.4, "sfx_bell", 0.7), (0.8, "sfx_whisper", 0.5)],
            tension=0.2,
            emotion="worried",
            gesture="none",
            acting_intensity=0.45,
            character=True,
        ),
    }
    movements = {
        "hook": Movement("hook", "Open with a bloodcurdling hook, terrifying warning, or mysterious forbidden rule that stops the scroll instantly.", "establish", 0.06),
        "setup": Movement("setup", "Establish the isolated setting, eerie silence, and the protagonist stepping into forbidden territory.", "establish", 0.12),
        "inciting": Movement("inciting", "The first supernatural disturbance: an unexplained sound, a shifting shadow, or a sealed door swinging open.", "dread", 0.12),
        "rising": Movement("rising", "Rapidly escalating psychological terror: the entity closes in, escape routes vanish, and dread peaks.", "approach", 0.32),
        "climax": Movement("climax", "The terrifying confrontation or bloodcurdling jumpscare revelation at maximum visceral shock.", "reveal", 0.16),
        "resolution": Movement("resolution", "The horrifying aftermath: the grim fate realized, the sinister curse unleashed.", "react", 0.14),
        "button": Movement("button", "A chilling parting warning or eerie twist that leaves the viewer unsettled in the dark.", "resolve", 0.08),
    }
    return GenreProfile(
        key="horror",
        name="Horror & Suspense",
        aliases=["horror", "scary", "ghost", "haunted", "dread", "bhoot", "chudail", "creepy", "paranormal", "suspense", "thriller", "dark"],
        tone="slow, restrained, atmospheric, building dread and chilling suspense",
        writer_instructions=(
            "You are writing narration for a chilling horror and suspense story video. "
            "Build psychological dread, ominous mystery, eerie tension, and a spine-tingling resolution with restrained, detail-oriented terror."
        ),
        system_prompt="You write narration for chilling suspense and horror videos. You build dark atmosphere and gripping tension.",
        visual_style_prefix="Masterpiece cinematic dark horror atmosphere, terrifying 2D digital concept art, eerie volumetric fog, sinister low-key lighting, deep dramatic shadows, ominous mystery: ",
        visual_style_suffix=", highly detailed dark atmospheric horror background, 8k cinematic composition, chilling moody lighting, photorealistic textures.",
        search_keywords_map={
            "forest": "haunted dense misty forest twisted black trees moonlight eerie",
            "house": "abandoned creepy haunted mansion dark cracked windows flickering light",
            "grave": "misty ancient cemetery graveyard crooked tombstones gothic horror night",
            "corridor": "dark decaying asylum corridor flickering lights blood red glow shadows",
            "candle": "single dying candle flickering in pitch black sinister room",
            "figure": "tall shadowy paranormal entity demonic silhouette glowing eyes mist",
            "eyes": "sinister glowing yellow red eyes staring from pitch black void",
            "rain": "torrential thunder storm dark rain lightning striking abandoned haveli",
            "fog": "dense swirling cold ground mist supernatural dark aura",
            "void": "pitch black abyss creeping darkness eldritch horror shadow tendrils",
        },
        default_keywords="abandoned haunted mansion eerie mist dark cemetery horror atmosphere",
        theme={
            "accent": "#dc2626",
            "text": "#fef2f2",
            "muted": "#fca5a5",
            "panel": "rgba(8, 6, 10, 0.82)",
            "headlineFont": "Georgia, 'Times New Roman', serif",
            "bodyFont": "'Segoe UI', system-ui, -apple-system, sans-serif",
        },
        movements=movements,
        beats=beats,
        rising_beats=["approach", "dread"],
        default_voice="cinematic_horror",
        default_wpm=110,
        default_speed=1.0,
        bg_color="#04060a",
    )


GENRES: dict[str, GenreProfile] = {
    "anime": _make_anime_genre(),
    "fantasy": _make_fantasy_genre(),
    "scifi": _make_scifi_genre(),
    "documentary": _make_documentary_genre(),
    "explainer": _make_explainer_genre(),
    "mythology": _make_mythology_genre(),
    "motivational": _make_motivational_genre(),
    "drama": _make_drama_genre(),
    "comedy": _make_comedy_genre(),
    "horror": _make_horror_genre(),
    "ancient_history": _make_ancient_history_genre(),
}


def get_genre(name_or_alias: str | None) -> GenreProfile:
    """Look up a GenreProfile by name or alias, falling back to horror."""
    if not name_or_alias:
        return GENRES["horror"]
    norm = name_or_alias.lower().strip().replace("-", "_").replace(" ", "_")
    if norm in GENRES:
        return GENRES[norm]
    for g in GENRES.values():
        if norm in g.aliases:
            return g
    return GENRES["horror"]


def detect_genre(topic: str = "", explicit_genre: str | None = None) -> GenreProfile:
    """Automatically detect or resolve the best GenreProfile for a topic."""
    if explicit_genre:
        return get_genre(explicit_genre)

    t = topic.lower().strip()
    if not t:
        return GENRES["horror"]

    # 1. Anime & Action (e.g. "anime", "fight", "battle", "goku", "naruto", "shonen", "ninja", "superhero")
    anime_kw = [
        "anime", "action", "fight", "fighting", "battle", "battles", "goku", "naruto",
        "shonen", "manga", "dragon ball", "dragonball", "dbz", "luffy", "one piece",
        "sasuke", "vegeta", "bleach", "ichigo", "jujutsu", "sukuna", "gojo",
        "demon slayer", "tanjiro", "hero academia", "deku", "superhero", "superheroes",
        "martial art", "martial arts", "ninja", "shinobi", "samurai", "katana",
        "powers", "superpower", "superpowers", "warrior", "warriors", "clash", "showdown",
        "duel", "brawler", "combat", "ladai", "mukabla", "yuddh", "dangal",
        "एनिमे", "लड़ाई", "युद्ध", "मुकाबला", "दंगल", "पहलवान", "योद्धा", "कराटे", "मार्शल आर्ट"
    ]
    for kw in anime_kw:
        if re.search(r"\b" + re.escape(kw) + r"\b", t) or kw in t:
            return GENRES["anime"]

    # 2. Fantasy & Fairy Tale (high priority: "son pari", "pari", "magic", "fairy")
    fantasy_kw = [
        "son pari", "sonpari", "pari", "fairy", "fairies", "magic", "magical", "jadui",
        "jadu", "wizard", "witch", "princess", "prince", "rajkumari", "rajkumar",
        "enchanted", "fantasi", "fantasy", "wonderland", "fae", "dragon", "mermaid",
        "परी", "सोन परी", "जादुई", "जादू", "राजकुमारी", "राजकुमार", "महल", "परियों"
    ]
    for kw in fantasy_kw:
        if re.search(r"\b" + re.escape(kw) + r"\b", t) or kw in t:
            return GENRES["fantasy"]

    # 3. Ancient Civilizations & Temple Engineering / Regional Folklore
    ancient_kw = [
        "ancient civilization", "ancient civilizations", "temple engineering", "ancient temple",
        "ancient temples", "ancient history", "kailasa", "kailash temple", "ellora", "ajanta",
        "konark", "brihadeeswara", "hampi", "harappa", "mohenjo daro", "indus valley",
        "monolithic", "megalithic", "rock cut", "granite carving", "lost technology",
        "ancient architecture", "ancient engineering", "ancient india", "ancient egypt",
        "pyramid", "pyramids", "archaeological mystery", "archaeology", "ancient monument",
        "fictional ancient", "regional story", "regional folklore", "ancient lore", "cursed village",
        "kuldhara", "shrapit gaon", "lok katha", "lokkatha", "regional accent",
        "प्राचीन मंदिर", "मंदिर वास्तुकला", "कैलाश मंदिर", "एलोरा", "हम्पी", "सिंधु घाटी", "प्राचीन सभ्यता",
        "शापित गांव", "प्राचीन रहस्य", "कुलधरा", "लोककथा"
    ]
    for kw in ancient_kw:
        if re.search(r"\b" + re.escape(kw) + r"\b", t) or kw in t:
            return GENRES["ancient_history"]

    # 4. Mythology & Epics (e.g. "shiv", "krishna", "ram", "mahabharat", "vedic")
    myth_kw = [
        "shiv", "shiva", "ram", "rama", "krishna", "hanuman", "mahabharat", "ramayan",
        "puran", "vedic", "deva", "asura", "avatar", "bhagwan", "goddess", "mandir",
        "temple", "epic", "legend", "dharma", "yoddha", "brahma", "vishnu", "ganesha",
        "arjun", "karna", "parshuram", "mahadev", "ganpati", "kali maa", "mythology",
        "शिव", "राम", "कृष्ण", "हनुमान", "महाभारत", "रामायण", "पुराण", "वेद", "देवता", "मंदिर", "धर्म"
    ]
    for kw in myth_kw:
        if re.search(r"\b" + re.escape(kw) + r"\b", t) or kw in t:
            return GENRES["mythology"]

    # 4. Sci-Fi & Cyberpunk (e.g. "space", "alien", "robot", "cyberpunk", "future", "ai")
    scifi_kw = [
        "scifi", "sci-fi", "cyberpunk", "robot", "robots", "alien", "aliens",
        "matrix", "android", "cyborg", "spaceship", "starship", "dystopia",
        "artificial intelligence", "singularity", "time travel", "multiverse",
        "रोबोट", "अंतरिक्ष यान", "भविष्य काल", "एलियन", "साइंस फिक्शन"
    ]
    for kw in scifi_kw:
        if re.search(r"\b" + re.escape(kw) + r"\b", t) or kw in t:
            return GENRES["scifi"]

    # 5. Motivational & Success (e.g. "discipline", "never give up", "success", "habits")
    mot_kw = [
        "motivation", "motivational", "inspire", "inspirational", "success", "habit",
        "habits", "discipline", "never give up", "growth", "struggle", "life lesson",
        "parable", "mindset", "overcome", "prerna", "safar", "hard work", "mindset",
        "fitness", "gym", "workout", "achieve", "millionaire",
        "प्रेरणा", "सफलता", "मेहनत", "अनुशासन", "सीख", "कहानी सीख"
    ]
    for kw in mot_kw:
        if re.search(r"\b" + re.escape(kw) + r"\b", t) or kw in t:
            return GENRES["motivational"]

    # 6. Comedy & Satire (e.g. "funny", "comedy", "joke", "prank", "laugh")
    com_kw = [
        "comedy", "funny", "humor", "humour", "joke", "jokes", "prank", "laugh",
        "satire", "hilarious", "hasya", "mazaak", "chutkule", "slapstick", "parody",
        "meme", "memes", "fun", "mischief",
        "कॉमेडी", "मज़ाक", "हँसी", "चुटकुले", "हास्य"
    ]
    for kw in com_kw:
        if re.search(r"\b" + re.escape(kw) + r"\b", t) or kw in t:
            return GENRES["comedy"]

    # 7. Drama & Emotional (e.g. "love", "emotional", "family", "friendship", "sacrifice")
    drama_kw = [
        "emotional", "heartfelt", "tear", "tears", "sacrifice",
        "mother", "father", "family", "friend", "friendship", "maa", "baap",
        "dosti", "pyaar", "dil", "prem", "rishte", "sad story", "love story",
        "romance", "romantic",
        "प्यार", "दोस्ती", "माँ", "पिता", "परिवार", "रिश्ते", "भावुक", "दर्द", "आंसू"
    ]
    for kw in drama_kw:
        if re.search(r"\b" + re.escape(kw) + r"\b", t) or kw in t:
            return GENRES["drama"]

    # 8. Horror & Suspense (e.g. "horror", "scary", "ghost", "haunted", "bhoot", "chudail")
    horror_kw = [
        "horror", "scary", "bhoot", "chudail", "haunted", "ghost", "creepy", "dread",
        "daravna", "aatma", "graveyard", "shamshan", "dayan", "khooni", "shaitan",
        "spooky", "monster", "witchcraft", "evil",
        "भूत", "भूतिया", "चुड़ैल", "डरावना", "आत्मा", "शमशान", "दहशत", "खौफ", "डायन"
    ]
    for kw in horror_kw:
        if re.search(r"\b" + re.escape(kw) + r"\b", t) or kw in t:
            return GENRES["horror"]

    # 9. Explainer & Deep Dive Breakdown (e.g. "how it works", "explainer", "breakdown", "why does", "case study")
    explainer_kw = [
        "explainer", "how it works", "how do", "how does", "why does", "why do",
        "breakdown", "deep dive", "mechanism", "science breakdown",
        "tech breakdown", "infotainment", "psychology of", "economics of",
        "analysis", "how to understand", "explained in hindi", "video explainer",
        "एक्सप्लेनर", "कैसे काम करता है", "विश्लेषण", "कारण", "समझिए"
    ]
    for kw in explainer_kw:
        if re.search(r"\b" + re.escape(kw) + r"\b", t) or kw in t:
            return GENRES["explainer"]

    # 10. Documentary & Educational (e.g. "documentary", "history", "facts", "mystery of", "discovery")
    doc_kw = [
        "documentary", "docu", "fact", "facts", "history", "historic", "science",
        "discovery", "mystery of", "secret of", "explained", "case study", "biography",
        "ocean", "earth", "black hole", "tathya", "itihaas", "khoj", "planet",
        "educational", "education", "tutorial", "learn", "lesson", "guide",
        "how to", "how it works", "physics", "chemistry", "biology", "astronomy",
        "knowledge", "curiosity", "nature", "inventions", "origins",
        "इतिहास", "तथ्य", "खोज", "रहस्य", "विज्ञान", "डॉक्यूमेंट्री", "शिक्षा"
    ]
    for kw in doc_kw:
        if re.search(r"\b" + re.escape(kw) + r"\b", t) or kw in t:
            return GENRES["documentary"]

    # If general "ai" alone was passed without other sci-fi keywords
    if re.search(r"\bai\b", t):
        return GENRES["scifi"]

    # Fallback heuristic: inquisitions / questions default to documentary, otherwise horror
    if re.search(r"\b(why|how|what|who|where|when|guide|concept)\b", t):
        return GENRES["documentary"]

    return GENRES["horror"]


def movements_for(scene_count: int, movements_map: dict[str, Movement] | None = None) -> list[Movement]:
    m_map = movements_map or MOVEMENTS
    for threshold, keys in MOVEMENT_SETS:
        if scene_count >= threshold:
            return [m_map[key] for key in keys if key in m_map]
    return [m_map[key] for key in MOVEMENT_SETS[-1][1] if key in m_map]


def allocate(scene_count: int, movements: list[Movement]) -> list[int]:
    """Split scenes across movements by proportion, with one scene minimum each.

    Uses largest-remainder so the counts always total exactly, instead of drifting
    a scene or two on rounding.
    """
    count = len(movements)
    if scene_count <= count:
        return [1] * count

    spare = scene_count - count
    total_weight = sum(m.weight for m in movements) or 1.0
    exact = [spare * m.weight / total_weight for m in movements]
    share = [int(value) for value in exact]

    remainder = spare - sum(share)
    order = sorted(range(count), key=lambda i: -(exact[i] - share[i]))
    for i in range(remainder):
        share[order[i]] += 1

    return [1 + extra for extra in share]


def duty_for(movement: Movement, step: int, total: int, scene_count: int = 4) -> str:
    """The instruction for one scene, sharpened when a movement spans several or for short form."""
    if scene_count <= 5:
        short_duties = {
            "hook": "Introduce protagonist/subject and the terrifying catalyst immediately in one breath.",
            "inciting": "The point of no return: the anomaly escalates directly and traps the subject.",
            "rising": "High stakes complication: rapid escalation with no escape.",
            "climax": "Direct confrontation or horrific revelation at peak intensity.",
            "resolution": "Definitive conclusion: reveal the final fate/twist and bring total closure to the story.",
            "button": "Chilling final punchline that seals the story's fate.",
        }
        if movement.key in short_duties:
            return short_duties[movement.key]

    if total <= 1:
        return movement.duty
    if movement.key == "rising":
        return (
            f"{movement.duty} This is complication {step} of {total}, and it must "
            f"be worse than the one before it."
        )
    # Say "continue", never "Part N of M": calling a scene a part invites the
    # writer to treat it as a separate piece, which is how one story becomes a
    # row of disconnected fragments.
    return (
        f"{movement.duty} Step {step} of {total} inside this movement; continue the "
        f"same story from the previous scene, do not restart it and do not conclude it."
    )


def story_plan(scene_count: int, genre_profile: GenreProfile | None = None) -> list[tuple[Movement, str, str]]:
    """One (movement, beat, duty) per scene, in story order.

    The runtime is apportioned so the piece is complete at any length: the same
    movements, scaled to the time available.
    """
    movements_map = genre_profile.movements if genre_profile else MOVEMENTS
    rising_beats = genre_profile.rising_beats if genre_profile else RISING_BEATS
    movements = movements_for(scene_count, movements_map)
    counts = allocate(scene_count, movements)

    plan: list[tuple[Movement, str, str]] = []
    rising_index = 0

    for movement, count in zip(movements, counts):
        for step in range(1, count + 1):
            if movement.key == "rising":
                beat = rising_beats[rising_index % len(rising_beats)]
                rising_index += 1
            else:
                beat = movement.beat
            plan.append((movement, beat, duty_for(movement, step, count, scene_count)))

    return plan


def seed_for(*parts: object) -> int:
    """Stable seed so the same request always yields the same plan."""
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


# --------------------------------------------------------------------------
# Video Format & Framing Presets (Reels, Shorts, Landscape, Square, etc.)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class VideoFormat:
    key: str
    name: str
    aspect_ratio: str  # e.g. "9:16", "16:9", "1:1", "4:5", "21:9", "4:3"
    width: int
    height: int
    fps: int = 30
    description: str = ""
    aliases: tuple[str, ...] = ()
    sd_width: int = 512
    sd_height: int = 512
    caption_scale: float = 1.0
    bottom_margin: float = 0.08  # Fraction of frame height for bottom caption offset
    prompt_suffix: str = "cinematic widescreen 16:9"

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "name": self.name,
            "aspect_ratio": self.aspect_ratio,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "description": self.description,
            "aliases": list(self.aliases),
            "sd_width": self.sd_width,
            "sd_height": self.sd_height,
            "caption_scale": self.caption_scale,
            "bottom_margin": self.bottom_margin,
            "prompt_suffix": self.prompt_suffix,
        }


VIDEO_FORMATS: dict[str, VideoFormat] = {
    "reel": VideoFormat(
        key="reel",
        name="Vertical Reel / Short",
        aspect_ratio="9:16",
        width=1080,
        height=1920,
        fps=30,
        description="YouTube Shorts, Instagram Reels, TikTok vertical 9:16 (1080x1920)",
        aliases=("shorts", "short", "tiktok", "vertical", "9:16", "story", "stories", "reels"),
        sd_width=432,
        sd_height=768,
        caption_scale=1.15,
        bottom_margin=0.28,  # Raised higher up to clear social overlays and sit in optimal viewing line
        prompt_suffix="cinematic vertical 9:16 composition, full vertical view",
    ),
    "landscape": VideoFormat(
        key="landscape",
        name="Widescreen Landscape",
        aspect_ratio="16:9",
        width=1920,
        height=1080,
        fps=30,
        description="Standard YouTube widescreen, Desktop, TV 16:9 (1920x1080)",
        aliases=("youtube", "widescreen", "horizontal", "16:9", "video", "standard", "default"),
        sd_width=768,
        sd_height=432,
        caption_scale=1.0,
        bottom_margin=0.14,
        prompt_suffix="cinematic widescreen 16:9 composition, wide landscape framing",
    ),
    "square": VideoFormat(
        key="square",
        name="Square Feed Post",
        aspect_ratio="1:1",
        width=1080,
        height=1080,
        fps=30,
        description="Square feed post for Instagram, Facebook, LinkedIn 1:1 (1080x1080)",
        aliases=("feed", "1:1", "post", "square_post", "instagram_feed"),
        sd_width=512,
        sd_height=512,
        caption_scale=1.0,
        bottom_margin=0.18,
        prompt_suffix="cinematic square 1:1 composition, balanced central framing",
    ),
    "portrait": VideoFormat(
        key="portrait",
        name="Instagram Portrait Post",
        aspect_ratio="4:5",
        width=1080,
        height=1350,
        fps=30,
        description="Vertical portrait feed post for Instagram 4:5 (1080x1350)",
        aliases=("4:5", "insta_portrait", "social_portrait", "feed_portrait"),
        sd_width=464,
        sd_height=576,
        caption_scale=1.05,
        bottom_margin=0.22,
        prompt_suffix="cinematic portrait 4:5 composition, vertical framed shot",
    ),
    "cinema": VideoFormat(
        key="cinema",
        name="Cinematic Ultrawide",
        aspect_ratio="21:9",
        width=2560,
        height=1080,
        fps=30,
        description="Cinematic anamorphic ultrawide 21:9 (2560x1080)",
        aliases=("21:9", "ultrawide", "cinematic", "anamorphic", "movie"),
        sd_width=896,
        sd_height=384,
        caption_scale=0.95,
        bottom_margin=0.12,
        prompt_suffix="cinematic ultrawide 21:9 anamorphic composition, panoramic scope",
    ),
    "classic": VideoFormat(
        key="classic",
        name="Classic Retro TV",
        aspect_ratio="4:3",
        width=1440,
        height=1080,
        fps=30,
        description="Classic vintage / retro television 4:3 (1440x1080)",
        aliases=("4:3", "retro", "tv", "vintage", "standard_tv"),
        sd_width=640,
        sd_height=480,
        caption_scale=1.0,
        bottom_margin=0.14,
        prompt_suffix="cinematic 4:3 vintage composition, retro broadcast frame",
    ),
}


def get_format(name_or_alias: str | None) -> VideoFormat:
    """Look up a VideoFormat by key or alias (case-insensitive), defaulting to landscape."""
    if not name_or_alias:
        return VIDEO_FORMATS["landscape"]
    cleaned = str(name_or_alias).strip().lower().replace(" ", "_").replace("-", "_")
    if cleaned in VIDEO_FORMATS:
        return VIDEO_FORMATS[cleaned]
    for fmt in VIDEO_FORMATS.values():
        if (
            cleaned in fmt.aliases
            or cleaned == fmt.aspect_ratio
            or cleaned.replace(":", "_") == fmt.aspect_ratio.replace(":", "_")
        ):
            return fmt
    return VIDEO_FORMATS["landscape"]


def list_formats() -> list[dict]:
    """Return all available video format presets as dictionaries."""
    return [fmt.to_dict() for fmt in VIDEO_FORMATS.values()]


def resolve_dimensions(
    format_key: str | None = None,
    width: int | None = None,
    height: int | None = None,
    fps: int | None = None,
    aspect_ratio: str | None = None,
) -> VideoFormat:
    """Resolve full video framing parameters from format preset or custom width/height/fps."""
    lookup_key = format_key or aspect_ratio
    base_fmt = get_format(lookup_key)

    resolved_w = int(width) if width is not None and width > 0 else base_fmt.width
    resolved_h = int(height) if height is not None and height > 0 else base_fmt.height
    resolved_fps = int(fps) if fps is not None and fps > 0 else base_fmt.fps

    # If custom width and height were passed, compute custom aspect ratio & sd dimensions
    if (width is not None and width > 0) or (height is not None and height > 0):
        import math
        divisor = math.gcd(resolved_w, resolved_h)
        simp_w = resolved_w // divisor
        simp_h = resolved_h // divisor
        ratio_str = f"{simp_w}:{simp_h}"
        if ratio_str not in ("16:9", "9:16", "1:1", "4:5", "21:9", "4:3"):
            ratio_str = f"{resolved_w}:{resolved_h}"

        # Calculate matching SD 1.5 dimensions (~512x512 total pixels, multiple of 8)
        ar = resolved_w / resolved_h
        target_pixels = 512 * 512
        sd_h = int(round(math.sqrt(target_pixels / ar) / 8.0) * 8)
        sd_w = int(round((sd_h * ar) / 8.0) * 8)
        sd_w = max(256, min(1024, sd_w))
        sd_h = max(256, min(1024, sd_h))

        cap_scale = max(0.6, min(1.8, round(resolved_w / 1080.0, 2)))

        return VideoFormat(
            key="custom",
            name=f"Custom ({resolved_w}x{resolved_h})",
            aspect_ratio=ratio_str,
            width=resolved_w,
            height=resolved_h,
            fps=resolved_fps,
            description=f"Custom dimensions: {resolved_w}x{resolved_h} @ {resolved_fps}fps ({ratio_str})",
            sd_width=sd_w,
            sd_height=sd_h,
            caption_scale=cap_scale,
            bottom_margin=0.28 if resolved_h > resolved_w else 0.14,
            prompt_suffix=f"cinematic composition, {ratio_str} frame",
        )

    if resolved_fps != base_fmt.fps:
        return VideoFormat(
            key=base_fmt.key,
            name=base_fmt.name,
            aspect_ratio=base_fmt.aspect_ratio,
            width=base_fmt.width,
            height=base_fmt.height,
            fps=resolved_fps,
            description=base_fmt.description,
            aliases=base_fmt.aliases,
            sd_width=base_fmt.sd_width,
            sd_height=base_fmt.sd_height,
            caption_scale=base_fmt.caption_scale,
            bottom_margin=base_fmt.bottom_margin,
            prompt_suffix=base_fmt.prompt_suffix,
        )

    return base_fmt


# --------------------------------------------------------------------------

# Narration pacing. Hindi narration at speed 0.94 measured ~115 wpm.
DEFAULT_WPM = 115

# Target seconds per scene. Longer scenes mean fewer headline slots to write and
# a calmer pace.
DEFAULT_SCENE_SECONDS = 24


def plan(
    topic: str,
    minutes: float = 1.0,
    *,
    seconds: float | None = None,
    scenes: int | None = None,
    genre: str | None = None,
    theme: str | dict | None = None,
    video_format: str | None = None,
    width: int | None = None,
    height: int | None = None,
    fps: int | None = None,
    aspect_ratio: str | None = None,
    wpm: int = DEFAULT_WPM,
    scene_seconds: int = DEFAULT_SCENE_SECONDS,
    shots_per_scene: int = 2,
    language: str = "hi",
    with_characters: bool = False,
    voice: str | None = None,
) -> dict:
    """Build a full scene plan: structure complete, narration left blank.

    Returns a dict with `topic`, `target`, and `scenes` where each scene already
    has its shots, effects, ambience and sound cues. Only `headline` and the
    narration text are placeholders for the writer to fill.
    """
    if seconds is not None and seconds > 0:
        total_seconds = max(10.0, float(seconds))
        minutes = total_seconds / 60.0
    else:
        minutes = max(0.15, float(minutes))
        total_seconds = minutes * 60.0

    total_words = int(round((total_seconds / 60.0) * wpm))

    # Allow explicit override of scene/image count via `scenes` parameter
    if scenes is not None and int(scenes) > 0:
        max_practical = max(1, int(total_seconds / 2.5))
        if int(scenes) > max_practical:
            print(f"note: {scenes} scenes for a {total_seconds:.0f}s video results in under 2.5s per scene; clamping to {max_practical} scenes.", file=sys.stderr)
            scene_count = max_practical
        else:
            scene_count = max(MIN_SCENES, int(scenes))
    else:
        # Never drop below the scene count a whole story needs, even if that makes
        # scenes shorter than the target. A short video is a short story, not a
        # fragment of one.
        scene_count = max(MIN_SCENES, int(round(total_seconds / max(1.0, float(scene_seconds)))))

    words_per_scene = max(4, int(round(total_words / scene_count)))

    genre_profile = detect_genre(topic, genre)
    beats_map = genre_profile.beats
    theme_profile = resolve_theme(theme, genre=genre_profile.key, topic=topic)
    format_profile = resolve_dimensions(
        format_key=video_format,
        width=width,
        height=height,
        fps=fps,
        aspect_ratio=aspect_ratio,
    )

    rng = random.Random(seed_for(topic, minutes, wpm, shots_per_scene))
    story = story_plan(scene_count, genre_profile)

    scenes: list[dict] = []
    for index in range(scene_count):
        movement, beat_name, duty = story[index]
        beat = beats_map.get(beat_name, BEATS.get(beat_name, BEATS["establish"]))
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
        audio_seed = seed_for(topic, minutes, genre_profile.key, "scene_audio")
        ambience_spec, cues = resolve_dynamic_audio(
            genre_key=genre_profile.key,
            movement_key=movement.key,
            beat=beat,
            scene_index=index,
            scene_count=scene_count,
            seed=audio_seed,
            estimated_seconds=estimated,
            tension=tension,
        )

        scenes.append(
            {
                "id": f"scene-{index + 1:03d}",
                "beat": beat.name,
                "_story": movement.key,
                "_storyNote": duty,
                "headline": "",
                "subtext": "",
                "bullets": [],
                "headlinePosition": "center",
                "shots": shots,
                "ambience": ambience_spec,
                "soundCues": cues,
                "background": {"type": "color", "color": theme_profile.bg_color or genre_profile.bg_color},
                "cast": cast_for(index, beat, progress) if with_characters else [],
                "acting": {
                    "emotion": beat.emotion,
                    "intensity": round(min(1.0, beat.acting_intensity * (0.75 + progress * 0.5)), 2),
                    "head": FACINGS[index % len(FACINGS)],
                    "eyes": FACINGS[index % len(FACINGS)],
                    "gesture": beat.gesture,
                },
                "padBefore": 0.0,
                "padAfter": 0.04,
                "holdSeconds": 3.0,
                "lines": [],
                "_targetWords": words_per_scene,
            }
        )

    # Master continuous baseline ambience across the entire video
    global_amb_pool = GENRE_AUDIO_POOLS.get(genre_profile.key, GENRE_AUDIO_POOLS["horror"])
    global_amb_options = global_amb_pool.get("ambience_setup") or ["amb_wind", "amb_drone"]
    global_amb_src = rng.choice(global_amb_options)
    global_ambience = {"src": global_amb_src, "volume": 0.20}

    # The length contract, published as one object. The story bible, the chunk
    # writer and the validator all read the runtime from here rather than
    # re-deriving it, so "how long is this going to be" has exactly one answer
    # for the whole run.
    movement_span: dict[str, dict] = {}
    for index, (movement, _beat_name, _duty) in enumerate(story):
        span = movement_span.setdefault(movement.key, {"scenes": [], "weight": movement.weight})
        span["scenes"].append(index + 1)

    blueprint = {
        "minutes": minutes,
        "seconds": round(total_seconds, 1),
        "sceneCount": scene_count,
        "totalWords": total_words,
        "wordsPerScene": words_per_scene,
        "wpm": wpm,
        "movements": [
            {
                "key": key,
                "weight": span["weight"],
                "scenes": span["scenes"],
                "sceneCount": len(span["scenes"]),
                "words": words_per_scene * len(span["scenes"]),
                "range": [span["scenes"][0], span["scenes"][-1]],
            }
            for key, span in movement_span.items()
        ],
    }

    return {
        "topic": topic,
        "language": language,
        "genre": genre_profile.key,
        "genreProfile": genre_profile,
        "ambience": global_ambience,
        "theme": theme_profile.to_dict(),
        "themeProfile": theme_profile,
        "format": format_profile.key,
        "formatProfile": format_profile,
        "width": format_profile.width,
        "height": format_profile.height,
        "aspectRatio": format_profile.aspect_ratio,
        "fps": format_profile.fps,
        "voice": voice or genre_profile.voice_for_language(language),
        "characters": DEFAULT_CHARACTERS if with_characters else {},
        "blueprint": blueprint,
        "target": {
            "minutes": minutes,
            "seconds": round(total_seconds, 1),
            "wpm": wpm,
            "totalWords": total_words,
            "scenes": scene_count,
            "wordsPerScene": words_per_scene,
            "shots": sum(len(s["shots"]) for s in scenes),
            "castScenes": sum(1 for s in scenes if s["cast"]),
            "format": format_profile.key,
            "aspectRatio": format_profile.aspect_ratio,
            "dimensions": f"{format_profile.width}x{format_profile.height}",
            "fps": format_profile.fps,
        },
        "scenes": scenes,
    }


def describe(planned: dict) -> str:
    """Human-readable summary of a plan, for --dry-run."""
    target = planned["target"]
    genre_key = planned.get("genre", "horror")
    genre_name = GENRES.get(genre_key, GENRES["horror"]).name
    theme_info = planned.get("theme", {})
    theme_name = theme_info.get("name", theme_info.get("key", "Default"))
    theme_accent = theme_info.get("accent", "#c9524a")
    fmt = planned.get("formatProfile") or get_format(planned.get("format"))
    voice_str = planned.get("voice", "")
    lines = [
        f"topic        : {planned['topic']}",
        f"genre        : {genre_name} ({genre_key})",
        f"format       : {fmt.name} [{fmt.aspect_ratio}, {fmt.width}x{fmt.height} @ {fmt.fps}fps]",
        f"theme        : {theme_name} [accent: {theme_accent}]",
        f"voice        : {voice_str}" if voice_str else "",
        f"target       : {target['minutes']:g} min ({target.get('seconds', round(target['minutes'] * 60))}s)  ~{target['totalWords']} words  "
        f"@ {target['wpm']} wpm",
        f"structure    : {target['scenes']} scenes, {target['shots']} shots "
        f"({target['wordsPerScene']} words each)",
        f"cast         : {target.get('castScenes', 0)} of {target['scenes']} scenes "
        f"({', '.join(planned.get('characters', {})) or 'none'})",
        "",
        f"{'#':>3} {'story':<11} {'beat':<10} {'words':>6} {'emotion':<9} visuals",
        "-" * 84,
    ]
    lines = [line for line in lines if line != ""]
    for index, scene in enumerate(planned["scenes"], start=1):
        kinds = ", ".join(shot["visual"]["type"] for shot in scene["shots"])
        acting = scene.get("acting", {})
        lines.append(
            f"{index:>3} {scene.get('_story', ''):<11} {scene['beat']:<10} "
            f"{scene['_targetWords']:>6} {acting.get('emotion', ''):<9} {kinds}"
        )

    cues = sum(len(s["soundCues"]) for s in planned["scenes"])
    ambiences = sorted({s["ambience"]["src"] for s in planned["scenes"]})
    sfx_types = sorted({c["sfx"] for s in planned["scenes"] for c in s["soundCues"]})

    # Report the story shape, because completeness is the thing that used to fail.
    order: list[str] = []
    for scene in planned["scenes"]:
        key = scene.get("_story", "")
        if key and (not order or order[-1] != key):
            order.append(key)
    counts: dict[str, int] = {}
    for scene in planned["scenes"]:
        key = scene.get("_story", "")
        counts[key] = counts.get(key, 0) + 1

    lines += [
        "",
        f"story arc    : {' -> '.join(order)}",
        f"scene split  : " + ", ".join(f"{k} {v}" for k, v in counts.items()),
        f"sound cues   : {cues} ({', '.join(sfx_types) if sfx_types else 'none'})",
        f"ambience     : {', '.join(ambiences)}",
    ]
    return "\n".join(lines)
