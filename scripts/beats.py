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
    (0, ["hook", "inciting", "climax", "resolution"]),
]

# Below this, a story cannot be whole, so the planner refuses to go shorter.
MIN_SCENES = 4

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
    "anime": {
        "ambience_setup": ["amb_tension", "amb_wind", "amb_cyber"],
        "ambience_rising": ["amb_tension", "amb_drone", "amb_cyber"],
        "ambience_climax": ["amb_tension", "amb_heartbeat", "amb_cyber"],
        "ambience_resolution": ["amb_uplift", "amb_wind", "amb_warm"],
        "sfx_setup": ["sfx_whoosh", "sfx_riser", "sfx_glitch"],
        "sfx_rising": ["sfx_whoosh", "sfx_riser", "sfx_cinematic_hit"],
        "sfx_climax": ["sfx_boom", "sfx_cinematic_hit", "sfx_impact"],
        "sfx_resolution": ["sfx_swell", "sfx_bell", "sfx_whoosh"],
    },
    "fantasy": {
        "ambience_setup": ["amb_ethereal", "amb_wind", "amb_warm"],
        "ambience_rising": ["amb_ethereal", "amb_warm", "amb_wind"],
        "ambience_climax": ["amb_ethereal", "amb_tension"],
        "ambience_resolution": ["amb_warm", "amb_wind", "amb_ethereal"],
        "sfx_setup": ["sfx_sparkle", "sfx_chime", "sfx_bell"],
        "sfx_rising": ["sfx_sparkle", "sfx_whoosh", "sfx_riser"],
        "sfx_climax": ["sfx_swell", "sfx_sparkle", "sfx_cinematic_hit", "sfx_boom"],
        "sfx_resolution": ["sfx_chime", "sfx_bell", "sfx_sparkle"],
    },
    "scifi": {
        "ambience_setup": ["amb_cyber", "amb_hum"],
        "ambience_rising": ["amb_cyber", "amb_tension", "amb_drone"],
        "ambience_climax": ["amb_cyber", "amb_tension"],
        "ambience_resolution": ["amb_hum", "amb_cyber", "amb_wind"],
        "sfx_setup": ["sfx_glitch", "sfx_whoosh", "sfx_clock_tick"],
        "sfx_rising": ["sfx_glitch", "sfx_riser", "sfx_clock_tick"],
        "sfx_climax": ["sfx_cinematic_hit", "sfx_boom", "sfx_glitch"],
        "sfx_resolution": ["sfx_chime", "sfx_glitch", "sfx_whoosh"],
    },
    "horror": {
        "ambience_setup": ["amb_wind", "amb_drone"],
        "ambience_rising": ["amb_drone", "amb_rain", "amb_tension"],
        "ambience_climax": ["amb_heartbeat", "amb_drone"],
        "ambience_resolution": ["amb_wind", "amb_drone", "amb_rain"],
        "sfx_setup": ["sfx_whisper", "sfx_clock_tick"],
        "sfx_rising": ["sfx_riser", "sfx_whisper", "sfx_clock_tick"],
        "sfx_climax": ["sfx_stinger", "sfx_impact", "sfx_boom"],
        "sfx_resolution": ["sfx_bell", "sfx_impact", "sfx_whisper"],
    },
    "mythology": {
        "ambience_setup": ["amb_temple", "amb_ethereal"],
        "ambience_rising": ["amb_temple", "amb_drone", "amb_wind"],
        "ambience_climax": ["amb_temple", "amb_drone"],
        "ambience_resolution": ["amb_temple", "amb_warm", "amb_wind"],
        "sfx_setup": ["sfx_bell", "sfx_horn", "sfx_swell"],
        "sfx_rising": ["sfx_riser", "sfx_horn", "sfx_whoosh"],
        "sfx_climax": ["sfx_cinematic_hit", "sfx_horn", "sfx_boom"],
        "sfx_resolution": ["sfx_bell", "sfx_swell", "sfx_chime"],
    },
    "motivational": {
        "ambience_setup": ["amb_warm", "amb_wind"],
        "ambience_rising": ["amb_tension", "amb_warm", "amb_drone"],
        "ambience_climax": ["amb_uplift", "amb_warm"],
        "ambience_resolution": ["amb_uplift", "amb_warm", "amb_wind"],
        "sfx_setup": ["sfx_clock_tick", "sfx_swell"],
        "sfx_rising": ["sfx_riser", "sfx_whoosh", "sfx_clock_tick"],
        "sfx_climax": ["sfx_cinematic_hit", "sfx_swell", "sfx_boom"],
        "sfx_resolution": ["sfx_chime", "sfx_swell", "sfx_bell"],
    },
    "drama": {
        "ambience_setup": ["amb_warm", "amb_rain"],
        "ambience_rising": ["amb_tension", "amb_rain", "amb_drone"],
        "ambience_climax": ["amb_tension", "amb_drone"],
        "ambience_resolution": ["amb_warm", "amb_rain", "amb_wind"],
        "sfx_setup": ["sfx_bell", "sfx_clock_tick"],
        "sfx_rising": ["sfx_riser", "sfx_whisper", "sfx_clock_tick"],
        "sfx_climax": ["sfx_impact", "sfx_swell", "sfx_cinematic_hit"],
        "sfx_resolution": ["sfx_swell", "sfx_bell", "sfx_whisper"],
    },
    "comedy": {
        "ambience_setup": ["amb_uplift", "amb_warm"],
        "ambience_rising": ["amb_hum", "amb_uplift"],
        "ambience_climax": ["amb_uplift", "amb_warm"],
        "ambience_resolution": ["amb_uplift", "amb_warm"],
        "sfx_setup": ["sfx_pop", "sfx_whoosh"],
        "sfx_rising": ["sfx_pop", "sfx_whoosh", "sfx_chime"],
        "sfx_climax": ["sfx_pop", "sfx_sparkle", "sfx_cinematic_hit"],
        "sfx_resolution": ["sfx_pop", "sfx_chime", "sfx_sparkle"],
    },
    "documentary": {
        "ambience_setup": ["amb_wind", "amb_hum"],
        "ambience_rising": ["amb_tension", "amb_drone", "amb_wind"],
        "ambience_climax": ["amb_tension", "amb_drone"],
        "ambience_resolution": ["amb_wind", "amb_hum", "amb_warm"],
        "sfx_setup": ["sfx_clock_tick", "sfx_whoosh"],
        "sfx_rising": ["sfx_riser", "sfx_clock_tick", "sfx_whoosh"],
        "sfx_climax": ["sfx_cinematic_hit", "sfx_boom", "sfx_impact"],
        "sfx_resolution": ["sfx_bell", "sfx_chime", "sfx_clock_tick"],
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
    """Determine dynamic ambience bed and sound cues tailored to genre, movement, and video seed."""
    pool = GENRE_AUDIO_POOLS.get(genre_key, GENRE_AUDIO_POOLS["horror"])
    rng = random.Random(seed + scene_index * 37)

    # 1. Select dynamic Ambience
    if movement_key in ("hook", "setup"):
        amb_options = pool.get("ambience_setup") or ["amb_wind", "amb_drone"]
        default_vol = 0.36
    elif movement_key in ("inciting", "rising"):
        amb_options = pool.get("ambience_rising") or ["amb_drone", "amb_tension"]
        default_vol = 0.44
    elif movement_key == "climax":
        amb_options = pool.get("ambience_climax") or ["amb_heartbeat", "amb_tension"]
        default_vol = 0.52
    else:  # resolution, button
        amb_options = pool.get("ambience_resolution") or ["amb_wind", "amb_warm"]
        default_vol = 0.35

    chosen_amb = rng.choice(amb_options) if amb_options else beat.ambience[0]
    amb_volume = round(min(0.65, default_vol * (0.85 + tension * 0.3)), 2)

    # 2. Select dynamic Sound Cues
    cues_spec = beat.cues
    if not cues_spec:
        if movement_key == "climax":
            cues_spec = [(0.05, "sfx_cinematic_hit", 0.9), (0.6, "sfx_boom", 0.75)]
        elif movement_key in ("inciting", "rising"):
            cues_spec = [(0.35, "sfx_riser", 0.45)]
        elif movement_key in ("resolution", "button"):
            cues_spec = [(0.65, "sfx_bell", 0.6)]
        else:
            cues_spec = [(0.2, "sfx_whoosh", 0.35)]

    sound_cues: list[dict] = []
    for c_idx, (pos, default_sfx, base_vol) in enumerate(cues_spec):
        if movement_key == "climax":
            sfx_options = pool.get("sfx_climax") or ["sfx_cinematic_hit", "sfx_stinger", "sfx_boom"]
        elif movement_key in ("inciting", "rising"):
            sfx_options = pool.get("sfx_rising") or ["sfx_riser", "sfx_whoosh", "sfx_clock_tick"]
        elif movement_key in ("resolution", "button"):
            sfx_options = pool.get("sfx_resolution") or ["sfx_bell", "sfx_chime", "sfx_swell"]
        else:
            sfx_options = pool.get("sfx_setup") or ["sfx_chime", "sfx_sparkle", "sfx_whisper"]

        chosen_sfx = rng.choice(sfx_options) if sfx_options else default_sfx
        # For the very first hook/opening cue of scene 1, hit instantaneously at 0.0s for viral audio retention
        if scene_index == 0 and c_idx == 0:
            at_sec = 0.0
        else:
            at_sec = round(min(estimated_seconds * 0.92, max(0.2, estimated_seconds * pos)), 2)
        final_vol = round(min(1.0, base_vol * (0.75 + tension * 0.4)), 2)
        sound_cues.append({"at": at_sec, "sfx": chosen_sfx, "volume": final_vol})

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
        visual_style_suffix=", dynamic camera angle, vibrant celestial lighting, sharp cel-shaded outlines, 16:9 widescreen, 4k anime wallpaper.",
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
        visual_style_suffix=", storybook illustration style, highly detailed fantasy background, 16:9 widescreen, cinematic composition, breathtaking color palette.",
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
        visual_style_suffix=", cinematic sci-fi background, Unreal Engine 5 render style, 16:9 widescreen, dramatic volumetric lighting.",
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
        aliases=["documentary", "docu", "history", "historic", "science", "facts", "educational", "explainer", "biography", "case_study", "nature", "curiosity"],
        tone="engaging, authoritative, captivating, informative, and insightful",
        writer_instructions=(
            "You are writing narration for an engaging documentary and educational video. "
            "Present fascinating facts, historical depth, mysteries, and compelling insights with clear, immersive storytelling."
        ),
        system_prompt="You write narration for educational, historical, scientific, and investigative documentary videos. You present rich facts and gripping insights.",
        visual_style_prefix="Cinematic documentary photography and digital painting, National Geographic aesthetic, elegant historical realism: ",
        visual_style_suffix=", wide cinematic shot, museum quality lighting, 16:9 widescreen, rich natural colors, highly detailed.",
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
        visual_style_suffix=", masterpiece epic fantasy art, 16:9 widescreen, opulent divine lighting, sacred atmosphere.",
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
        visual_style_suffix=", cinematic 8k quality, motivational masterpiece, 16:9 widescreen, warm radiant glow.",
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
        visual_style_suffix=", cinematic 16:9 composition, touching depth of field, delicate pastel and warm tones.",
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
        visual_style_suffix=", clean vibrant aesthetic, 16:9 widescreen, cheerful energetic color palette.",
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


def _make_horror_genre() -> GenreProfile:
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
        visual_style_prefix="Cinematic horror atmosphere, eerie 2D digital painting, dark moody lighting, volumetric fog, deep shadows, ominous mystery: ",
        visual_style_suffix=", highly detailed dark background, cinematic composition, widescreen 16:9, dramatic lighting.",
        search_keywords_map={
            "forest": "dark mist forest haunted trees night",
            "house": "haunted house eerie mansion dark night",
            "grave": "cemetery graveyard tombstone foggy night",
            "corridor": "creepy dark hallway corridor flickering light",
            "candle": "single flickering candle dark gloomy room",
            "figure": "mysterious shadow figure dark silhouette sinister",
            "eyes": "glowing eerie eyes in pitch darkness",
            "rain": "dark rain storm lightning haunted night",
            "fog": "dense foggy mist ominous darkness",
            "void": "deep pitch black creeping shadow darkness",
        },
        default_keywords="haunted mansion dark mist graveyard horror landscape",
        theme={
            "accent": "#c9524a",
            "text": "#f2ece4",
            "muted": "#a89f96",
            "panel": "rgba(4, 4, 6, 0.65)",
            "headlineFont": "Georgia, 'Times New Roman', serif",
            "bodyFont": "'Segoe UI', system-ui, -apple-system, sans-serif",
        },
        movements=MOVEMENTS,
        beats=BEATS,
        rising_beats=RISING_BEATS,
        default_wpm=100,
        default_speed=0.88,
        bg_color="#04060a",
    )


GENRES: dict[str, GenreProfile] = {
    "anime": _make_anime_genre(),
    "fantasy": _make_fantasy_genre(),
    "scifi": _make_scifi_genre(),
    "documentary": _make_documentary_genre(),
    "mythology": _make_mythology_genre(),
    "motivational": _make_motivational_genre(),
    "drama": _make_drama_genre(),
    "comedy": _make_comedy_genre(),
    "horror": _make_horror_genre(),
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

    # 3. Mythology & Epics (e.g. "shiv", "krishna", "ram", "mahabharat", "temple", "vedic")
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

    # 9. Documentary & Educational (e.g. "documentary", "history", "facts", "mystery of", "discovery")
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
    return f"{movement.duty} Part {step} of {total}; continue this, do not restart it."


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
        bottom_margin=0.18,  # Raised up to avoid TikTok/Instagram UI overlay controls
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
        bottom_margin=0.08,
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
        bottom_margin=0.10,
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
        bottom_margin=0.12,
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
        bottom_margin=0.06,
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
        bottom_margin=0.08,
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
            bottom_margin=0.14 if resolved_h > resolved_w else 0.08,
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
        scene_count = max(MIN_SCENES, int(scenes))
    else:
        # Never drop below the scene count a whole story needs, even if that makes
        # scenes shorter than the target. A short video is a short story, not a
        # fragment of one.
        scene_count = max(MIN_SCENES, int(round(total_seconds / scene_seconds)))

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
                "padAfter": 0.55,
                "holdSeconds": 3.0,
                "lines": [],
                "_targetWords": words_per_scene,
            }
        )

    return {
        "topic": topic,
        "language": language,
        "genre": genre_profile.key,
        "genreProfile": genre_profile,
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
