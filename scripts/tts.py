#!/usr/bin/env python3
"""Turn ``script.json`` into narration audio, word timings and lip-sync data.

Voice engine: **Kokoro-82M** (Apache-2.0). It runs locally on CPU, so there is
no API key, no per-character cost and no licensing problem for monetised video.
Measured on the reference machine (Intel 120U, no GPU): ~0.56 realtime factor,
so roughly 17 minutes to synthesise a 30 minute narration.

Two problems had to be solved to get here:

1. **Word timings.** ``edge-tts`` reported a timestamp per word; Kokoro does not.
   So the text is split into sentences, each sentence is synthesised on its own
   (making its duration exact from the sample count), and that duration is then
   distributed across the sentence's words in proportion to their phoneme counts
   from ``misaki``. Timing is exact at sentence level and approximate from word
   to word, which is well within what burned-in captions need.

2. **Lip sync.** Rhubarb Lip Sync (MIT) analyses the generated audio and emits
   mouth shapes from the standard Hanna-Barbera set (A-H plus X). Those are the
   shapes the character rig draws.

Outputs:
  public/audio/<scene>[-<n>].wav    narration per line, 24 kHz mono PCM16
  public/audio/<scene>[-<n>].json   cache sidecar (hash, words, mouth cues)
  src/generated/project.json        normalised script + timeline for Remotion

Usage:
  .venv\\Scripts\\python.exe scripts\\tts.py             # build everything
  .venv\\Scripts\\python.exe scripts\\tts.py --force     # ignore the cache
  .venv\\Scripts\\python.exe scripts\\tts.py --voices    # list Kokoro voices
  .venv\\Scripts\\python.exe scripts\\tts.py --no-lipsync
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# Allow importing the sibling phonemise module when this file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from phonemise import Phonemiser  # noqa: E402
from make_assets import ensure_assets  # noqa: E402
from generate_images import load_env_file  # noqa: E402
from transliterate import deva_word_to_hinglish, deva_to_hinglish, is_devanagari, export_srt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PUBLIC = ROOT / "public"
AUDIO_DIR = PUBLIC / "audio"
GENERATED = ROOT / "src" / "generated"

# Load .env configuration if present
load_env_file(ROOT / ".env")

SAMPLE_RATE = 24000  # Kokoro's native rate

# Bumped whenever the prosody or mastering chain changes, so audio cached by an
# older chain is rebuilt instead of being silently reused.
AUDIO_PIPELINE_VERSION = 3

# How long to hold silence after a sentence, relative to the configured base
# pause. Gaps and pauses are restricted strictly to full stops and terminal marks
# (dandas, periods, exclamation, questions) to prevent unnatural pauses on minor punctuation.
PAUSE_SCALE = {
    ",": 0.0,
    ";": 0.0,
    ":": 0.0,
    "\u2014": 0.0,
    "-": 0.0,
    ".": 1.0,
    "\u0964": 1.0,  # Devanagari danda
    "\u0965": 1.2,  # Devanagari double danda
    "!": 1.0,
    "?": 1.0,
}

# Quotes and brackets that can follow the mark which actually ends a sentence.
TRAILING_MARKS = "\"'\u201d\u2019)]} \t\u00a0"

# Kokoro's output is clean but dry and uneven, so the narration is run through a
# small mastering chain before it is written out.
DEFAULT_MASTER = {
    "enabled": True,
    "highpassHz": 75.0,
    "presence": 0.3,
    "compress": True,
    "thresholdDbfs": -24.0,
    "ratio": 3.0,
    "targetLoudnessDbfs": -18.0,
    "peakCeilingDbfs": -1.5,
}

# Kokoro voice catalog across Hindi, English (US/UK), Japanese, Chinese, Spanish, French, Italian, Portuguese.
VOICE_CATALOG: dict[str, dict] = {
    # Hindi (hi)
    "hm_omega": {
        "id": "hm_omega", "name": "Aarav", "lang": "hi", "lang_name": "Hindi", "gender": "male",
        "style": "Deep, resonant & authoritative (ideal for horror, mythology, epics)",
        "aliases": ["hindi_male", "hindi_male_deep", "omega", "hindi_deep", "hindi_horror"],
    },
    "hm_psi": {
        "id": "hm_psi", "name": "Kabir", "lang": "hi", "lang_name": "Hindi", "gender": "male",
        "style": "Calm, natural & conversational (ideal for scifi, documentary, explainer)",
        "aliases": ["hindi_male_calm", "psi", "hindi_calm", "hindi_doc"],
    },
    "hf_alpha": {
        "id": "hf_alpha", "name": "Priya", "lang": "hi", "lang_name": "Hindi", "gender": "female",
        "style": "Warm, expressive & engaging (ideal for fantasy, drama, storytelling)",
        "aliases": ["hindi_female", "hindi_female_warm", "alpha", "hindi_warm", "hindi_fantasy"],
    },
    "hf_beta": {
        "id": "hf_beta", "name": "Ananya", "lang": "hi", "lang_name": "Hindi", "gender": "female",
        "style": "Soft, cheerful & pleasant (ideal for comedy, slice of life, children)",
        "aliases": ["hindi_female_soft", "beta", "hindi_soft", "hindi_comedy"],
    },
    # American English (en-us)
    "af_heart": {
        "id": "af_heart", "name": "Heart", "lang": "en-us", "lang_name": "English (US)", "gender": "female",
        "style": "Warm, rich, narrative (top-rated voice, ideal for storytelling, drama)",
        "aliases": ["heart", "english_female", "female_warm", "us_female"],
    },
    "af_bella": {
        "id": "af_bella", "name": "Bella", "lang": "en-us", "lang_name": "English (US)", "gender": "female",
        "style": "Bright, vibrant & energetic (ideal for anime, action, comedy)",
        "aliases": ["bella", "female_energetic", "anime", "energetic"],
    },
    "af_nicole": {
        "id": "af_nicole", "name": "Nicole", "lang": "en-us", "lang_name": "English (US)", "gender": "female",
        "style": "Soft, calm & soothing (ideal for meditation, gentle stories)",
        "aliases": ["nicole", "female_soft"],
    },
    "af_sarah": {
        "id": "af_sarah", "name": "Sarah", "lang": "en-us", "lang_name": "English (US)", "gender": "female",
        "style": "Clear, professional & crisp (ideal for explainer, tutorials)",
        "aliases": ["sarah", "female_clear"],
    },
    "af_sky": {
        "id": "af_sky", "name": "Sky", "lang": "en-us", "lang_name": "English (US)", "gender": "female",
        "style": "Light, youthful & dynamic",
        "aliases": ["sky"],
    },
    "af_alloy": {
        "id": "af_alloy", "name": "Alloy", "lang": "en-us", "lang_name": "English (US)", "gender": "female",
        "style": "Modern, balanced & neutral",
        "aliases": ["alloy"],
    },
    "af_aoede": {
        "id": "af_aoede", "name": "Aoede", "lang": "en-us", "lang_name": "English (US)", "gender": "female",
        "style": "Melodic, theatrical & lyrical",
        "aliases": ["aoede"],
    },
    "af_jessica": {
        "id": "af_jessica", "name": "Jessica", "lang": "en-us", "lang_name": "English (US)", "gender": "female",
        "style": "Conversational, natural & casual",
        "aliases": ["jessica"],
    },
    "af_kore": {
        "id": "af_kore", "name": "Kore", "lang": "en-us", "lang_name": "English (US)", "gender": "female",
        "style": "Balanced, expressive & clear",
        "aliases": ["kore"],
    },
    "af_nova": {
        "id": "af_nova", "name": "Nova", "lang": "en-us", "lang_name": "English (US)", "gender": "female",
        "style": "Documentary, narrative & engaging",
        "aliases": ["nova"],
    },
    "af_river": {
        "id": "af_river", "name": "River", "lang": "en-us", "lang_name": "English (US)", "gender": "female",
        "style": "Relaxed, easygoing & smooth",
        "aliases": ["river"],
    },
    "am_michael": {
        "id": "am_michael", "name": "Michael", "lang": "en-us", "lang_name": "English (US)", "gender": "male",
        "style": "Calm narration, balanced & reliable (default US male)",
        "aliases": ["michael", "english_male", "male_calm", "us_male"],
    },
    "am_onyx": {
        "id": "am_onyx", "name": "Onyx", "lang": "en-us", "lang_name": "English (US)", "gender": "male",
        "style": "Deep, authoritative, bass & intense (ideal for horror, movie trailers)",
        "aliases": ["onyx", "deep_male", "male_deep", "trailer"],
    },
    "am_adam": {
        "id": "am_adam", "name": "Adam", "lang": "en-us", "lang_name": "English (US)", "gender": "male",
        "style": "Direct, confident & energetic (ideal for motivational, sports, tech)",
        "aliases": ["adam", "motivational", "motivational_male"],
    },
    "am_echo": {
        "id": "am_echo", "name": "Echo", "lang": "en-us", "lang_name": "English (US)", "gender": "male",
        "style": "Resonant, clear & focused",
        "aliases": ["echo"],
    },
    "am_eric": {
        "id": "am_eric", "name": "Eric", "lang": "en-us", "lang_name": "English (US)", "gender": "male",
        "style": "Friendly, approachable & informative",
        "aliases": ["eric"],
    },
    "am_fenrir": {
        "id": "am_fenrir", "name": "Fenrir", "lang": "en-us", "lang_name": "English (US)", "gender": "male",
        "style": "Intense, dramatic & gritty (ideal for action, battles)",
        "aliases": ["fenrir", "dramatic", "action_male"],
    },
    "am_liam": {
        "id": "am_liam", "name": "Liam", "lang": "en-us", "lang_name": "English (US)", "gender": "male",
        "style": "Youthful, modern & conversational",
        "aliases": ["liam"],
    },
    "am_puck": {
        "id": "am_puck", "name": "Puck", "lang": "en-us", "lang_name": "English (US)", "gender": "male",
        "style": "Playful, quirky & animated (ideal for comedy, satire)",
        "aliases": ["puck", "comedy_male", "comedy"],
    },
    "am_santa": {
        "id": "am_santa", "name": "Santa", "lang": "en-us", "lang_name": "English (US)", "gender": "male",
        "style": "Older, warm & grandfatherly",
        "aliases": ["santa", "elderly_male"],
    },
    # British English (en-gb)
    "bm_george": {
        "id": "bm_george", "name": "George", "lang": "en-gb", "lang_name": "English (UK)", "gender": "male",
        "style": "Authoritative BBC documentary tone (ideal for history, science, nature)",
        "aliases": ["george", "documentary", "british_male", "bbc"],
    },
    "bm_daniel": {
        "id": "bm_daniel", "name": "Daniel", "lang": "en-gb", "lang_name": "English (UK)", "gender": "male",
        "style": "Deep, commanding & dramatic UK tone",
        "aliases": ["daniel", "british_deep"],
    },
    "bm_lewis": {
        "id": "bm_lewis", "name": "Lewis", "lang": "en-gb", "lang_name": "English (UK)", "gender": "male",
        "style": "Modern, crisp & articulate",
        "aliases": ["lewis"],
    },
    "bm_fable": {
        "id": "bm_fable", "name": "Fable", "lang": "en-gb", "lang_name": "English (UK)", "gender": "male",
        "style": "Theatrical storyteller & audiobook narrator",
        "aliases": ["fable", "storyteller"],
    },
    "bf_emma": {
        "id": "bf_emma", "name": "Emma", "lang": "en-gb", "lang_name": "English (UK)", "gender": "female",
        "style": "Warm, approachable & clear UK narration",
        "aliases": ["emma", "british_female"],
    },
    "bf_alice": {
        "id": "bf_alice", "name": "Alice", "lang": "en-gb", "lang_name": "English (UK)", "gender": "female",
        "style": "Classic British RP, refined & crystal clear",
        "aliases": ["alice", "british_refined"],
    },
    "bf_isabella": {
        "id": "bf_isabella", "name": "Isabella", "lang": "en-gb", "lang_name": "English (UK)", "gender": "female",
        "style": "Refined, elegant & poised",
        "aliases": ["isabella"],
    },
    "bf_lily": {
        "id": "bf_lily", "name": "Lily", "lang": "en-gb", "lang_name": "English (UK)", "gender": "female",
        "style": "Youthful, bright & cheerful UK tone",
        "aliases": ["lily"],
    },
    # Japanese (ja)
    "jf_alpha": {
        "id": "jf_alpha", "name": "Sakura", "lang": "ja", "lang_name": "Japanese", "gender": "female",
        "style": "Clear, natural standard Japanese",
        "aliases": ["japanese_female", "jf_alpha"],
    },
    "jf_gongitsune": {
        "id": "jf_gongitsune", "name": "Gongitsune", "lang": "ja", "lang_name": "Japanese", "gender": "female",
        "style": "Expressive Japanese fairytale & storytelling",
        "aliases": ["gongitsune"],
    },
    "jf_nezumi": {
        "id": "jf_nezumi", "name": "Nezumi", "lang": "ja", "lang_name": "Japanese", "gender": "female",
        "style": "Light, energetic anime character style",
        "aliases": ["anime_female", "nezumi"],
    },
    "jf_tebukuro": {
        "id": "jf_tebukuro", "name": "Tebukuro", "lang": "ja", "lang_name": "Japanese", "gender": "female",
        "style": "Soft & soothing Japanese",
        "aliases": ["tebukuro"],
    },
    "jm_kumo": {
        "id": "jm_kumo", "name": "Kumo", "lang": "ja", "lang_name": "Japanese", "gender": "male",
        "style": "Calm, natural anime & documentary narrator",
        "aliases": ["anime_male", "japanese_male", "kumo"],
    },
    # Mandarin Chinese (cmn)
    "zf_xiaobei": {
        "id": "zf_xiaobei", "name": "Xiaobei", "lang": "cmn", "lang_name": "Mandarin Chinese", "gender": "female",
        "style": "Clear & bright standard Mandarin",
        "aliases": ["chinese_female", "xiaobei"],
    },
    "zf_xiaoni": {
        "id": "zf_xiaoni", "name": "Xiaoni", "lang": "cmn", "lang_name": "Mandarin Chinese", "gender": "female",
        "style": "Warm & gentle Mandarin",
        "aliases": ["xiaoni"],
    },
    "zf_xiaoxiao": {
        "id": "zf_xiaoxiao", "name": "Xiaoxiao", "lang": "cmn", "lang_name": "Mandarin Chinese", "gender": "female",
        "style": "Expressive Mandarin storytelling",
        "aliases": ["xiaoxiao"],
    },
    "zf_xiaoyi": {
        "id": "zf_xiaoyi", "name": "Xiaoyi", "lang": "cmn", "lang_name": "Mandarin Chinese", "gender": "female",
        "style": "Smooth & pleasant Mandarin",
        "aliases": ["xiaoyi"],
    },
    "zm_yunjian": {
        "id": "zm_yunjian", "name": "Yunjian", "lang": "cmn", "lang_name": "Mandarin Chinese", "gender": "male",
        "style": "Formal, clear & newsreader tone",
        "aliases": ["chinese_male", "yunjian"],
    },
    "zm_yunxi": {
        "id": "zm_yunxi", "name": "Yunxi", "lang": "cmn", "lang_name": "Mandarin Chinese", "gender": "male",
        "style": "Natural & calm conversational",
        "aliases": ["yunxi"],
    },
    "zm_yunxia": {
        "id": "zm_yunxia", "name": "Yunxia", "lang": "cmn", "lang_name": "Mandarin Chinese", "gender": "male",
        "style": "Dynamic & lively male",
        "aliases": ["yunxia"],
    },
    "zm_yunyang": {
        "id": "zm_yunyang", "name": "Yunyang", "lang": "cmn", "lang_name": "Mandarin Chinese", "gender": "male",
        "style": "Deep & resonant male narration",
        "aliases": ["yunyang"],
    },
    # Spanish (es)
    "ef_dora": {
        "id": "ef_dora", "name": "Dora", "lang": "es", "lang_name": "Spanish", "gender": "female",
        "style": "Warm & natural Spanish",
        "aliases": ["spanish_female", "dora"],
    },
    "em_alex": {
        "id": "em_alex", "name": "Alex", "lang": "es", "lang_name": "Spanish", "gender": "male",
        "style": "Clear & conversational Spanish",
        "aliases": ["spanish_male", "alex"],
    },
    "em_santa": {
        "id": "em_santa", "name": "Santa ES", "lang": "es", "lang_name": "Spanish", "gender": "male",
        "style": "Mature & deep Spanish",
        "aliases": ["em_santa"],
    },
    # French (fr)
    "ff_siwis": {
        "id": "ff_siwis", "name": "Siwis", "lang": "fr", "lang_name": "French", "gender": "female",
        "style": "Natural & melodic French narration",
        "aliases": ["french_female", "siwis"],
    },
    # Italian (it)
    "if_sara": {
        "id": "if_sara", "name": "Sara", "lang": "it", "lang_name": "Italian", "gender": "female",
        "style": "Expressive Italian narration",
        "aliases": ["italian_female", "sara"],
    },
    "im_nicola": {
        "id": "im_nicola", "name": "Nicola", "lang": "it", "lang_name": "Italian", "gender": "male",
        "style": "Natural Italian male",
        "aliases": ["italian_male", "nicola"],
    },
    # Portuguese (pt-br)
    "pf_dora": {
        "id": "pf_dora", "name": "Dora PT", "lang": "pt-br", "lang_name": "Portuguese (BR)", "gender": "female",
        "style": "Clear Portuguese narration",
        "aliases": ["portuguese_female", "pf_dora"],
    },
    "pm_alex": {
        "id": "pm_alex", "name": "Alex PT", "lang": "pt-br", "lang_name": "Portuguese (BR)", "gender": "male",
        "style": "Expressive Portuguese male",
        "aliases": ["portuguese_male", "pm_alex"],
    },
}

VOICE_ALIASES: dict[str, str] = {}
for vid, vinfo in VOICE_CATALOG.items():
    VOICE_ALIASES[vid.lower()] = vid
    for alias in vinfo.get("aliases", []):
        VOICE_ALIASES[alias.lower().replace("-", "_")] = vid

# Backwards compatibility map
EN_VOICES = {
    vid: f"{vinfo['lang_name']} {vinfo['gender']}, {vinfo['style']}"
    for vid, vinfo in VOICE_CATALOG.items()
    if vinfo["lang"].startswith("en")
}


def resolve_voice(
    voice_input: str | None,
    default_lang: str = "hi",
    genre: str | None = None,
) -> str:
    """Resolve a voice ID, alias, or multi-voice blend expression.

    Supports:
      - Direct IDs: 'hm_omega', 'af_heart', 'bm_george'
      - Friendly aliases: 'hindi_male', 'hindi_female', 'documentary', 'deep_male', 'anime'
      - Multi-voice blending: 'af_heart,af_bella', 'af_heart+af_bella', 'hindi_male+hindi_female'
      - Fallback / Auto: language & genre aware selection
    """
    if not voice_input or voice_input.strip().lower() in ("auto", "default", "none"):
        lang = (default_lang or "hi").lower().strip()
        if lang in ("hi", "hindi", "hinglish"):
            return "hm_omega"
        elif lang in ("en", "en-us", "en-gb", "english"):
            if genre == "documentary":
                return "bm_george"
            elif genre == "horror":
                return "am_onyx"
            elif genre == "motivational":
                return "am_adam"
            elif genre in ("anime", "comedy"):
                return "af_bella"
            elif genre in ("fantasy", "drama"):
                return "af_heart"
            return "am_michael"
        elif lang in ("ja", "japanese"):
            return "jf_nezumi" if genre == "anime" else "jm_kumo"
        return "hm_omega"

    # Common typo fix: 'hm_beta' does not exist in Kokoro; map it to 'hf_beta' (Hindi female Ananya)
    if voice_input.strip().lower() == "hm_beta":
        return "hf_beta"

    # Replace '+' with ',' for blend syntax
    normalized = voice_input.replace("+", ",")
    parts = [p.strip() for p in normalized.split(",") if p.strip()]

    resolved_parts = []
    for part in parts:
        key = part.lower().replace("-", "_")
        resolved_parts.append(VOICE_ALIASES.get(key, part))

    return ",".join(resolved_parts)


def list_voices(lang_filter: str | None = None) -> list[dict]:
    """Return structured voice catalog items, optionally filtered by language."""
    items = []
    for vid, vinfo in VOICE_CATALOG.items():
        if lang_filter:
            filt = lang_filter.lower().strip()
            if filt == "hinglish":
                filt = "hi"
            # Match language code (e.g. 'hi', 'en', 'ja') or full/prefix name (e.g. 'hindi', 'english')
            lang_code = vinfo["lang"].lower()
            lang_name = vinfo["lang_name"].lower()
            if lang_code != filt and not lang_code.startswith(f"{filt}-") and filt != lang_name and not lang_name.startswith(filt):
                continue
        items.append(dict(vinfo))
    return items


def format_voices_table() -> str:
    """Return a terminal string displaying all voices grouped by language."""
    groups = {
        "Hindi (hi)": [v for v in VOICE_CATALOG.values() if v["lang"] == "hi"],
        "American English (en-us)": [v for v in VOICE_CATALOG.values() if v["lang"] == "en-us"],
        "British English (en-gb)": [v for v in VOICE_CATALOG.values() if v["lang"] == "en-gb"],
        "Japanese (ja)": [v for v in VOICE_CATALOG.values() if v["lang"] == "ja"],
        "Mandarin Chinese (cmn)": [v for v in VOICE_CATALOG.values() if v["lang"] == "cmn"],
        "Spanish (es)": [v for v in VOICE_CATALOG.values() if v["lang"] == "es"],
        "European Languages (fr, it, pt)": [
            v for v in VOICE_CATALOG.values() if v["lang"] in ("fr", "it", "pt-br")
        ],
    }

    lines = [
        "=" * 74,
        "KOKORO-82M VOICE CATALOG (Local Neural TTS, Zero API Cost)",
        "=" * 74,
    ]
    for grp_name, vlist in groups.items():
        lines.append(f"\n--- {grp_name} ---")
        for v in vlist:
            aliases_str = f" [Aliases: {', '.join(v['aliases'])}]" if v.get("aliases") else ""
            lines.append(f"  {v['id']:<14} : {v['name']:<10} | {v['gender']:<6} | {v['style']}")
            if aliases_str:
                lines.append(f"                 {aliases_str}")

    lines.extend([
        "\n" + "=" * 74,
        "VOICE BLENDING & MULTI-CHARACTER USAGE",
        "=" * 74,
        "1. Voice Blending (Mix voices seamlessly):",
        "     --voice \"af_heart+af_bella\"         (warm + bright English blend)",
        "     --voice \"hm_omega+hf_alpha\"         (deep male + expressive female Hindi blend)",
        "     --voice \"hindi_male,hindi_female\"    (blend by alias)",
        "",
        "2. Aliases:",
        "     --voice hindi_female                (maps to hf_alpha)",
        "     --voice documentary                 (maps to bm_george - BBC documentary style)",
        "     --voice deep_male                   (maps to am_onyx - deep bass trailer voice)",
        "     --voice anime_female                (maps to jf_nezumi)",
        "",
        "3. Multi-Character Dialogue:",
        "     In script.json, assign voices per-character or per-scene lines:",
        "       \"characters\": {",
        "           \"hero\": {\"name\": \"Goku\", \"voice\": \"am_adam\"},",
        "           \"villain\": {\"name\": \"Shadow\", \"voice\": \"am_onyx\"}",
        "       }",
    ])
    return "\n".join(lines)

# Rhubarb's Hanna-Barbera mouth shape set.
MOUTH_SHAPES = {"A", "B", "C", "D", "E", "F", "G", "H", "X"}

DEFAULT_SCRIPT = {
    "title": "Untitled",
    "voice": "am_michael",
    "speed": 1.0,
    "fps": 30,
    "width": 1920,
    "height": 1080,
    "tailPaddingSeconds": 1.2,
    "sentencePauseSeconds": 0.28,
    "master": DEFAULT_MASTER,
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
    "characters": {},
}

DEFAULT_SCENE = {
    "id": "",
    "narration": "",
    "lines": [],
    "headline": "",
    "subtext": "",
    "bullets": [],
    "headlinePosition": "center",
    "background": {"type": "gradient", "from": "#0b1220", "to": "#132a44", "angle": 160},
    "cast": [],
    # Procedural horror shots. When present these replace `background`.
    "shots": [],
    # Looping ambience bed, e.g. {"src": "amb_drone", "volume": 0.35}.
    "ambience": None,
    # One-shot effects at story beats: {"at": 12.5, "sfx": "sfx_stinger"}.
    "soundCues": [],
    "padBefore": 0.0,
    "padAfter": 0.55,
    "holdSeconds": 3.0,
}

ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "inc",
    "ltd", "co", "no", "vol", "fig", "approx", "dept", "est",
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


def slugify(value: str, fallback: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-")
    return slug or fallback


def find_rhubarb() -> Path | None:
    """Locate the Rhubarb binary unpacked by setup.ps1."""
    base = ROOT / "tools" / "rhubarb"
    if not base.exists():
        return None
    matches = sorted(base.rglob("rhubarb.exe")) or sorted(base.rglob("rhubarb"))
    return matches[0] if matches else None


def rate_to_speed(script: dict) -> float:
    """Accept either a `speed` multiplier or an edge-tts style `rate` string."""
    if isinstance(script.get("speed"), (int, float)):
        return max(0.5, min(2.0, float(script["speed"])))
    raw = str(script.get("rate") or "").strip().rstrip("%")
    try:
        return max(0.5, min(2.0, 1.0 + float(raw) / 100.0))
    except ValueError:
        return 1.0


# --------------------------------------------------------------------------
# Sentence handling
# --------------------------------------------------------------------------


def split_sentences(text: str) -> list[str]:
    """Split narration into sentences, keeping common abbreviations intact.

    Handles the Devanagari danda (।) and double danda (॥) as well as Latin
    punctuation, since Hindi narration uses them exclusively.
    """
    text = " ".join(text.split())
    if not text:
        return []

    sentences: list[str] = []
    for part in re.split(r"(?<=[.!?।॥])\s+", text):
        if sentences:
            tail = re.split(r"[\s.।॥]+", sentences[-1].rstrip())[-1].lower().strip(".")
            if tail in ABBREVIATIONS:
                sentences[-1] = f"{sentences[-1]} {part}"
                continue
        sentences.append(part)
    return sentences


def phoneme_weight(phonemes: str) -> float:
    """Kept for backwards compatibility; the real implementation lives in
    ``phonemise.py`` so it can be shared and tested on its own."""
    from phonemise import phoneme_weight as weight

    return weight(phonemes)


# --------------------------------------------------------------------------
# Speech synthesis
# --------------------------------------------------------------------------


class Voice:
    """Wraps the Kokoro pipeline, loading one per language code."""

    def __init__(self) -> None:
        self._pipelines: dict[str, object] = {}
        self.phonemiser = Phonemiser()

    def _pipeline(self, lang_code: str):
        if lang_code not in self._pipelines:
            from kokoro import KPipeline

            self._pipelines[lang_code] = KPipeline(lang_code=lang_code)
        return self._pipelines[lang_code]

    def say(self, text: str, voice: str, speed: float):
        """Synthesise one sentence; returns a float32 mono numpy array.

        The Kokoro language code is the first letter of the primary voice id, so the
        same call handles English, Hindi, blends (e.g. 'af_heart,af_bella'), and aliases.
        Minor internal punctuation like commas, semicolons, and dashes is cleaned to prevent
        unnatural mid-sentence audio pauses.
        """
        import numpy as np

        # Strip internal pauses caused by minor punctuation (commas, colons, semicolons, em-dashes)
        clean_text = re.sub(r"[,;:—–\-]+", " ", text)
        clean_text = " ".join(clean_text.split())
        if not clean_text:
            clean_text = text

        resolved_voice = resolve_voice(voice)
        primary = resolved_voice.split(",")[0].strip()
        lang_code = primary[:1].lower() or "a"
        chunks = [
            np.asarray(audio, dtype=np.float32)
            for _g, _p, audio in self._pipeline(lang_code)(clean_text, voice=resolved_voice, speed=speed)
        ]
        if not chunks:
            raise RuntimeError(f"Kokoro produced no audio for: {clean_text[:60]!r}")
        return np.concatenate(chunks)


def pause_after(sentence: str, base: float, index: int) -> float:
    """Silence to leave after a sentence, taken from the mark that ends it.

    The small wobble is deterministic, so the same script always renders to the
    same audio, but a run of full stops no longer ticks like a metronome.
    """
    stripped = sentence.rstrip(TRAILING_MARKS)
    mark = stripped[-1:] if stripped else ""
    scale = PAUSE_SCALE.get(mark, 1.0)
    wobble = 1.0 + (((index * 2654435761) % 997) / 997.0 - 0.5) * 0.12
    return max(0.0, base * scale * wobble)


def apply_fades(audio, seconds: float = 0.008):
    """Fade a segment in and out so joining segments cannot click.

    Without this, concatenating sentences that do not start and end at zero
    leaves a step discontinuity, which is audible as a tick between lines.
    """
    import numpy as np

    length = min(int(SAMPLE_RATE * seconds), len(audio) // 2)
    if length <= 0:
        return audio

    out = audio.copy()
    ramp = np.linspace(0.0, 1.0, length, dtype=out.dtype)
    out[:length] *= ramp
    out[-length:] *= ramp[::-1]
    return out


def compress_segments(audio, threshold_dbfs: float, ratio: float):
    """Block-based soft compressor.

    Works on 10 ms blocks with a one-pole smoothed gain, which is cheap and
    keeps the level steady across a long video without the pumping a per-sample
    envelope produces.
    """
    import numpy as np
    from scipy import signal

    block = max(1, int(0.01 * SAMPLE_RATE))
    pad = (-len(audio)) % block
    original_length = len(audio)
    if pad:
        audio = np.concatenate([audio, np.zeros(pad, dtype=audio.dtype)])

    frames = audio.reshape(-1, block)
    rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)

    over = rms / (10 ** (threshold_dbfs / 20.0))
    gain = np.where(over > 1.0, over ** (1.0 / ratio - 1.0), 1.0)
    gain = signal.lfilter([0.35], [1.0, -0.65], gain)

    out = (frames * gain[:, None]).reshape(-1)[:original_length]
    return np.ascontiguousarray(out)


def limit_peaks(audio, ceiling: float, *, hold_seconds: float = 0.002,
                release_seconds: float = 0.05):
    """Hold transient peaks under a ceiling without scaling the whole line.

    Scaling a line down to its single loudest sample (almost always a plosive)
    throws away that gain on everything else, which is exactly how line-to-line
    loudness drifts across a long video. Ducking the few offending transients
    keeps the level where it was put.
    """
    import numpy as np
    from scipy import signal
    from scipy.ndimage import minimum_filter1d

    magnitude = np.abs(audio)
    over = magnitude > ceiling
    if not over.any():
        return audio

    required = np.ones_like(magnitude)
    required[over] = ceiling / magnitude[over]

    # Hold the duck across the whole transient, not just the one sample that
    # happened to breach, then release slowly so the gain cannot chatter.
    hold = max(1, int(hold_seconds * SAMPLE_RATE))
    gain = minimum_filter1d(required, size=hold, mode="nearest")

    tau = max(1.0, release_seconds * SAMPLE_RATE)
    decay = float(np.exp(-1.0 / tau))
    gain = signal.lfilter([1.0 - decay], [1.0, -decay], gain)

    # Smoothing alone can leave the gain above what a peak needs, so take the
    # stricter of the two: the ceiling is guaranteed and the release is smooth.
    return audio * np.minimum(gain, required)


def master_audio(audio, settings: dict | None = None):
    """Condition a raw Kokoro line into something that sounds like narration.

    Kokoro is clean but dry: sentence loudness drifts, there is low-frequency
    rumble, and consonants sit back in the mix. Filter, presence lift, gentle
    compression, loudness match and a peak ceiling give a steady, intelligible
    track that needs no volume riding over thirty minutes.
    """
    import numpy as np

    settings = {**DEFAULT_MASTER, **(settings or {})}
    if audio.size == 0 or not settings.get("enabled", True):
        return audio

    x = np.asarray(audio, dtype=np.float64)
    short = x.size < 256  # sosfiltfilt needs room for its padding

    from scipy import signal

    nyquist = SAMPLE_RATE / 2.0

    # Everything below the vocal fundamental is headroom spent on rumble.
    highpass = float(settings.get("highpassHz") or 0.0)
    if highpass > 0 and not short:
        sos = signal.butter(2, highpass / nyquist, btype="highpass", output="sos")
        x = signal.sosfiltfilt(sos, x)

    # Lift the consonant band. A couple of dB is the difference between muffled
    # and clearly intelligible on a phone speaker.
    presence = float(settings.get("presence") or 0.0)
    if presence > 0 and not short:
        low = max(700.0, highpass * 6.0) / nyquist
        high = min(5500.0, nyquist * 0.9) / nyquist
        if low < high:
            sos = signal.butter(2, [low, high], btype="bandpass", output="sos")
            x = x + presence * signal.sosfiltfilt(sos, x)

    if settings.get("compress", True):
        x = compress_segments(
            x,
            float(settings.get("thresholdDbfs", -24.0)),
            float(settings.get("ratio", 3.0)),
        )

    # Match loudness so every line of a long video sits at the same level.
    target = float(settings.get("targetLoudnessDbfs", -18.0))
    rms = float(np.sqrt(np.mean(x**2))) or 1e-9
    x = x * (10 ** (target / 20.0)) / rms

    # Then guarantee headroom, because the mix adds ambience and sound cues.
    ceiling = 10 ** (float(settings.get("peakCeilingDbfs", -1.5)) / 20.0)
    x = limit_peaks(x, ceiling)

    return np.clip(x, -1.0, 1.0).astype(np.float32)


def synthesise_line(
    engine: Voice,
    text: str,
    voice: str,
    speed: float,
    pause: float,
    master: dict | None = None,
):
    """Synthesise a whole line, returning (audio, words).

    Each sentence is rendered separately so its duration is exact from the
    sample count, then that duration is spread across the sentence's words in
    proportion to how long each pronunciation actually is.

    Weighting is per spoken word and works for every language: English uses
    misaki's tokens, other languages zip espeak's phoneme groups against the
    words, and anything that fails falls back to syllable counts.
    """
    import numpy as np

    sentences = split_sentences(text)
    if not sentences:
        return np.zeros(0, dtype=np.float32), []

    pieces: list = []
    words: list[dict] = []
    offset = 0.0

    for index, sentence in enumerate(sentences):
        # Kokoro handles prosody inside a sentence (including commas), so it is
        # left intact and only the silence between sentences is shaped here.
        audio = apply_fades(engine.say(sentence, voice, speed))
        duration = len(audio) / SAMPLE_RATE

        spoken = [w for w in sentence.split() if w]
        weights = engine.phonemiser.weights(sentence, voice)
        if len(weights) != len(spoken):
            # Should not happen, but even weighting beats dropping the caption.
            weights = [1.0] * len(spoken)

        total = sum(weights) or 1.0
        cursor = offset
        for word, weight in zip(spoken, weights):
            span = duration * weight / total
            words.append(
                {
                    "text": word,
                    "start": round(cursor, 4),
                    "end": round(cursor + span, 4),
                }
            )
            cursor += span

        pieces.append(audio)
        offset += duration

        if index < len(sentences) - 1:
            gap = pause_after(sentence, pause, index)
            pieces.append(np.zeros(int(SAMPLE_RATE * gap), dtype=np.float32))
            offset += gap

    return master_audio(np.concatenate(pieces), master), words


# --------------------------------------------------------------------------
# Lip sync
# --------------------------------------------------------------------------


def run_rhubarb(rhubarb: Path, wav: Path, text: str) -> list[dict]:
    """Return Rhubarb mouth cues for a WAV file, or [] if it cannot be run."""
    out_json = wav.with_suffix(".mouth.json")
    dialog = wav.with_suffix(".dialog.txt")
    dialog.write_text(text, encoding="utf-8")

    command = [
        str(rhubarb),
        "-f", "json",
        "-r", "phonetic",
        "-o", str(out_json),
        "-d", str(dialog),
        str(wav),
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, timeout=900)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        detail = getattr(exc, "stderr", b"") or b""
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", "replace")
        print(f"      rhubarb failed ({type(exc).__name__}): {detail.strip()[:180]}", file=sys.stderr)
        return []

    try:
        payload = json.loads(out_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    return [
        {
            "start": round(float(cue["start"]), 4),
            "end": round(float(cue["end"]), 4),
            "value": str(cue["value"]),
        }
        for cue in (payload.get("mouthCues") or [])
        if str(cue.get("value")) in MOUTH_SHAPES
    ]


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------


def build_line(
    engine: Voice,
    *,
    audio_path: Path,
    text: str,
    voice: str,
    speed: float,
    pause: float,
    master: dict | None = None,
    rhubarb: Path | None,
    force: bool,
    label: str,
) -> dict:
    meta_path = audio_path.with_suffix(".json")
    key = hashlib.sha256(
        json.dumps(
            [AUDIO_PIPELINE_VERSION, voice, round(speed, 4), round(pause, 4), master, text]
        ).encode("utf-8")
    ).hexdigest()

    if not force and audio_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("hash") == key:
                print(
                    f"    {label}: cached ({meta['duration']:.1f}s, "
                    f"{len(meta.get('words', []))} words)"
                )
                return meta
        except (OSError, json.JSONDecodeError, KeyError):
            pass

    import soundfile as sf

    audio, words = synthesise_line(engine, text, voice, speed, pause, master)
    if len(audio) == 0:
        return {"hash": key, "duration": 0.0, "words": [], "mouthCues": []}

    duration = round(len(audio) / SAMPLE_RATE, 4)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    sf.write(str(audio_path), audio, SAMPLE_RATE, subtype="PCM_16")

    cues = run_rhubarb(rhubarb, audio_path, text) if rhubarb is not None else []

    meta = {"hash": key, "duration": duration, "words": words, "mouthCues": cues}
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    print(f"    {label}: {duration:.1f}s, {len(words)} words, {len(cues)} mouth cues")
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("script", nargs="?", default=str(ROOT / "script.json"))
    parser.add_argument("--speed", type=float, default=None, help="speech speed multiplier (e.g. 0.8 to 1.5, default from script.json or .env)")
    parser.add_argument("--force", action="store_true", help="re-synthesise cached audio")
    parser.add_argument("--no-lipsync", action="store_true", help="skip Rhubarb mouth shapes")
    parser.add_argument("--voices", action="store_true", help="list voices and exit")
    parser.add_argument("--hinglish", action="store_true", help="force Roman Hinglish on-screen captions & subtitles")
    parser.add_argument("--no-hinglish", action="store_true", help="disable Roman Hinglish transliteration (keep Devanagari subtitles)")
    args = parser.parse_args()

    # Ensure audio assets (ambience beds, sfx) are present
    try:
        ensure_assets()
    except Exception as e:
        print(f"note: could not verify all procedural audio assets: {e}", file=sys.stderr)

    if args.voices:
        print(format_voices_table())
        return 0

    script_path = Path(args.script).resolve()
    if not script_path.exists():
        print(f"error: {script_path} not found", file=sys.stderr)
        return 1

    raw = json.loads(script_path.read_text(encoding="utf-8"))
    script = merge(DEFAULT_SCRIPT, raw)

    captions_cfg = script.get("captions", {})
    script_lang = (script.get("language") or "hi").lower().strip()
    is_hinglish_captions = (
        (
            script_lang in ("hinglish", "hi", "hindi")
            and captions_cfg.get("hinglish") is not False
            and captions_cfg.get("subtitles") not in ("devanagari", "hindi")
        )
        or bool(captions_cfg.get("hinglish"))
        or getattr(args, "hinglish", False)
    ) and not getattr(args, "no_hinglish", False)

    speed = rate_to_speed(script)
    if args.speed is not None:
        speed = max(0.5, min(2.0, float(args.speed)))
    else:
        env_speed = os.environ.get("SPEECH_SPEED") or os.environ.get("VOICE_SPEED")
        if env_speed:
            try:
                speed = max(0.5, min(2.0, float(env_speed)))
            except ValueError:
                pass
    script["speed"] = speed
    voice_default = script["voice"]
    pause = float(script["sentencePauseSeconds"])
    master = script.get("master") or {}
    characters = script.get("characters") or {}

    scenes_in = raw.get("scenes") or []
    if not scenes_in:
        print("error: script.json has no scenes", file=sys.stderr)
        return 1

    scenes = [merge(DEFAULT_SCENE, s) for s in scenes_in]
    for index, scene in enumerate(scenes):
        scene["id"] = slugify(scene.get("id") or "", f"scene-{index + 1}")
        if is_hinglish_captions and is_devanagari(scene.get("headline", "")):
            scene["headline"] = deva_to_hinglish(scene["headline"])
        # `narration` is shorthand for a single spoken line. `lines` allows
        # dialogue, where each line can be given its own character and voice.
        if not scene["lines"] and scene["narration"]:
            speaker = scene["cast"][0].get("id") if scene["cast"] else None
            scene["lines"] = [{"character": speaker, "text": scene["narration"]}]
    script["scenes"] = scenes

    if not any(scene["lines"] for scene in scenes):
        print("error: no narration found in any scene", file=sys.stderr)
        return 1

    rhubarb_name = "disabled"
    rhubarb = None
    if not args.no_lipsync:
        rhubarb = find_rhubarb()
        rhubarb_name = str(rhubarb) if rhubarb else "NOT FOUND (run setup.ps1)"

    print(f"Voice    : {voice_default}  speed {speed:g}  ({len(scenes)} scenes)")
    print(f"Prosody  : {pause:g}s base pause, mark-aware, pipeline v{AUDIO_PIPELINE_VERSION}")
    print(f"Mastering: {'on' if master.get('enabled', True) else 'OFF'}")
    print(f"Rhubarb  : {rhubarb_name}")
    print("Synthesising...")

    engine = Voice()
    timeline_scenes: list[dict] = []
    cursor = 0.0
    total_words = 0

    for scene_index, scene in enumerate(scenes):
        scene_start = cursor
        pad_before = float(scene.get("padBefore") or 0.0)
        pad_after = float(scene.get("padAfter") or 0.0)

        line_cursor = scene_start + pad_before
        built: list[dict] = []

        for line_index, line in enumerate(scene["lines"]):
            text = " ".join(str(line.get("text") or "").split())
            if not text:
                continue

            character_id = line.get("character")
            character = (characters.get(character_id) or {}) if character_id else {}
            line_raw_voice = line.get("voice") or character.get("voice") or scene.get("voice") or voice_default
            voice = resolve_voice(line_raw_voice)
            line_speed = float(character.get("speed") or scene.get("speed") or speed)

            suffix = f"-{line_index + 1}" if len(scene["lines"]) > 1 else ""
            audio_path = AUDIO_DIR / f"{scene['id']}{suffix}.wav"

            meta = build_line(
                engine,
                audio_path=audio_path,
                text=text,
                voice=voice,
                speed=line_speed,
                pause=pause,
                master=master,
                rhubarb=rhubarb,
                force=args.force,
                label=f"[{scene_index + 1}.{line_index + 1}] {audio_path.name}",
            )
            if meta["duration"] <= 0:
                continue

            built.append(
                {
                    "characterId": character_id,
                    "audio": f"audio/{audio_path.name}",
                    "_abs": line_cursor,
                    "duration": meta["duration"],
                    "words": meta["words"],
                    "mouthCues": meta.get("mouthCues") or [],
                }
            )
            line_cursor += meta["duration"]
            total_words += len(text.split())

        if not built:
            # A hold card: no speech, the scene is simply held on screen.
            length = pad_before + max(pad_after, float(scene.get("holdSeconds") or 3.0))
            timeline_scenes.append(
                {
                    "id": scene["id"],
                    "index": scene_index,
                    "start": round(scene_start, 4),
                    "duration": round(length, 4),
                    # Absolute, matching scene-level `start`, so the renderer can
                    # place audio and captions without knowing scene boundaries.
                    "audioStart": round(scene_start + pad_before, 4),
                    "audio": None,
                    "words": [],
                    "lines": [],
                }
            )
            cursor = scene_start + length
            continue

        # Word times are line-relative; shift them so they are relative to the
        # start of the scene's audio, which is what the captions use.
        audio_start_abs = scene_start + pad_before
        words = [
            {
                "text": (
                    deva_word_to_hinglish(word["text"])
                    if (is_hinglish_captions and is_devanagari(word["text"]))
                    else word["text"]
                ),
                "start": round(word["start"] + line["_abs"] - audio_start_abs, 4),
                "end": round(word["end"] + line["_abs"] - audio_start_abs, 4),
            }
            for line in built
            for word in line["words"]
        ]

        lines = [
            {
                "characterId": line["characterId"],
                "audio": line["audio"],
                "start": round(line["_abs"], 4),
                "duration": line["duration"],
                "words": line["words"],
                "mouthCues": line["mouthCues"],
            }
            for line in built
        ]

        total_line_duration = sum(line["duration"] for line in built)
        length = pad_before + total_line_duration + pad_after

        timeline_scenes.append(
            {
                "id": scene["id"],
                "index": scene_index,
                "start": round(scene_start, 4),
                "duration": round(length, 4),
                "audioStart": round(scene_start + pad_before, 4),
                "audio": built[0]["audio"],
                "words": words,
                "lines": lines,
            }
        )
        cursor = scene_start + length

    GENERATED.mkdir(parents=True, exist_ok=True)
    project = {
        "script": script,
        "timings": {
            "totalDuration": round(cursor, 4),
            "totalWords": total_words,
            "scenes": timeline_scenes,
        },
    }
    (GENERATED / "project.json").write_text(
        json.dumps(project, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Export synchronized .srt subtitle files
    out_dir = ROOT / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    srt_out = out_dir / "subtitles.srt"
    gen_srt = GENERATED / "subtitles.srt"
    try:
        words_per_cue = int(captions_cfg.get("wordsPerCue") or 4)
        export_srt(
            timeline_scenes,
            srt_out,
            words_per_cue=words_per_cue,
            transliterate_hinglish=is_hinglish_captions,
        )
        export_srt(
            timeline_scenes,
            gen_srt,
            words_per_cue=words_per_cue,
            transliterate_hinglish=is_hinglish_captions,
        )
    except Exception as exc:
        print(f"note: could not export .srt subtitles: {exc}", file=sys.stderr)

    minutes, seconds = divmod(cursor, 60)
    wpm = total_words / (cursor / 60) if cursor else 0
    print(
        f"\nTimeline : {int(minutes)}m {seconds:04.1f}s | {total_words} words | {wpm:.0f} wpm"
    )
    print(f"Wrote    : {(GENERATED / 'project.json').relative_to(ROOT)}")
    if srt_out.exists():
        print(f"Subtitles: {srt_out.relative_to(ROOT)}")
    if wpm > 185:
        print("Note: over 185 wpm is fast for narration; consider fewer words per scene.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
