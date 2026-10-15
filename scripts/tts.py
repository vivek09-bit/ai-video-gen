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
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

# Allow importing the sibling phonemise module when this file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Ensure UTF-8 console output for Devanagari and Unicode text on Windows
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from phonemise import Phonemiser  # noqa: E402
from make_assets import ensure_assets  # noqa: E402
from generate_images import load_env_file  # noqa: E402
from transliterate import deva_word_to_hinglish, deva_to_hinglish, is_devanagari, export_srt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PUBLIC = ROOT / "public"
AUDIO_DIR = PUBLIC / "audio" / "voice"
GENERATED = ROOT / "src" / "generated"

# Load .env configuration if present
load_env_file(ROOT / ".env")

SAMPLE_RATE = 24000  # Kokoro's native rate

# Bumped whenever the prosody or mastering chain changes, so audio cached by an
# older chain is rebuilt instead of being silently reused.
AUDIO_PIPELINE_VERSION = 13


# Emotional and narrative beat speed multipliers for natural, humanized delivery.
# In human storytelling, setup and dread are spoken slower and more deliberately,
# while action and climax accelerate with rising tension.
BEAT_SPEED_MULTIPLIERS = {
    "establish": 0.96,   # Deliberate, atmospheric opening
    "dread": 0.92,       # Slow, tense, unsettling whisper pace
    "inciting": 1.02,    # Hooking the listener with slight urgency
    "approach": 1.08,    # Rising action, acceleration
    "rising": 1.10,      # Building suspense and pace
    "reveal": 1.18,      # Fast, urgent, climactic burst
    "climax": 1.20,      # Peak adrenaline and shock delivery
    "react": 0.94,       # Catching breath, sobering realization
    "resolve": 0.95,     # Lingering, chilling conclusion
    "button": 0.98,      # Punchy closing takeaway
}

EMOTION_SPEED_MULTIPLIERS = {
    "calm": 0.95,
    "worried": 0.93,
    "shock": 1.16,
    "sad": 0.90,
    "neutral": 1.00,
}

# How long to hold silence after a sentence, relative to the configured base
# pause. Tightened for natural flowing speech without awkward silent gaps.
PAUSE_SCALE = {
    ",": 0.25,
    ";": 0.0,
    ":": 0.0,
    "\u2014": 0.0,
    "-": 0.0,
    ".": 0.50,
    "\u0964": 0.50,  # Devanagari danda
    "\u0965": 0.70,  # Devanagari double danda
    "!": 0.35,  # Urgent breathless continuation after exclamations
    "?": 0.60,  # Reflective pause after questions
}

# Quotes and brackets that can follow the mark which actually ends a sentence.
TRAILING_MARKS = "\"'\u201d\u2019)]} \t\u00a0"

# Kokoro's output is clean but dry and uneven, so the narration is run through a
# small mastering chain before it is written out.
DEFAULT_MASTER = {
    "enabled": True,
    "highpassHz": 85.0,
    "presence": 0.35,
    "compress": True,
    "thresholdDbfs": -24.0,
    "ratio": 3.0,
    "targetLoudnessDbfs": -18.0,
    "peakCeilingDbfs": -1.5,
}

# Hindi colloquial phonetic normalizations for TTS pronunciation.
# In standard written Hindi literature, 'वह' is written for 'woh' (/voː/) but standard
# eSpeak / Kokoro G2P reads it strictly literally as 'vah'/'wah' (/vəh/).
# Similarly 'यह' is read as 'yah' instead of natural 'ye' (/jeː/).
# Normalizing these spoken homophones prior to synthesis gives natural, authentic pronunciation.
HINDI_0_TO_99 = [
    "शून्य", "एक", "दो", "तीन", "चार", "पांच", "छह", "सात", "आठ", "नौ",
    "दस", "ग्यारह", "बारह", "तेरह", "चौदह", "पंद्रह", "सोलह", "सत्रह", "अठारह", "उन्नीस",
    "बीस", "इक्कीस", "बाईस", "तेईस", "चौबीस", "पच्चीस", "छब्बीस", "सत्ताईस", "अट्ठाईस", "उनतीस",
    "तीस", "इकत्तीस", "बत्तीस", "तैंतीस", "चौंतीस", "पैंतीस", "छत्तीस", "सैंतीस", "अड़तीस", "उनतालीस",
    "चालीस", "इकतालीस", "बयालीस", "तैंतालीस", "चवालीस", "पैंतालीस", "छियालीस", "सैंतालीस", "अड़तालीस", "उनचास",
    "पचास", "इक्यावन", "बावन", "तिरपन", "चौवन", "पचपन", "छप्पन", "सत्तावन", "अट्ठावन", "उनसठ",
    "साठ", "इकसठ", "बासठ", "तिरसठ", "चौंसठ", "पैंसठ", "छियासठ", "सरसठ", "अड़सठ", "उनहत्तर",
    "सत्तर", "इकहत्तर", "बहत्तर", "तिहत्तर", "चौहत्तर", "पचहत्तर", "छिहत्तर", "सतहत्तर", "अठहत्तर", "उनासी",
    "अस्सी", "इक्यासी", "बयासी", "तिरासी", "चौरासी", "पचासी", "छियासी", "सत्तासी", "अट्ठासी", "नवासी",
    "नब्बे", "इक्यानवे", "बानवे", "तिरानवे", "चौरानवे", "पचानवे", "छियानवे", "सत्तानवे", "अट्ठानवे", "निन्यानवे",
]

DEVA_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def num_to_hindi(n: int) -> str:
    """Convert integer to natural conversational Hindi words."""
    if n < 0:
        return "माइनस " + num_to_hindi(-n)
    if n < 100:
        return HINDI_0_TO_99[n]
    if 100 <= n < 1000:
        hundreds = n // 100
        rem = n % 100
        res = HINDI_0_TO_99[hundreds] + " सौ"
        if rem > 0:
            res += " " + num_to_hindi(rem)
        return res
    if 1000 <= n < 100000:
        thousands = n // 1000
        rem = n % 1000
        res = num_to_hindi(thousands) + " हज़ार"
        if rem > 0:
            res += " " + num_to_hindi(rem)
        return res
    if 100000 <= n < 10000000:
        lakhs = n // 100000
        rem = n % 100000
        res = num_to_hindi(lakhs) + " लाख"
        if rem > 0:
            res += " " + num_to_hindi(rem)
        return res
    crores = n // 10000000
    rem = n % 10000000
    res = num_to_hindi(crores) + " करोड़"
    if rem > 0:
        res += " " + num_to_hindi(rem)
    return res


def expand_hindi_numbers(text: str) -> str:
    """Replace all Arabic and Devanagari digits with spoken Hindi Devanagari words."""
    text = text.translate(DEVA_DIGITS)

    def repl(m: re.Match) -> str:
        try:
            return num_to_hindi(int(m.group(0)))
        except Exception:
            return m.group(0)

    return re.sub(r"(?<![0-9])[0-9]+(?![0-9])", repl, text)


def _deva_word_regex(word: str) -> str:
    # Python \b doesn't treat Unicode combining vowel signs (matras) as \w.
    # We use Devanagari Unicode block [0900-097F] lookarounds for exact word boundaries.
    return rf"(?<![ऀ-ॿ]){re.escape(word)}(?![ऀ-ॿ])"


# Common English acronyms & alphanumeric abbreviations frequently found in Hindi scripts
# that should be expanded into natural spoken Hindi phonemes before digit-to-number expansion.
ACRONYM_TO_SPOKEN_HINDI: list[tuple[str, str]] = [
    (r"\b3[dD]\b", "थ्री-डी"),
    (r"\b3-[dD]\b", "थ्री-डी"),
    (r"(?<![ऀ-ॿ])3डी(?![ऀ-ॿ])", "थ्री-डी"),
    (r"(?<![ऀ-ॿ])3-डी(?![ऀ-ॿ])", "थ्री-डी"),
    (r"\b2[dD]\b", "टू-डी"),
    (r"\b2-[dD]\b", "टू-डी"),
    (r"(?<![ऀ-ॿ])2डी(?![ऀ-ॿ])", "टू-डी"),
    (r"(?<![ऀ-ॿ])2-डी(?![ऀ-ॿ])", "टू-डी"),
    (r"\b4[kK]\b", "फोर-के"),
    (r"\b4-[kK]\b", "फोर-के"),
    (r"(?<![ऀ-ॿ])4के(?![ऀ-ॿ])", "फोर-के"),
    (r"(?<![ऀ-ॿ])4-के(?![ऀ-ॿ])", "फोर-के"),
    (r"\b8[kK]\b", "एट-के"),
    (r"\b8-[kK]\b", "एट-के"),
    (r"(?<![ऀ-ॿ])8के(?![ऀ-ॿ])", "एट-के"),
    (r"(?<![ऀ-ॿ])8-के(?![ऀ-ॿ])", "एट-के"),
    (r"\b[hH][dD]\b", "एच-डी"),
    (r"(?<![ऀ-ॿ])एचडी(?![ऀ-ॿ])", "एच-डी"),
    (r"\b[aA][iI]\b", "ए-आई"),
    (r"\b[aA]\.[iI]\.\b", "ए-आई"),
    (r"(?<![ऀ-ॿ])एआई(?![ऀ-ॿ])", "ए-आई"),
    (r"\b[cC][cC][tT][vV]\b", "सीसीटीवी"),
    (r"\b[dD][nN][aA]\b", "डीएनए"),
    (r"\b[gG][pP][sS]\b", "जीपीएस"),
    (r"\b[uU][sS][bB]\b", "यूएसबी"),
    (r"\b[sS][iI][mM]\b", "सिम"),
    (r"\b[rR][aA][mM]\b", "रैम"),
    (r"\b[cC][pP][uU]\b", "सीपीयू"),
    (r"\b[pP][dD][fF]\b", "पीडीएफ"),
    (r"\b[wW][iI]-?[fF][iI]\b", "वाइ-फ़ाई"),
    (r"\b[lL][iI][dD][aA][rR]\b", "लिडार"),
    (r"\bLiDAR\b", "लिडार"),
    (r"\b[bB]\.?[cC]\.?\b", "बी-सी"),
    (r"\b[aA]\.?[dD]\.?\b", "ए-डी"),
    (r"\b[bB][cC][eE]\b", "बी-सी-ई"),
    (r"\b[cC][eE]\b", "सी-ई"),
    (r"\b[uU][fF][oO]\b", "यूएफओ"),
    (r"\b[uU][fF][oO]s\b", "यूएफओ"),
]

HINDI_PHONETIC_PAIRS = [
    ("वह", "वो"),
    ("यह", "ये"),
    ("वही", "वोही"),
    ("यही", "येही"),
    ("वे", "वो"),
    ("इन्हें", "इन्हे"),
    ("उन्हें", "उन्हे"),
    ("जिन्हें", "जिन्हे"),
    ("किन्हें", "किन्हे"),
    ("क्योंकि", "क्युंकी"),
    ("बहुत", "बोहत"),
    ("पहला", "पेहला"),
    ("पहली", "पेहली"),
    ("पहले", "पेहले"),
    ("रहता", "रेहता"),
    ("रहती", "रेहती"),
    ("रहते", "रेहते"),
    ("कहना", "केहना"),
    ("कहता", "केहता"),
    ("कहती", "केहती"),
    ("कहते", "केहते"),
    ("शहर", "शेहर"),
    ("जहर", "ज़ेहर"),
    # Spoken loanword pronunciation improvements (e.g. /z/ nukta for 'laser', proper vowels)
    ("लेजर", "लेज़र"),
    ("लेजरों", "लेज़रों"),
    ("लेजरस", "लेज़र्स"),
    ("इंजिनियरिंग", "इंजीनियरिंग"),
    ("इंजिनियर", "इंजीनियर"),
    ("टेक्नॉलॉजी", "टेक्नोलॉजी"),
    ("एकॉस्टिक", "अकॉस्टिक"),
]

HINDI_PHONETIC_REPLACEMENTS = [
    (_deva_word_regex(src), dst) for src, dst in HINDI_PHONETIC_PAIRS
]

# Common English loanwords frequently found in Hindi/Hinglish scripts that should be
# written in Devanagari so Kokoro's Hindi G2P phonemises them with authentic native pronunciation.
LATIN_LOANWORDS_TO_DEVA: dict[str, str] = {
    "laser": "लेज़र",
    "lazer": "लेज़र",
    "lasers": "लेज़र्स",
    "lazers": "लेज़र्स",
    "engineering": "इंजीनियरिंग",
    "injiniyaring": "इंजीनियरिंग",
    "engineer": "इंजीनियर",
    "engineers": "इंजीनियर्स",
    "architecture": "आर्किटेक्चर",
    "architect": "आर्किटेक्ट",
    "architects": "आर्किटेक्ट्स",
    "scanner": "स्कैनर",
    "scanners": "स्कैनर्स",
    "scanning": "स्कैनिंग",
    "scan": "स्कैन",
    "scans": "स्कैन",
    "acoustic": "अकॉस्टिक",
    "acoustics": "अकॉस्टिक्स",
    "frequency": "फ्रीक्वेंसी",
    "frequencies": "फ्रीक्वेंसीज",
    "vibration": "वाइब्रेशन",
    "vibrations": "वाइब्रेशन्स",
    "pyramid": "पिरामिड",
    "pyramids": "पिरामिड्स",
    "megalith": "मेगालिथ",
    "megalithic": "मेगालिथिक",
    "underground": "अंडरग्राउंड",
    "chamber": "चैंबर",
    "chambers": "चैंबर्स",
    "dimension": "डायमेंशन",
    "dimensions": "डायमेंशन्स",
    "structure": "स्ट्रक्चर",
    "structures": "स्ट्रक्चर्स",
    "technology": "टेक्नोलॉजी",
    "technologies": "टेक्नोलॉजीज",
    "civilization": "सिविलाइजेशन",
    "civilizations": "सिविलाइजेशन्स",
    "energy": "एनर्जी",
    "magnetic": "मैग्नेटिक",
    "radiation": "रेडिएशन",
    "satellite": "सैटेलाइट",
    "satellites": "सैटेलाइट्स",
    "radar": "रडार",
    "sonar": "सोनार",
    "machine": "मशीन",
    "machines": "मशीनें",
    "machinery": "मशीनरी",
    "drill": "ड्रिल",
    "drilling": "ड्रिलिंग",
    "cutting": "कटिंग",
    "cut": "कट",
    "cuts": "कट",
    "block": "ब्लॉक",
    "blocks": "ब्लॉक्स",
    "stone": "स्टोन",
    "stones": "स्टोन्स",
    "granite": "ग्रेनाइट",
    "marble": "मार्बल",
    "metal": "मेटल",
    "metals": "मेटल्स",
    "mystery": "मिस्ट्री",
    "mysteries": "मिस्ट्रीज",
    "mysterious": "मिस्टीरियस",
    "theory": "थ्योरी",
    "theories": "थ्योरीज",
    "evidence": "एविडेंस",
    "scientist": "साइंटिस्ट",
    "scientists": "साइंटिस्ट्स",
    "research": "रिसर्च",
    "researcher": "रिसर्चर",
    "researchers": "रिसर्चर्स",
    "microscope": "माइक्रोस्कोप",
    "telescope": "टेलिस्कोप",
    "discovery": "डिस्कवरी",
    "station": "स्टेशन",
    "train": "ट्रेन",
    "room": "रूम",
    "hotel": "होटल",
    "hospital": "हॉस्पिटल",
    "doctor": "डॉक्टर",
    "phone": "फ़ोन",
    "call": "कॉल",
    "video": "वीडियो",
    "camera": "कैमरा",
    "screen": "स्क्रीन",
    "police": "पुलिस",
    "secret": "सीक्रेट",
    "secrets": "सीक्रेट्स",
    "horror": "हॉरर",
    "ghost": "घोस्ट",
    "ghosts": "घोस्ट्स",
    "alien": "एलियन",
    "aliens": "एलियन्स",
    "forest": "फॉरेस्ट",
    "mirror": "मिरर",
    "shadow": "शैडो",
    "tunnel": "टनल",
    "tunnels": "टनल्स",
    "track": "ट्रैक",
    "light": "लाइट",
    "sound": "साउंड",
    "voice": "वॉइस",
    "door": "डोर",
    "signal": "सिग्नल",
    "message": "मैसेज",
    "story": "स्टोरी",
    "scam": "स्कैम",
    "hacker": "हैकर",
    "hack": "हैक",
    "password": "पासवर्ड",
    "computer": "कंप्यूटर",
    "laptop": "लैपटॉप",
    "internet": "इंटरनेट",
    "link": "लिंक",
    "app": "ऐप",
    "online": "ऑनलाइन",
    "offline": "ऑफलाइन",
    "game": "गेम",
    "level": "लेवल",
    "alert": "अलर्ट",
    "viral": "वायरल",
    "box": "बॉक्स",
    "floor": "फ़्लोर",
    "entry": "एंट्री",
    "exit": "एग्जिट",
    "radio": "रेडियो",
    "car": "कार",
    "bus": "बस",
    "road": "रोड",
    "drive": "ड्राइव",
    "driver": "ड्राइवर",
    "night": "नाईट",
    "house": "हाउस",
    "lock": "लॉक",
}


def normalize_hindi_phonetics(text: str) -> str:
    """Pre-normalize Hindi literary spellings, acronyms, loanwords, and digits to authentic spoken Devanagari."""
    normalized = text

    # Step 1: Pre-process acronyms & alphanumeric combos (e.g. 3D -> थ्री-डी, 4K -> फोर-के, AI -> ए-आई)
    # This MUST run before digit expansion so '3D' isn't broken into 'तीन' + 'D'.
    for pattern, replacement in ACRONYM_TO_SPOKEN_HINDI:
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)

    # Step 2: Convert Latin loanwords to Devanagari (e.g. 'laser' -> 'लेज़र', 'engineering' -> 'इंजीनियरिंग')
    for eng, deva in LATIN_LOANWORDS_TO_DEVA.items():
        normalized = re.sub(rf"\b{re.escape(eng)}\b", deva, normalized, flags=re.IGNORECASE)

    # Step 3: Expand standalone digits and numbers to spoken Hindi words (e.g. 5000 -> पांच हज़ार)
    normalized = expand_hindi_numbers(normalized)

    # Step 4: Apply Devanagari phonetic replacements (e.g. 'वह' -> 'वो', 'यह' -> 'ये', 'लेजर' -> 'लेज़र')
    for pattern, replacement in HINDI_PHONETIC_REPLACEMENTS:
        normalized = re.sub(pattern, replacement, normalized)

    return normalized

# Kokoro voice catalog across Hindi, English (US/UK), Japanese, Chinese, Spanish, French, Italian, Portuguese.
VOICE_CATALOG: dict[str, dict] = {
    # Hindi (hi)
    "hm_omega": {
        "id": "hm_omega", "name": "Aarav", "lang": "hi", "lang_name": "Hindi / Hinglish", "gender": "male",
        "style": "Deep, resonant & authoritative (ideal for horror, mythology, epics)",
        "aliases": ["hindi_male", "hindi_male_deep", "omega", "hindi_deep", "hindi_horror"],
    },
    "hm_psi": {
        "id": "hm_psi", "name": "Kabir", "lang": "hi", "lang_name": "Hindi / Hinglish", "gender": "male",
        "style": "Calm, natural & conversational (ideal for scifi, documentary, explainer)",
        "aliases": ["hindi_male_calm", "psi", "hindi_calm", "hindi_doc"],
    },
    "hf_alpha": {
        "id": "hf_alpha", "name": "Priya", "lang": "hi", "lang_name": "Hindi / Hinglish", "gender": "female",
        "style": "Warm, expressive & engaging (ideal for fantasy, drama, storytelling)",
        "aliases": ["hindi_female", "hindi_female_warm", "alpha", "hindi_warm", "hindi_fantasy"],
    },
    "hf_beta": {
        "id": "hf_beta", "name": "Ananya", "lang": "hi", "lang_name": "Hindi / Hinglish", "gender": "female",
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

# --------------------------------------------------------------------------
# Cartesia Sonic Multilingual Neural TTS Voices & Integration
# --------------------------------------------------------------------------

CARTESIA_VOICES: dict[str, dict] = {
    "cartesia:hindi_male": {
        "id": "20e68f5c-08e5-42d0-8e9b-6e716fd1ae66",
        "name": "Vivek",
        "lang": "hi",
        "gender": "male",
        "style": "Deep, grounded & eerie (low-pitch horror / dark mystery narration)",
        "fallback_kokoro": "hm_omega",
        "aliases": [
            "cartesia_male",
            "cartesia_hindi_male",
            "cartesia:horror",
            "cartesia_horror",
            "cartesia:vivek",
            "cartesia_vivek",
            "cartesia",
        ],
    },
    "cartesia:storyteller": {
        "id": "7e8cb11d-37af-476b-ab8f-25da99b18644",
        "name": "Anuj",
        "lang": "hi",
        "gender": "male",
        "style": "Engaging, emotional & suspenseful (storytelling / thrillers)",
        "fallback_kokoro": "hm_omega",
        "aliases": [
            "cartesia_storyteller",
            "cartesia:anuj",
            "cartesia_anuj",
            "cartesia:expressive",
            "cartesia_expressive",
        ],
    },
    "cartesia:actor": {
        "id": "bdab08ad-4137-4548-b9db-6142854c7525",
        "name": "Imran",
        "lang": "hi",
        "gender": "male",
        "style": "Intense Bollywood actor style for dramatic, serious and shocking moments",
        "fallback_kokoro": "hm_omega",
        "aliases": [
            "cartesia_actor",
            "cartesia_cinema",
            "cartesia:imran",
            "cartesia_imran",
        ],
    },
    "cartesia:announcer": {
        "id": "be79f378-47fe-4f9c-b92b-f02cefa62ccf",
        "name": "Sunil",
        "lang": "hi",
        "gender": "male",
        "style": "Deep authoritative official announcer (crime / investigative)",
        "fallback_kokoro": "hm_omega",
        "aliases": [
            "cartesia_announcer",
            "cartesia:sunil",
            "cartesia_sunil",
            "cartesia:deep",
            "cartesia_deep",
        ],
    },
    "cartesia:host": {
        "id": "910fb75e-1d20-4840-ac63-ac6b26a71bdc",
        "name": "Dev",
        "lang": "hi",
        "gender": "male",
        "style": "Warm, genuine conversational Hindi male (explainers / daily podcasts)",
        "fallback_kokoro": "hm_psi",
        "aliases": [
            "cartesia_host",
            "cartesia:dev",
            "cartesia_dev",
            "cartesia:karan",
            "cartesia_karan",
            "cartesia:calm",
            "cartesia_calm",
            "cartesia:hindi_calm",
        ],
    },
    "cartesia:hindi_female": {
        "id": "4459a9a5-69d6-4680-b970-e13dc51845b6",
        "name": "Siya",
        "lang": "hi",
        "gender": "female",
        "style": "Heartfelt, captivating & expressive female narrative voice",
        "fallback_kokoro": "hf_alpha",
        "aliases": [
            "cartesia_female",
            "cartesia_hindi_female",
            "cartesia:siya",
            "cartesia_siya",
            "cartesia:pooja",
            "cartesia_pooja",
        ],
    },
    "cartesia:calm_female": {
        "id": "209d9a43-03eb-40d8-a7b7-51a6d54c052f",
        "name": "Anita",
        "lang": "hi",
        "gender": "female",
        "style": "Soft-spoken, calming & gentle adult female (ghost whisper / meditation)",
        "fallback_kokoro": "hf_beta",
        "aliases": [
            "cartesia_calm_female",
            "cartesia:anita",
            "cartesia_anita",
            "cartesia:soft",
            "cartesia_soft",
            "cartesia:hindi_soft",
        ],
    },
    "cartesia:hinglish": {
        "id": "95d51f79-c397-46f9-b49a-23763d3eaa2d",
        "name": "Arushi",
        "lang": "hi",
        "gender": "female",
        "style": "Modern urban Hinglish female (viral reels / tech / youth content)",
        "fallback_kokoro": "hf_beta",
        "aliases": [
            "cartesia_hinglish",
            "cartesia:arushi",
            "cartesia_arushi",
            "cartesia:viral",
            "cartesia_viral",
            "cartesia:ananya",
            "cartesia_ananya",
        ],
    },
    "cartesia:horror_narrator": {
        "id": "20e68f5c-08e5-42d0-8e9b-6e716fd1ae66",
        "name": "Vivek",
        "lang": "hi",
        "gender": "male",
        "style": "Eerie, dark & suspenseful (dark folklore / ghosts)",
        "fallback_kokoro": "hm_omega",
        "aliases": [
            "cartesia_horror_narrator",
            "cartesia:whisper",
            "cartesia_whisper",
            "cartesia:eerie",
            "cartesia_eerie",
            "cartesia:shadow",
            "cartesia_shadow",
        ],
    },
}

# AI4Bharat "Indic Parler-TTS" description-conditioned multilingual voice catalog.
#
# Unlike the zero-shot cloner it replaces, Parler-TTS has no reference clip: the voice is
# steered entirely by a natural-language *description* of the speaker (gender, timbre,
# pace, recording quality). That is far more stable than ref-audio alignment and removes
# the need to ship or upload reference voices.
INDIC_PARLER_MODEL = os.environ.get("INDIC_PARLER_MODEL", "").strip()
# Ordered candidates. The official AI4Bharat repo is GATED (401 without a token that has
# accepted its terms), so ungated mirrors are listed as fallbacks. Set INDIC_PARLER_MODEL
# to pin one repo explicitly (which disables the chain).
INDIC_PARLER_MODEL_FALLBACKS: tuple[str, ...] = (
    "ai4bharat/indic-parler-tts",
    "RXD03/indic-parler-tts",
    "Yogi5/indic-parler-tts-backup",
    "naklitechie/indic-parler-tts",
)


def indic_parler_model_candidates() -> list[str]:
    """Repos to try, in order, when loading Indic Parler-TTS."""
    if INDIC_PARLER_MODEL:
        return [INDIC_PARLER_MODEL]
    return list(INDIC_PARLER_MODEL_FALLBACKS)


# The production default narrator is a reference-conditioned IndicF5 voice. Every scene
# uses it at one steady pace; per-beat/emotion speed modulation is skipped so tone and
# pacing do not drift. Set DEFAULT_VOICE to retain a legacy Indic Parler narrator.
DEFAULT_INDIC_VOICE = os.environ.get("DEFAULT_VOICE", "indicf5_aarav")


INDIC_PARLER_VOICES: dict[str, dict] = {
    "indic_parler_aarav": {
        "id": "indic_parler_aarav",
        "name": "Aarav",
        "lang": "hi",
        "lang_name": "Hindi / Hinglish",
        "gender": "male",
        "style": "Deep, resonant & authoritative (ideal for history, mythology, epics)",
        "description": (
            "A male speaker with a deep, resonant and authoritative voice speaks only in "
            "pure Hindi with a clear native Indian accent at a moderate pace in a very "
            "clear, close-sounding recording with no background noise."
        ),
        "fallback_kokoro": "hm_omega",
        "aliases": [
            "indic_parler", "indic_parler_male", "indic_aarav",
        ],
    },
    "indic_parler_kabir": {
        "id": "indic_parler_kabir",
        "name": "Kabir",
        "lang": "hi",
        "lang_name": "Hindi / Hinglish",
        "gender": "male",
        "style": "Calm, natural & conversational (ideal for documentary, explainer)",
        "description": (
            "A male speaker with a calm, natural and conversational voice speaks only in "
            "pure Hindi with a clear native Indian accent at a moderate pace in a very "
            "clear, close-sounding recording with no background noise."
        ),
        "fallback_kokoro": "hm_psi",
        "aliases": [
            "indic_parler_calm", "indic_kabir",
        ],
    },
    "indic_parler_rohan": {
        "id": "indic_parler_rohan",
        "name": "Rohan",
        "lang": "hi",
        "lang_name": "Hindi / Hinglish",
        "gender": "male",
        "style": "Direct & serious (ideal for mysteries)",
        "description": (
            "A male speaker with a direct, serious and slightly dry voice speaks only in "
            "pure Hindi with a clear native Indian accent at a moderate pace in a very "
            "clear, close-sounding recording with no background noise."
        ),
        "fallback_kokoro": "hm_omega",
        "aliases": [
            "indic_parler_mystery", "indic_rohan",
        ],
    },
    "indic_parler_priya": {
        "id": "indic_parler_priya",
        "name": "Priya",
        "lang": "hi",
        "lang_name": "Hindi / Hinglish",
        "gender": "female",
        "style": "Warm, expressive & engaging (ideal for drama, storytelling)",
        "description": (
            "A female speaker with a warm, expressive and engaging voice speaks only in "
            "pure Hindi with a clear native Indian accent at a moderate pace in a very "
            "clear, close-sounding recording with no background noise."
        ),
        "fallback_kokoro": "hf_alpha",
        "aliases": [
            "indic_parler_female", "indic_priya",
        ],
    },
    "indic_parler_ananya": {
        "id": "indic_parler_ananya",
        "name": "Ananya",
        "lang": "hi",
        "lang_name": "Hindi / Hinglish",
        "gender": "female",
        "style": "Soft, cheerful & pleasant (ideal for lifestyle, casual)",
        "description": (
            "A female speaker with a soft, cheerful and pleasant voice speaks only in "
            "pure Hindi with a clear native Indian accent at a moderate pace in a very "
            "clear, close-sounding recording with no background noise."
        ),
        "fallback_kokoro": "hf_beta",
        "aliases": [
            "indic_parler_soft", "indic_ananya",
        ],
    },
}


# ---------------------------------------------------------------------------
# IndicF5 (AI4Bharat) reference-conditioned voices.
#
# Production uses the remote RTX 4090 server. The Python 3.11 subprocess runtime in
# examples/indicf5 remains the deliberate local/--no-vast fallback because its pinned
# dependency stack must not be imported into this project's primary Python environment.
# ---------------------------------------------------------------------------
INDICF5_LOCAL_VOICES: dict[str, dict] = {
    "indicf5_aarav": {"id": "indicf5_aarav", "reference": "aarav", "lang": "hi", "fallback_kokoro": "hm_omega"},
    "indicf5_kabir": {"id": "indicf5_kabir", "reference": "kabir", "lang": "hi", "fallback_kokoro": "hm_psi"},
    "indicf5_rohan": {"id": "indicf5_rohan", "reference": "rohan", "lang": "hi", "fallback_kokoro": "hm_omega"},
    "indicf5_priya": {"id": "indicf5_priya", "reference": "priya", "lang": "hi", "fallback_kokoro": "hf_alpha"},
    "indicf5_ananya": {"id": "indicf5_ananya", "reference": "ananya", "lang": "hi", "fallback_kokoro": "hf_beta"},
}

INDICF5_EXAMPLE_DIR = ROOT / "examples" / "indicf5"
INDICF5_VOICES_DIR = INDICF5_EXAMPLE_DIR / "voices"
INDICF5_BATCH_SCRIPT = INDICF5_EXAMPLE_DIR / "indicf5_batch.py"
INDICF5_VENV_PYTHON = INDICF5_EXAMPLE_DIR / ".venv-indicf5" / "Scripts" / "python.exe"
# The batch child publishes its result in this file inside the run's temp dir. It cannot
# use stdout: f5_tts and its dependencies print progress there (the jieba banner,
# "Download Vocos ...", "vocab :", "token :"), which would precede the JSON document.
INDICF5_RESULT_FILE = "_result.json"
INDICF5_RESULT_PREFIX = "INDICF5_RESULT "

_INDICF5_BATCH_USABLE = True

_INDIC_ENGINE = None


def resample_audio(audio, src_rate: int, dst_rate: int):
    """Resample mono float32 audio to ``dst_rate`` (torchaudio, numpy fallback).

    The narration pipeline assumes a single 24 kHz sample rate end to end, but Parler-TTS
    natively emits 44.1 kHz, so everything it returns is brought back to ``SAMPLE_RATE``.
    """
    import numpy as np

    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if src_rate == dst_rate or x.size == 0:
        return x
    try:
        import torch
        import torchaudio.functional as AF

        return AF.resample(torch.from_numpy(x), src_rate, dst_rate).numpy().astype(np.float32)
    except Exception:  # noqa: BLE001 - linear interpolation is a safe last resort
        n = max(1, int(round(x.size * dst_rate / src_rate)))
        return np.interp(np.linspace(0.0, x.size - 1, n), np.arange(x.size), x).astype(np.float32)


def _pace_word(speed: float) -> str:
    """Map a numeric speed multiplier onto the pace phrase Parler-TTS understands."""
    if speed <= 0.92:
        return "a slow, deliberate pace"
    if speed >= 1.12:
        return "a fast, urgent pace"
    return "a moderate pace"


class IndicParlerEngine:
    """AI4Bharat Indic Parler-TTS: description-conditioned multilingual TTS.

    Unlike the zero-shot cloner this replaces, Parler-TTS needs no reference clip -- the
    voice is a natural-language description of the speaker. It is loaded in float32, which
    keeps the decoder numerically stable and avoids the fp16 NaN class of bug entirely.
    """

    def __init__(self) -> None:
        import torch
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float32
        last_err: Exception | None = None
        model = None
        loaded_repo = ""
        for repo_id in indic_parler_model_candidates():
            try:
                print(f"[Indic Parler-TTS] Loading {repo_id} on {self.device}...")
                model = ParlerTTSForConditionalGeneration.from_pretrained(
                    repo_id, torch_dtype=self.dtype
                )
                loaded_repo = repo_id
                break
            except Exception as err:  # noqa: BLE001 - gated/offline repos fall through
                last_err = err
                print(
                    f"[Indic Parler-TTS] {repo_id} unavailable "
                    f"({type(err).__name__}: {str(err)[:120]}); trying next candidate...",
                    file=sys.stderr,
                )
        if model is None:
            raise RuntimeError(f"no Indic Parler-TTS model could be loaded: {last_err}")
        self.model = model.to(self.device).eval()
        self.description_tokenizer = AutoTokenizer.from_pretrained(
            self.model.config.text_encoder._name_or_path
        )
        self.prompt_tokenizer = AutoTokenizer.from_pretrained(loaded_repo)
        self.sample_rate = int(getattr(self.model.config, "sampling_rate", None) or 44100)
        # Parler-TTS samples a fresh speaker at generation time (do_sample=True in its
        # generation_config). Greedy decoding (the default) plus a fixed seed pins ONE
        # speaker across every sentence. Set INDIC_PARLER_GREEDY=0 to restore sampling.
        self.seed = int(os.environ.get("INDIC_PARLER_SEED", "1234"))
        # Greedy decoding is OFF by default: Parler-TTS is trained for sampling and greedy
        # degenerates to near-silent output on some segments. Opt in with INDIC_PARLER_GREEDY=1.
        self.do_sample = os.environ.get("INDIC_PARLER_GREEDY", "0") != "1"
        print(
            f"[Indic Parler-TTS] Model loaded (native {self.sample_rate} Hz -> {SAMPLE_RATE} Hz)."
        )

    def synthesize(self, text: str, description: str, speed: float = 1.0, seed: int | None = None):
        import numpy as np
        import torch

        input_ids = self.description_tokenizer(
            description.strip(), return_tensors="pt"
        ).input_ids.to(self.device)
        prompt_ids = self.prompt_tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        # Pin the speaker: seed + greedy decoding. Without both, each sentence is a
        # different random voice and the narration sounds like several people in a row.
        use_seed = self.seed if seed is None else int(seed)
        torch.manual_seed(use_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(use_seed)
        with torch.no_grad():
            generation = self.model.generate(
                input_ids=input_ids,
                prompt_input_ids=prompt_ids,
                do_sample=self.do_sample,
                num_beams=1,
            )
        wav = generation.cpu().numpy().squeeze().astype(np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=0)
        return resample_audio(wav, self.sample_rate, SAMPLE_RATE)


def synthesize_indic_parler(text: str, voice_key: str, speed: float = 1.0) -> np.ndarray | None:
    """Synthesise one sentence with Indic Parler-TTS (Vast.ai GPU first, else local CPU)."""
    global _INDIC_ENGINE
    vinfo = INDIC_PARLER_VOICES.get(voice_key.lower())
    if not vinfo:
        for vk, vi in INDIC_PARLER_VOICES.items():
            if voice_key.lower() == vk.lower() or voice_key.lower() in [a.lower() for a in vi.get("aliases", [])]:
                vinfo = vi
                break
    if not vinfo:
        return None

    try:
        norm_text = normalize_hindi_phonetics(text)
        # Served from the one-shot Vast.ai batch pre-fetch when it already produced this
        # exact (text, voice, speed) segment.
        cache_key = (norm_text, voice_key.lower(), round(speed, 3))
        cached = _VAST_AUDIO_CACHE.get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]
        # The GPU already produced this segment on an earlier run. Serving it from disk
        # costs nothing and is the whole point of persisting batch results.
        disk_cached = read_gpu_cache(norm_text, voice_key, speed)
        if disk_cached is not None:
            _VAST_AUDIO_CACHE[cache_key] = disk_cached
            return disk_cached  # type: ignore[return-value]
        if not is_devanagari(norm_text):
            print(
                "[Indic Parler-TTS Notice] Input text is in Roman script. Indic Parler-TTS "
                "is trained on native Devanagari script; provide Devanagari for best results.",
                file=sys.stderr,
            )

        # Parler-TTS has no speed knob, so the requested pace is folded into the description.
        description = vinfo["description"].replace("a moderate pace", _pace_word(speed))

        # Check if Vast.ai remote GPU is configured
        vast_url = os.environ.get("VAST_SERVER_URL", "").strip()
        use_vast = os.environ.get("USE_VAST", "0") == "1" or os.environ.get("TTS_PROVIDER", "").lower() == "vast"
        if (use_vast or vast_url) and vast_url:
            last_err: Exception | None = None
            for attempt in range(1, VAST_AUDIO_ATTEMPTS + 1):
                try:
                    from vast_manager import VastManager
                    import io
                    import soundfile as sf
                    mgr = VastManager(server_url=vast_url or None)
                    audio_bytes, dur = mgr.synthesize_tts(
                        text=norm_text,
                        voice=voice_key,
                        speed=speed,
                    )
                    wav_np, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
                    wav_np = resample_audio(wav_np, sr or SAMPLE_RATE, SAMPLE_RATE)
                    if audio_is_bad(wav_np):
                        # A structurally-valid but silent/DC payload. Treat it as a failed
                        # attempt (never cache it) so the retry/fallback logic runs.
                        raise RuntimeError(
                            "remote Indic Parler-TTS returned silent/non-finite audio (corrupt WAV)"
                        )
                    write_gpu_cache(norm_text, voice_key, speed, audio_bytes)
                    _VAST_AUDIO_CACHE[cache_key] = wav_np
                    return wav_np
                except Exception as vast_err:
                    last_err = vast_err
                    print(
                        f"[Indic Parler Remote] attempt {attempt}/{VAST_AUDIO_ATTEMPTS} failed: "
                        f"{type(vast_err).__name__}: {vast_err}",
                        file=sys.stderr,
                    )
                    if is_permanent_remote_tts_error(vast_err):
                        print(
                            "[Indic Parler Remote] that is an environment fault on the remote "
                            "instance, not a transient one; retrying cannot change it. "
                            "Repair it (python scripts/vast_manager.py repair <instance-id>) "
                            "or destroy it and rent a fresh GPU.",
                            file=sys.stderr,
                        )
                        break
                    if attempt < VAST_AUDIO_ATTEMPTS:
                        time.sleep(VAST_AUDIO_RETRY_WAIT)
            if os.environ.get("REQUIRE_VAST_TTS", "0") == "1":
                raise RuntimeError(f"Required Vast.ai Indic Parler-TTS synthesis failed: {last_err}") from last_err
            print(
                f"[Indic Parler Remote Vast Warning] Remote synthesis failed after "
                f"{VAST_AUDIO_ATTEMPTS} attempts ({last_err}), falling back to local...",
                file=sys.stderr,
            )
        elif use_vast and not vast_url:
            print("[Indic Parler-TTS Warning] Vast.ai requested but VAST_SERVER_URL is empty, falling back to local...", file=sys.stderr)

        # Local engine (CPU or a local GPU). No reference files are needed for Parler-TTS.
        if _INDIC_ENGINE is None:
            _INDIC_ENGINE = IndicParlerEngine()
        audio = _INDIC_ENGINE.synthesize(norm_text, description, speed=speed)
        return audio
    except Exception as e:
        if os.environ.get("REQUIRE_VAST_TTS", "0") == "1":
            raise
        print(f"[Indic Parler-TTS Warning] generation failed ({e}), falling back to Kokoro...", file=sys.stderr)
        return None


_CARTESIA_WARNED = False


def synthesize_cartesia(
    text: str,
    voice_key_or_id: str,
    speed: float = 1.0,
    language: str = "hi",
    fallback_voice: str = "hf_beta",
) -> np.ndarray | None:
    """Synthesise text via Cartesia Sonic REST API with automatic Kokoro fallback."""
    global _CARTESIA_WARNED
    import os
    import json
    import urllib.request
    import urllib.error
    import numpy as np

    api_key = os.getenv("CARTESIA_API_KEY", "").strip()
    if not api_key:
        if not _CARTESIA_WARNED:
            print(
                f"[Cartesia TTS Notice] 'CARTESIA_API_KEY' not set in environment. "
                f"Automatically using local Kokoro-82M ({fallback_voice}).",
                file=sys.stderr,
            )
            _CARTESIA_WARNED = True
        return None

    # Resolve Cartesia voice ID
    voice_id = voice_key_or_id
    if voice_key_or_id in CARTESIA_VOICES:
        voice_id = CARTESIA_VOICES[voice_key_or_id]["id"]
    else:
        for ckey, cinfo in CARTESIA_VOICES.items():
            if voice_key_or_id.lower() == ckey.lower() or voice_key_or_id.lower() in [
                a.lower() for a in cinfo.get("aliases", [])
            ]:
                voice_id = cinfo["id"]
                break

    url = "https://api.cartesia.ai/tts/bytes"
    payload = {
        "model_id": "sonic-preview",
        "transcript": text,
        "voice": {
            "mode": "id",
            "id": voice_id,
        },
        "output_format": {
            "container": "raw",
            "encoding": "pcm_s16le",
            "sample_rate": 24000,
        },
        "language": language,
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Cartesia-Version": "2024-06-10",
            "X-API-Key": api_key,
            "Content-Type": "application/json",
            "User-Agent": "Tube2DScript/1.0",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw_bytes = resp.read()
            if not raw_bytes:
                return None
            pcm_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
            return (pcm_int16.astype(np.float32) / 32768.0)
    except urllib.error.HTTPError as e:
        if not _CARTESIA_WARNED:
            reason = (
                "Expired/Invalid API Token (401)"
                if e.code == 401
                else "Account Balance / Credits Depleted (402)"
                if e.code == 402
                else "Rate limit reached (429)"
                if e.code == 429
                else f"HTTP {e.code}"
            )
            print(
                f"[Cartesia TTS Notice] Cartesia API unavailable ({reason}). "
                f"Automatically falling back to local Kokoro-82M ({fallback_voice})...",
                file=sys.stderr,
            )
            _CARTESIA_WARNED = True
        return None
    except Exception as e:
        if not _CARTESIA_WARNED:
            print(
                f"[Cartesia TTS Notice] Cartesia request failed ({e}). "
                f"Automatically falling back to local Kokoro-82M ({fallback_voice})...",
                file=sys.stderr,
            )
            _CARTESIA_WARNED = True
        return None


VOICE_ALIASES: dict[str, str] = {}
for vid, vinfo in VOICE_CATALOG.items():
    VOICE_ALIASES[vid.lower()] = vid
    for alias in vinfo.get("aliases", []):
        VOICE_ALIASES[alias.lower().replace("-", "_")] = vid

for ivid, ivinfo in INDIC_PARLER_VOICES.items():
    VOICE_ALIASES[ivid.lower()] = ivid
    for alias in ivinfo.get("aliases", []):
        VOICE_ALIASES[alias.lower().replace("-", "_")] = ivid

for fivid in INDICF5_LOCAL_VOICES:
    VOICE_ALIASES[fivid.lower()] = fivid
VOICE_ALIASES["indicf5"] = "indicf5_aarav"

for cvid, cvinfo in CARTESIA_VOICES.items():
    VOICE_ALIASES[cvid.lower()] = cvid
    for alias in cvinfo.get("aliases", []):
        VOICE_ALIASES[alias.lower().replace("-", "_")] = cvid

# Backwards compatibility map
EN_VOICES = {
    vid: f"{vinfo['lang_name']} {vinfo['gender']}, {vinfo['style']}"
    for vid, vinfo in VOICE_CATALOG.items()
    if vinfo["lang"].startswith("en")
}

# Atmospheric & background character voice presets (blends of Kokoro neural embeddings)
# Hindi presets use 100% pure native Hindi embeddings to eliminate American/English acoustic accent drift.
BACKGROUND_VOICE_PRESETS: dict[str, str] = {
    # Top Hindi Narration & Storytelling Presets (100% Native Accent)
    "hindi_cinematic": "hm_omega",
    "hindi_cinematic_male": "hm_omega",
    "hindi_documentary": "hm_psi",
    "hindi_doc_male": "hm_psi",
    "hindi_dynamic": "hm_omega",
    "hindi_motivational": "hm_omega",
    "hindi_smooth": "hm_omega,hm_psi",
    "hindi_story": "hm_omega,hm_psi",
    "hindi_storyteller": "hf_alpha",
    "hindi_female_story": "hf_alpha",
    "hindi_viral": "hf_beta",
    "hindi_female_bright": "hf_beta",
    "hindi_female_doc": "hf_alpha,hf_beta",
    "hindi_soft_whisper": "hf_beta",
    # Horror / Paranormal / Spectral (Native Indian Textures)
    "ghost_whisper": "hf_alpha,hf_beta",
    "spectral_whisper": "hf_beta",
    "demon_voice": "hm_omega",
    "demon_bass": "hm_omega",
    "asura_voice": "hm_omega",
    "creepy_witch": "hf_alpha",
    "witch_voice": "hf_alpha",
    "chudail_voice": "hf_alpha",
    "ancient_spirit": "hm_omega,hm_psi",
    "aatma_voice": "hm_omega,hm_psi",
    "spectral_child": "hf_beta",
    "child_ghost": "hf_beta",
    "bhootia_bachha": "hf_beta",
    "eerie_narrator": "hm_omega",
    "horror_narrator": "hm_omega",
    "shadow_entity": "hm_omega,hm_psi",
    "chhaya_voice": "hm_omega,hm_psi",
    "mystic_oracle": "hm_omega",
    "tantrik_voice": "hm_omega",
    # Broadcast / Radio / Sci-Fi / Documentary
    "radio_dispatch": "bm_george,am_echo",
    "police_radio": "bm_george,am_echo",
    "cctv_voice": "bm_george,am_echo",
    "news_reporter": "hm_psi",
    "anchor_voice": "hm_psi",
    "ai_system": "af_alloy,af_sarah",
    "cyber_voice": "af_alloy,am_echo",
    "ancient_guru": "hm_omega",
}

for preset_name, preset_target in BACKGROUND_VOICE_PRESETS.items():
    VOICE_ALIASES[preset_name.lower()] = preset_target


def resolve_voice(
    voice_input: str | None,
    default_lang: str = "hi",
    genre: str | None = None,
) -> str:
    """Resolve a voice ID, alias, or multi-voice blend expression.

    Supports:
      - Direct IDs: 'hm_omega', 'af_heart', 'bm_george'
      - Friendly aliases: 'hindi_male', 'hindi_female', 'documentary', 'deep_male', 'anime'
      - Multi-voice blending: 'af_heart+af_bella', 'af_heart+af_bella', 'hindi_male+hindi_female'
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

    # Normalize blend keywords like 'blend', '_blend_', '+', etc.
    raw_clean = voice_input.replace("_blend_", "+").replace("blend", "+").replace("+", ",")
    normalized = raw_clean.replace("hm_iomega", "hm_omega").replace("iomega", "omega")
    parts = [p.strip() for p in normalized.split(",") if p.strip()]

    resolved_parts = []
    for part in parts:
        key = part.lower().replace("-", "_")
        resolved_parts.append(VOICE_ALIASES.get(key, part))

    return ",".join(resolved_parts)


def list_voices(lang_filter: str | None = None) -> list[dict]:
    """Return structured voice catalog items, optionally filtered by language."""
    items = []
    for ivid, ivinfo in INDIC_PARLER_VOICES.items():
        if lang_filter:
            filt = lang_filter.lower().strip()
            if filt in ("hi", "hindi", "hinglish"):
                items.append(dict(ivinfo))
        else:
            items.append(dict(ivinfo))
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
        "INDIC PARLER-TTS HINDI VOICES (AI4Bharat, description-conditioned multilingual TTS)",
        "=" * 74,
        "  * Native Hindi pronunciation steered by a natural-language speaker description.",
    ]
    for ikey, iinfo in INDIC_PARLER_VOICES.items():
        aliases_str = f" [Aliases: {', '.join(iinfo['aliases'])}]" if iinfo.get("aliases") else ""
        lines.append(f"  {ikey:<24} : {iinfo['name']:<15} | {iinfo['gender']:<6} | {iinfo['style']}")
        lines.append(f"                           Fallback: {iinfo['fallback_kokoro']}{aliases_str}")

    lines.extend([
        "\n" + "=" * 74,
        "CARTESIA SONIC HINDI VOICES (Ultra-Realistic Neural Cloud TTS)",
        "=" * 74,
        "  * Uses CARTESIA_API_KEY if present; auto falls back to local Kokoro if expired/unpaid.",
    ])
    for ckey, cinfo in CARTESIA_VOICES.items():
        aliases_str = f" [Aliases: {', '.join(cinfo['aliases'])}]" if cinfo.get("aliases") else ""
        lines.append(f"  {ckey:<24} : {cinfo['name']:<15} | {cinfo['gender']:<6} | {cinfo['style']}")
        lines.append(f"                           Fallback: {cinfo['fallback_kokoro']}{aliases_str}")

    lines.extend([
        "\n" + "=" * 74,
        "KOKORO-82M VOICE CATALOG (Local Neural TTS, Zero API Cost)",
        "=" * 74,
    ])
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
        "     --voice hindi_male                  (maps to hm_omega)",
        "     --voice hindi_female                (maps to hf_alpha)",
        "     --voice documentary                 (maps to bm_george - BBC documentary style)",
        "     --voice deep_male                   (maps to am_onyx - deep bass trailer voice)",
        "     --voice anime_female                (maps to jf_nezumi)",
        "",
        "3. Multi-Character Dialogue:",
        "     In script.json, assign voices per-character or per-scene lines:",
        "       \"characters\": {",
        "           \"hero\": {\"name\": \"Goku\", \"voice\": \"hm_omega\"},",
        "           \"villain\": {\"name\": \"Shadow\", \"voice\": \"am_onyx\"}",
        "       }",
    ])
    return "\n".join(lines)


def generate_voice_samples() -> int:
    """Synthesise and export short preview audio samples for Kokoro Hindi voices."""
    import soundfile as sf

    samples_dir = PUBLIC / "audio" / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    engine = Voice()
    sample_text = "नमस्ते, यह कोकोरो हिंदी हिंग्लिश आवाज़ का ऑडियो सैंपल है।"

    sample_voices = [
        ("kokoro_aarav", "hm_omega", "Kokoro Aarav (Deep / Resonant)"),
        ("kokoro_kabir", "hm_psi", "Kokoro Kabir (Calm / Documentary)"),
        ("kokoro_priya", "hf_alpha", "Kokoro Priya (Warm / Expressive)"),
        ("kokoro_ananya", "hf_beta", "Kokoro Ananya (Soft / Cheerful)"),
    ]

    print("=" * 70)
    print("GENERATING HINDI / HINGLISH VOICE SAMPLES (Kokoro)")
    print("=" * 70)
    for sample_name, vid, desc in sample_voices:
        out_file = samples_dir / f"{sample_name}.wav"
        print(f"  Generating sample for {sample_name} ({desc})...")
        audio, _words = synthesise_line(engine, sample_text, vid, speed=1.0, pause=0.28, master=DEFAULT_MASTER)
        sf.write(str(out_file), audio, SAMPLE_RATE, subtype="PCM_16")
        print(f"    -> Saved: {out_file.relative_to(ROOT)}")

    print("\nAll audio samples generated successfully in 'public/audio/samples/'.")
    return 0


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
    "sentencePauseSeconds": 0.25,
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
    "padAfter": 0.04,
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
    """Split narration into sentences (and clauses), keeping common abbreviations intact.

    Handles the Devanagari danda (।) and double danda (॥) as well as Latin
    punctuation, since Hindi narration uses them exclusively.
    Also splits at commas so that each clause gets a small explicit pause.
    """
    text = " ".join(text.split())
    if not text:
        return []

    sentences: list[str] = []
    for part in re.split(r"(?<=[.!?।॥,])\s+", text):
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


# Narration text already returned by a remote Vast.ai batch call, keyed by
# (spoken text, voice, speed). Filled by prebuild_vast_batch() and consumed by
# synthesize_indic_parler(), so the per-sentence path costs nothing extra.
_VAST_AUDIO_CACHE: dict[tuple[str, str, float], object] = {}

# Content-addressed local store for audio the GPU has already produced. Each segment is
# written here the moment it arrives, so a crash later in the run cannot throw away GPU
# time that has been paid for: the next run reads it straight off disk.
GPU_CACHE_DIR = ROOT / "out" / "gpu_audio_cache"

# Retry policy for a remote synthesis call. A single dropped connection used to fall
# straight through to local CPU Indic Parler-TTS, which is hours of work for a long script.
VAST_AUDIO_ATTEMPTS = int(os.environ.get("VAST_AUDIO_ATTEMPTS", "3"))
VAST_AUDIO_RETRY_WAIT = float(os.environ.get("VAST_AUDIO_RETRY_WAIT", "15"))

# Failures no amount of retrying can fix: the remote interpreter itself is broken, so the
# next attempt asks the same broken environment the same question. Observed as an HTTP 500
# whose body reads "ImportError: pyarrow requires NumPy 2.0 or newer, found 1.26.4".
PERMANENT_REMOTE_TTS_ERRORS: tuple[str, ...] = (
    "ImportError",
    "ModuleNotFoundError",
    "No module named",
    "cannot import name",
    "requires NumPy",
)


def is_permanent_remote_tts_error(err: object) -> bool:
    """True when retrying a remote Indic Parler-TTS request cannot possibly succeed.

    A dependency or ABI problem inside the server's environment is stable for the life of
    the instance, so the retry loop only adds waiting on top of an already-failed run. It
    has to be repaired (scripts/vast_manager.py repair <instance-id>) or replaced.
    """
    text = f"{type(err).__name__}: {err}"
    return any(marker in text for marker in PERMANENT_REMOTE_TTS_ERRORS)


def audio_is_bad(wav) -> bool:
    """True when a waveform is empty, non-finite, silent or pure DC.

    Guards the whole narration path. The Vast.ai Indic Parler-TTS server can return a WAV that is
    structurally valid but made of constant -32768 (0x8000) -- a NaN encoded by libsndfile
    as DC. Without this check that silent audio gets cached on disk and, because the cache
    is content-addressed, served again by every later run.
    """
    import numpy as np

    x = np.asarray(wav, dtype=np.float64).reshape(-1)
    if x.size == 0 or not np.all(np.isfinite(x)):
        return True
    if float(np.max(np.abs(x))) < 0.02:  # near-silence / blank output (~-34 dBFS peak)
        return True
    if float(np.sqrt(np.mean(x ** 2))) < 0.01:  # far too quiet to be speech
        return True
    if np.std(x) < 1e-5:  # constant / pure DC
        return True
    return False


def purge_audio_cache() -> int:
    """Delete every cached GPU segment and scene sidecar.

    A corrupt segment is content-addressed, so it would be served again by every later run.
    Purging (or bumping AUDIO_PIPELINE_VERSION) is the only way to force a clean rebuild.
    """
    removed = 0
    for path in list(GPU_CACHE_DIR.glob("*.wav")) + list(AUDIO_DIR.glob("*.json")):
        try:
            path.unlink()
            removed += 1
        except OSError:
            pass
    print(
        f"purged audio cache: {removed} file(s) removed from {GPU_CACHE_DIR} and {AUDIO_DIR}"
    )
    return 0


def gpu_cache_key(text: str, voice: str, speed: float) -> str:
    """Stable hash of everything that changes the audio, and nothing else."""
    payload = [AUDIO_PIPELINE_VERSION, voice.lower(), round(speed, 3), text]
    return hashlib.sha256(json.dumps(payload).encode("utf-8")).hexdigest()


def gpu_cache_path(text: str, voice: str, speed: float) -> Path:
    return GPU_CACHE_DIR / f"{gpu_cache_key(text, voice, speed)}.wav"


def read_gpu_cache(text: str, voice: str, speed: float):
    """Return the cached float32 waveform for this segment, or None."""
    path = gpu_cache_path(text, voice, speed)
    if not path.exists():
        return None
    try:
        import soundfile as sf

        wav_np, _sr = sf.read(str(path), dtype="float32")
        if audio_is_bad(wav_np):
            print(
                f"[Indic Parler Cache Warning] Discarding silent/non-finite cached "
                f"segment {path.name}",
                file=sys.stderr,
            )
            path.unlink(missing_ok=True)
            return None
        return wav_np
    except Exception as err:  # noqa: BLE001 - a corrupt entry must never be fatal
        print(f"[Indic Parler Cache Warning] Ignoring unreadable {path.name}: {err}", file=sys.stderr)
        return None


def write_gpu_cache(text: str, voice: str, speed: float, wav_bytes: bytes) -> None:
    """Persist one GPU segment. Best effort: caching must never break synthesis.

    A structurally-valid but silent/DC segment is refused here so corrupt audio can never
    enter the on-disk cache and be reused by a later run.
    """
    try:
        import io

        import soundfile as sf

        wav_np, _sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    except Exception as err:  # noqa: BLE001
        print(f"[Indic Parler Cache Warning] Refusing to cache undecodable segment: {err}", file=sys.stderr)
        return
    if audio_is_bad(wav_np):
        print(
            f"[Indic Parler Cache Warning] Refusing to cache silent/non-finite segment "
            f"({voice} @ {round(speed, 3)}x)",
            file=sys.stderr,
        )
        return
    try:
        path = gpu_cache_path(text, voice, speed)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(wav_bytes)
    except Exception as err:  # noqa: BLE001
        print(f"[Indic Parler Cache Warning] Could not persist GPU audio: {err}", file=sys.stderr)


def prepare_tts_text(text: str) -> str:
    """Normalise text exactly the way every engine receives it before synthesis.

    Shared by ``Voice.say`` and the Vast.ai batch pre-fetch so their cache keys agree.
    """
    clean_text = re.sub(r"[;:—–\-\"'\(\)\[\]\{\}…]+", " ", text)
    clean_text = " ".join(clean_text.split())
    if not clean_text:
        clean_text = text
    # Conversational Hindi phonetics (e.g. 'वह' -> 'वो', numbers to Hindi words)
    return normalize_hindi_phonetics(clean_text)


def prepare_remote_text(sentence: str) -> str:
    """Exactly the string the remote Indic Parler-TTS endpoint is asked to speak."""
    return normalize_hindi_phonetics(prepare_tts_text(sentence))


def is_indic_voice(voice: str) -> bool:
    """True for the Indic narration families (Indic Parler-TTS and local IndicF5)."""
    primary = (voice or "").strip().lower().split(",")[0].strip()
    if primary in INDICF5_LOCAL_VOICES:
        return True
    if primary in INDIC_PARLER_VOICES:
        return True
    if primary.startswith(("indic_parler", "indic_")):
        return True
    return any(primary in [a.lower() for a in vi.get("aliases", [])] for vi in INDIC_PARLER_VOICES.values())


def line_voice_and_speed(
    scene: dict,
    line: dict,
    characters: dict,
    voice_default: str,
    base_speed: float,
    dynamic: bool,
) -> tuple[str, float, dict]:
    """Resolve the voice, speaking rate, and character record for one narration line.

    Shared by the synthesis loop and the Vast.ai batch pre-fetch so a pre-built cache
    key always matches the request the loop would otherwise have made.
    """
    character_id = line.get("character")
    character = (characters.get(character_id) or {}) if character_id else {}
    line_raw_voice = line.get("voice") or character.get("voice") or scene.get("voice") or voice_default
    voice = resolve_voice(line_raw_voice)

    base = float(character.get("speed") or scene.get("speed") or base_speed)
    if not dynamic or is_indic_voice(voice):
        # The Indic Parler-TTS narrator is one consistent voice: beat/emotion speed
        # modulation is deliberately skipped so tone and pacing never drift between scenes.
        return voice, base, character

    beat_name = str(scene.get("beat") or scene.get("movement") or "").lower()
    scene_emotion = str(
        scene.get("emotion") or (scene.get("cast") and scene["cast"][0].get("emotion")) or ""
    ).lower()
    beat_factor = BEAT_SPEED_MULTIPLIERS.get(beat_name, 1.0)
    emotion_factor = EMOTION_SPEED_MULTIPLIERS.get(scene_emotion, 1.0)
    # Combine base multiplier with emotional inflection (bounded between 0.6x and 1.8x)
    return voice, round(max(0.6, min(1.8, base * beat_factor * emotion_factor)), 3), character


def remote_tts_enabled() -> bool:
    """Whether narration is explicitly configured to use the running Vast server."""
    return bool(os.environ.get("VAST_SERVER_URL", "").strip()) and (
        os.environ.get("USE_VAST", "0") == "1"
        or os.environ.get("TTS_PROVIDER", "").lower() == "vast"
    )


def prebuild_vast_batch(
    scenes: list[dict],
    characters: dict,
    voice_default: str,
    base_speed: float,
    dynamic_speed: bool,
) -> int:
    """Synthesise every narration sentence in one Vast.ai round trip.

    Returns the number of segments cached. Best effort: any failure leaves the cache
    empty and the normal per-sentence path still runs.
    """
    vast_url = os.environ.get("VAST_SERVER_URL", "").strip()
    if not remote_tts_enabled():
        return 0

    # (voice, speed) -> ordered unique sentences
    requests: dict[tuple[str, float], list[str]] = {}
    reused = 0
    for scene in scenes:
        for line in scene.get("lines") or []:
            text = " ".join(str(line.get("text") or "").split())
            if not text:
                continue
            voice, line_speed, _character = line_voice_and_speed(
                scene, line, characters, voice_default, base_speed, dynamic_speed
            )
            primary = resolve_voice(voice).split(",")[0].strip()
            bucket = requests.setdefault((primary, round(line_speed, 3)), [])
            for sentence in split_sentences(text):
                remote_text = prepare_remote_text(sentence)
                if not remote_text.strip() or remote_text in bucket:
                    continue
                # Already downloaded on an earlier run. Asking the GPU again would pay
                # for work the disk cache can serve for free.
                if gpu_cache_path(remote_text, primary, line_speed).exists():
                    reused += 1
                    continue
                bucket.append(remote_text)

    # A setdefault above creates a bucket even when every sentence in it was reused, so
    # drop the empties before deciding whether anything needs to be requested.
    requests = {key: value for key, value in requests.items() if value}
    if reused:
        print(f"[Indic Parler Vast Batch] {reused} segment(s) already on disk; requesting only the rest.")
    if not requests:
        return reused

    from vast_manager import VastManager

    manager = VastManager(server_url=vast_url)
    cached_count = 0
    for (primary, line_speed), sentences in requests.items():
        print(f"[Indic Parler Vast Batch] request {len(sentences)} segment(s) for {primary} @ {line_speed}x ...")
        items = [{"id": str(index), "text": sentence} for index, sentence in enumerate(sentences)]
        try:
            results = manager.synthesize_tts_batch(items, voice=primary, speed=line_speed)
        except Exception as err:  # noqa: BLE001 - fall back to per-sentence requests
            print(
                f"[Indic Parler Vast Batch Warning] batch request failed ({err}); "
                "falling back to per-sentence requests.",
                file=sys.stderr,
            )
            continue

        import io as _io
        import soundfile as sf

        missing: list[str] = []
        for index, sentence in enumerate(sentences):
            entry = results.get(str(index))
            if not entry:
                missing.append(sentence)
                continue
            wav_bytes, _duration = entry
            try:
                wav_np, _sr = sf.read(_io.BytesIO(wav_bytes), dtype="float32")
            except Exception:
                missing.append(sentence)
                continue
            if audio_is_bad(wav_np):
                print(
                    f"[Indic Parler Vast Batch] segment {index + 1} came back silent/non-finite; "
                    "retrying it individually.",
                    file=sys.stderr,
                )
                missing.append(sentence)
                continue
            # Persist only after the audio proved usable: a corrupt segment must never
            # reach the disk cache, or the next run would reuse it for free.
            write_gpu_cache(sentence, primary, line_speed, wav_bytes)
            _VAST_AUDIO_CACHE[(sentence, primary.lower(), line_speed)] = wav_np
            cached_count += 1
        print(f"[Indic Parler Vast Batch] cached {cached_count} segment(s) so far")

        # Items the server omitted are retried individually rather than dropped, which is
        # what the batch endpoint's contract expects of the caller.
        if missing:
            print(
                f"[Vast TTS Batch] {len(missing)} segment(s) missing from the batch "
                "response; retrying them individually."
            )
            for sentence in missing:
                synthesize = synthesize_indicf5_local if is_indicf5_local_voice(primary) else synthesize_indic_parler
                if synthesize(sentence, primary, line_speed) is not None:
                    cached_count += 1
    return reused + cached_count


# ---------------------------------------------------------------------------
# Local IndicF5 (zero-shot cloner) — subprocess bridge to examples/indicf5
# ---------------------------------------------------------------------------


def is_indicf5_local_voice(voice: str) -> bool:
    """True when the resolved voice is one of the local IndicF5 cloner voices."""
    primary = (voice or "").strip().lower().split(",")[0].strip()
    return primary in INDICF5_LOCAL_VOICES


def indicf5_reference_name(voice: str) -> str | None:
    """Reference clip stem for an IndicF5 voice ('indicf5_aarav' -> 'aarav'), or None."""
    primary = (voice or "").strip().lower().split(",")[0].strip()
    info = INDICF5_LOCAL_VOICES.get(primary)
    if info:
        return info["reference"]
    if primary.startswith("indicf5_"):
        return primary[len("indicf5_"):]
    return None


def indicf5_available() -> tuple[bool, str]:
    """Whether the local IndicF5 example is installed and runnable."""
    if not INDICF5_VENV_PYTHON.exists():
        return False, (
            f"the local IndicF5 environment is missing ({INDICF5_VENV_PYTHON}). "
            "Create it once with: npm run indicf5:setup"
        )
    if not INDICF5_BATCH_SCRIPT.exists():
        return False, f"missing {INDICF5_BATCH_SCRIPT}"
    return True, ""


def _indicf5_read_result(child_output: str, result_path: Path) -> dict:
    """Return the batch child's result payload.

    The result file is authoritative. The sentinel-prefixed stdout line is only a
    fallback for a child whose temp dir is unwritable; every other stdout line is
    third-party progress noise and must be ignored.
    """
    if result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))
    for line in reversed(child_output.splitlines()):
        stripped = line.strip()
        if stripped.startswith(INDICF5_RESULT_PREFIX):
            return json.loads(stripped[len(INDICF5_RESULT_PREFIX):])
    raise ValueError(
        "no result file and no 'INDICF5_RESULT ' line in the subprocess output"
    )


def _indicf5_run_batch(items: list[tuple[str, str, float]]) -> dict[int, bytes]:
    """Synthesise all ``(text, voice, speed)`` items in ONE IndicF5 subprocess.

    The model lives in its own Python 3.11 venv, so it is loaded once per call there
    rather than imported into this process. Returns {index: wav_bytes} for the successes.
    """
    global _INDICF5_BATCH_USABLE
    ok, why = indicf5_available()
    if not ok:
        _INDICF5_BATCH_USABLE = False
        raise RuntimeError(f"IndicF5 unavailable: {why}")
    job = {
        "voices_dir": str(INDICF5_VOICES_DIR),
        "items": [
            {"id": str(index), "text": text, "voice": indicf5_reference_name(voice) or voice}
            for index, (text, voice, _speed) in enumerate(items)
        ],
    }
    with tempfile.TemporaryDirectory(prefix="indicf5-") as tmp:
        job["out_dir"] = tmp
        # stderr is folded into stdout: the child's diagnostics must survive, and the
        # result no longer travels over stdout (see INDICF5_RESULT_FILE).
        proc = subprocess.run(
            [str(INDICF5_VENV_PYTHON), str(INDICF5_BATCH_SCRIPT), "--job", "-"],
            input=json.dumps(job).encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=float(os.environ.get("INDICF5_TIMEOUT", "7200")),
        )
        child_output = proc.stdout.decode("utf-8", "replace").strip()
        if proc.returncode != 0:
            tail = child_output.splitlines()[-6:]
            raise RuntimeError(f"IndicF5 batch failed (exit {proc.returncode}): {' | '.join(tail)}")
        try:
            payload = _indicf5_read_result(child_output, Path(tmp) / INDICF5_RESULT_FILE)
        except (OSError, ValueError) as err:
            tail = child_output.splitlines()[-6:]
            raise RuntimeError(
                f"IndicF5 batch produced no readable result ({err}); child output tail: "
                f"{' | '.join(tail) if tail else '(empty)'}"
            ) from err
        # A cold run costs minutes per segment, so surface the child's own progress and
        # warnings instead of hiding them; errors below are reported separately.
        for line in child_output.splitlines():
            if INDICF5_RESULT_PREFIX not in line:
                print(f"[IndicF5 Batch] {line}", file=sys.stderr)
        for entry_id, reason in (payload.get("errors") or {}).items():
            print(f"[IndicF5 Local] segment {entry_id} failed: {reason}", file=sys.stderr)
        out: dict[int, bytes] = {}
        for entry_id, entry in (payload.get("results") or {}).items():
            path = Path(entry["path"])
            if path.exists():
                out[int(entry_id)] = path.read_bytes()
        return out


def synthesize_indicf5_local(text: str, voice_key: str, speed: float = 1.0):
    """Synthesise one sentence with IndicF5, preferring remote Vast and falling back locally."""
    if indicf5_reference_name(voice_key) is None:
        return None
    norm_text = normalize_hindi_phonetics(text)
    primary = voice_key.lower()
    cache_key = (norm_text, primary, round(speed, 3))
    cached = _VAST_AUDIO_CACHE.get(cache_key)
    if cached is not None:
        return cached  # type: ignore[return-value]
    disk_cached = read_gpu_cache(norm_text, primary, speed)
    if disk_cached is not None:
        _VAST_AUDIO_CACHE[cache_key] = disk_cached
        return disk_cached  # type: ignore[return-value]

    import io
    import soundfile as sf

    vast_url = os.environ.get("VAST_SERVER_URL", "").strip()
    if remote_tts_enabled():
        try:
            from vast_manager import VastManager
            manager = VastManager(server_url=vast_url)
            manager.require_gpu("RTX 4090", vast_url, require_tts=True)
            audio_bytes, _duration = manager.synthesize_tts(
                text=norm_text, voice=primary, speed=speed
            )
            wav_np, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            wav_np = resample_audio(wav_np, sample_rate or SAMPLE_RATE, SAMPLE_RATE)
            if audio_is_bad(wav_np):
                raise RuntimeError("remote IndicF5 returned silent/non-finite audio")
            write_gpu_cache(norm_text, primary, speed, audio_bytes)
            _VAST_AUDIO_CACHE[cache_key] = wav_np
            return wav_np
        except Exception as err:  # noqa: BLE001 - local fallback is intentional
            print(f"[IndicF5 Remote Warning] {type(err).__name__}: {err}; falling back locally", file=sys.stderr)
            if os.environ.get("REQUIRE_VAST_TTS", "0") == "1":
                raise

    try:
        results = _indicf5_run_batch([(norm_text, voice_key, speed)])
    except Exception as err:  # noqa: BLE001 - fall back to Kokoro rather than crash
        print(f"[IndicF5 Local Warning] {err}", file=sys.stderr)
        return None
    wav_bytes = results.get(0)
    if not wav_bytes:
        return None
    wav_np, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    wav_np = resample_audio(wav_np, sr or SAMPLE_RATE, SAMPLE_RATE)
    if audio_is_bad(wav_np):
        print("[IndicF5 Local Warning] silent/non-finite audio; not cached", file=sys.stderr)
        return None
    write_gpu_cache(norm_text, primary, speed, wav_bytes)
    _VAST_AUDIO_CACHE[cache_key] = wav_np
    return wav_np


def prebuild_indicf5_batch(
    scenes: list[dict],
    characters: dict,
    voice_default: str,
    base_speed: float,
    dynamic_speed: bool,
) -> int:
    """Synthesise every IndicF5 narration sentence in ONE subprocess round trip.

    Mirrors prebuild_vast_batch: the model is loaded once for the whole script and each
    segment is persisted to the disk cache, so the per-sentence path costs nothing extra.
    """
    ok, why = indicf5_available()
    if not ok:
        raise RuntimeError(f"IndicF5 unavailable: {why}")

    requests: dict[tuple[str, float], list[str]] = {}
    reused = 0
    for scene in scenes:
        for line in scene.get("lines") or []:
            text = " ".join(str(line.get("text") or "").split())
            if not text:
                continue
            voice, line_speed, _character = line_voice_and_speed(
                scene, line, characters, voice_default, base_speed, dynamic_speed
            )
            primary = resolve_voice(voice).split(",")[0].strip()
            if not is_indicf5_local_voice(primary):
                continue
            bucket = requests.setdefault((primary, round(line_speed, 3)), [])
            for sentence in split_sentences(text):
                remote_text = prepare_remote_text(sentence)
                if not remote_text.strip() or remote_text in bucket:
                    continue
                if gpu_cache_path(remote_text, primary, line_speed).exists():
                    reused += 1
                    continue
                bucket.append(remote_text)

    requests = {key: value for key, value in requests.items() if value}
    if reused:
        print(f"[IndicF5 Local] {reused} segment(s) already on disk; synthesising only the rest.")
    if not requests:
        return reused

    import io

    import soundfile as sf

    cached_count = 0
    for (primary, line_speed), sentences in requests.items():
        print(
            f"[IndicF5 Local] synthesising {len(sentences)} segment(s) for {primary} "
            "(loads the model once)...",
            flush=True,
        )
        try:
            results = _indicf5_run_batch([(s, primary, line_speed) for s in sentences])
        except Exception as err:  # noqa: BLE001 - per-sentence synthesis still works
            print(
                f"[IndicF5 Local Warning] batch request failed ({err}); "
                "falling back to per-sentence synthesis.",
                file=sys.stderr,
            )
            continue
        for index, sentence in enumerate(sentences):
            wav_bytes = results.get(index)
            if not wav_bytes:
                continue
            try:
                wav_np, _sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
            except Exception:
                continue
            if audio_is_bad(wav_np):
                print(
                    f"[IndicF5 Local] segment {index + 1} came back silent/non-finite; "
                    "it will be retried individually.",
                    file=sys.stderr,
                )
                continue
            write_gpu_cache(sentence, primary, line_speed, wav_bytes)
            _VAST_AUDIO_CACHE[(sentence, primary.lower(), line_speed)] = wav_np
            cached_count += 1
        print(f"[IndicF5 Local] cached {cached_count} segment(s) so far")
    return reused + cached_count


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
        Minor internal punctuation like semicolons and dashes is cleaned to prevent
        unnatural mid-sentence audio pauses. Commas are preserved so the engine can
        produce natural clause-level prosody. Hindi text is normalized for natural
        conversational pronunciation.
        """
        import numpy as np

        # Strip internal pauses caused by minor punctuation and apply conversational Hindi
        # phonetics. Shared with the Vast.ai batch pre-fetch so both derive identical text.
        clean_text = prepare_tts_text(text)

        resolved_voice = resolve_voice(voice)
        primary = resolved_voice.split(",")[0].strip()

        # Local IndicF5 zero-shot cloner (examples/indicf5). Checked BEFORE the Indic
        # Parler-TTS branch because 'indicf5_aarav' also starts with 'indic'.
        if is_indicf5_local_voice(primary):
            indicf5_audio = synthesize_indicf5_local(clean_text, primary, speed=speed)
            if indicf5_audio is not None and len(indicf5_audio) > 0:
                return apply_fades(trim_edge_silence(indicf5_audio, threshold=0.005, padding=0.005))
            vinfo = INDICF5_LOCAL_VOICES.get(primary.lower())
            fallback = vinfo["fallback_kokoro"] if vinfo else "hm_omega"
            print(
                f"[IndicF5 Local] synthesis failed; falling back to Kokoro voice {fallback}",
                file=sys.stderr,
            )
            resolved_voice = fallback
            primary = resolved_voice.split(",")[0].strip()

        # Check if the voice is Indic Parler-TTS (AI4Bharat description-conditioned TTS)
        is_indic = primary.lower().startswith("indic") or primary.lower() in INDIC_PARLER_VOICES or any(
            primary.lower() in [a.lower() for a in vi.get("aliases", [])] for vi in INDIC_PARLER_VOICES.values()
        )
        if is_indic:
            indic_audio = synthesize_indic_parler(clean_text, primary, speed=speed)
            if indic_audio is not None and len(indic_audio) > 0:
                return apply_fades(trim_edge_silence(indic_audio, threshold=0.005, padding=0.005))
            # Fallback if Indic Parler-TTS fails
            vinfo = INDIC_PARLER_VOICES.get(primary.lower())
            fallback = vinfo["fallback_kokoro"] if vinfo else "hm_omega"
            resolved_voice = fallback
            primary = resolved_voice.split(",")[0].strip()

        # Check if voice is Cartesia Sonic (with zero-crash Kokoro fallback)
        is_cartesia = primary.lower().startswith("cartesia")
        cartesia_info = None
        if is_cartesia:
            cartesia_info = CARTESIA_VOICES.get(primary.lower())
            if not cartesia_info:
                for ckey, cinfo in CARTESIA_VOICES.items():
                    if primary.lower() == ckey.lower() or primary.lower() in [
                        a.lower() for a in cinfo.get("aliases", [])
                    ]:
                        cartesia_info = cinfo
                        break

            fallback = cartesia_info["fallback_kokoro"] if cartesia_info else "hf_beta"
            lang = cartesia_info.get("lang", "hi") if cartesia_info else "hi"
            cartesia_audio = synthesize_cartesia(
                clean_text,
                primary,
                speed=speed,
                language=lang,
                fallback_voice=fallback,
            )
            if cartesia_audio is not None and len(cartesia_audio) > 0:
                return cartesia_audio

            # Fallback to local Kokoro neural voice
            resolved_voice = fallback
            primary = resolved_voice.split(",")[0].strip()

        lang_code = primary[:1].lower() or "a"

        # If text is Devanagari Hindi, guarantee lang_code is 'h' for native Hindi phonemization
        if is_devanagari(clean_text) or any(v.startswith("h") for v in resolved_voice.split(",")):
            lang_code = "h"

        chunks = [
            apply_fades(trim_edge_silence(np.asarray(audio, dtype=np.float32), threshold=0.005, padding=0.005))
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


def trim_edge_silence(audio, threshold: float = 0.005, padding: float = 0.005):
    """Remove Kokoro vocoder boundary padding at a segment's edges, retaining a 5ms micro-breath.

    Kokoro neural vocoder emits boundary silence (~ -50 dBFS) before and after a spoken segment.
    Trimming it cleanly down to a 5ms boundary ensures pauses between sentences are
    strictly governed by the narrative prosody parameters rather than accumulated vocoder padding.
    """
    import numpy as np

    if len(audio) == 0:
        return audio

    frame = max(1, int(SAMPLE_RATE * 0.01))
    frame_count = (len(audio) + frame - 1) // frame
    padded = np.pad(audio, (0, frame_count * frame - len(audio)))
    levels = np.sqrt(np.mean(padded.reshape(frame_count, frame) ** 2, axis=1))
    active = np.flatnonzero(levels > threshold)
    if not len(active):
        return audio

    padding_samples = int(padding * SAMPLE_RATE)
    first, last = int(active[0]), int(active[-1])
    start = max(0, first * frame - padding_samples)
    end = min(len(audio), (last + 1) * frame + padding_samples)

    if start >= end:
        return audio
    return np.ascontiguousarray(audio[start:end])


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


def apply_whisper_dsp(audio, intensity: float = 0.85):
    """Transform vocal speech into an eerie, breathy, close-mic whisper.

    Acoustic whisper mechanics:
      1. Strips the vocal cord harmonic fundamental (F0 < 420Hz).
      2. Enhances the high-frequency air/sibilance band (3.2kHz - 7.5kHz).
      3. Applies subtle intimate close-mic compression.
    """
    import numpy as np
    from scipy import signal

    if audio.size < 256:
        return audio

    x = np.asarray(audio, dtype=np.float64)
    nyquist = SAMPLE_RATE / 2.0

    # 1. High-pass filter above 420Hz to strip vocal cord fundamental vibrations
    sos_hp = signal.butter(3, 420.0 / nyquist, btype="highpass", output="sos")
    whispered = signal.sosfiltfilt(sos_hp, x)

    # 2. Boost airy sibilance band (3200Hz - 7500Hz) for intimate breath texture
    sos_air = signal.butter(2, [3200.0 / nyquist, min(7600.0, nyquist * 0.95) / nyquist], btype="bandpass", output="sos")
    air = signal.sosfiltfilt(sos_air, x)
    whispered = whispered + 1.25 * air

    # 3. Blend filtered breath with original signal according to intensity
    blended = (1.0 - intensity * 0.3) * whispered + (1.0 - intensity) * 0.2 * x

    # 4. Normalize
    max_val = np.max(np.abs(blended)) + 1e-12
    if max_val > 0:
        blended = (blended / max_val) * 0.88
    return blended.astype(np.float32)


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

    # Apply acoustic whisper filter if configured
    if settings.get("whisper"):
        audio = apply_whisper_dsp(audio, intensity=float(settings.get("whisper_intensity", 0.85)))

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

    Each sentence is rendered as a single continuous vocal unit so that natural
    prosody, pitch contours, and word transitions remain uninterrupted.
    Duration is spread across the sentence's words in proportion to phonetic weights.
    """
    import numpy as np

    sentences = split_sentences(text)
    if not sentences:
        return np.zeros(0, dtype=np.float32), []

    pieces: list = []
    words: list[dict] = []
    offset = 0.0

    for index, sentence in enumerate(sentences):
        # Synthesise complete sentence as an uninterrupted vocal unit
        audio = apply_fades(trim_edge_silence(engine.say(sentence, voice, speed)))
        duration = len(audio) / SAMPLE_RATE

        spoken = [w for w in sentence.split() if w]
        weights = engine.phonemiser.weights(sentence, voice)
        if len(weights) != len(spoken):
            # Fallback to even weights if alignment fails
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
    parser.add_argument("-t", "--text", type=str, default=None, help="synthesise a single test sentence directly and save to --out or out/test_audio.wav")
    parser.add_argument("-v", "--voice", type=str, default=None, help="voice name for test synthesis (e.g. hf_beta, hm_omega, am_onyx)")
    parser.add_argument("-o", "--out", type=str, default=None, help="output .wav file path for --text test")
    parser.add_argument("--dynamic-speed", dest="dynamic_speed", action="store_true", default=True, help="modulate speed dynamically based on scene emotion & beats for natural human tone")
    parser.add_argument("--flat-speed", dest="dynamic_speed", action="store_false", help="disable emotion-based speed modulation and use strictly flat speed")
    parser.add_argument("--speed", type=float, default=None, help="speech speed multiplier (e.g. 0.8 to 1.5, default from script.json or .env)")
    parser.add_argument("--whisper", action="store_true", help="apply eerie breathy close-mic whisper acoustic filter")
    parser.add_argument("--force", action="store_true", help="re-synthesise cached audio")
    parser.add_argument("--no-lipsync", action="store_true", help="skip Rhubarb mouth shapes")
    parser.add_argument("--voices", action="store_true", help="list voices and exit")
    parser.add_argument("--samples", action="store_true", help="generate short preview .wav samples for Hindi / Hinglish voices into public/audio/samples/")
    parser.add_argument("--hinglish", action="store_true", help="force Roman Hinglish on-screen captions & subtitles")
    parser.add_argument("--no-hinglish", action="store_true", help="disable Roman Hinglish transliteration (keep Devanagari subtitles)")
    parser.add_argument("--vast", action="store_true", help="use the remote Vast.ai GPU instance for IndicF5 or legacy Indic Parler synthesis")
    parser.add_argument("--vast-url", type=str, default=None, help="Vast.ai server URL (e.g. http://ip:port)")
    parser.add_argument("--require-vast", action="store_true", help="fail rather than falling back if remote IndicF5 or legacy Indic Parler synthesis fails")
    parser.add_argument("--no-vast-batch", action="store_true", help="skip the one-shot bulk Vast.ai narration pre-fetch")
    parser.add_argument("--vast-batch", dest="no_vast_batch", action="store_false", help="(default) pre-fetch all narration in one Vast.ai round trip")
    parser.add_argument("--purge-audio-cache", action="store_true", help="delete cached GPU narration segments and scene sidecars, then exit")
    parser.set_defaults(no_vast_batch=False)
    args = parser.parse_args()

    if args.purge_audio_cache:
        return purge_audio_cache()

    if args.vast:
        os.environ["USE_VAST"] = "1"
    if args.vast_url:
        os.environ["VAST_SERVER_URL"] = args.vast_url
    if args.require_vast:
        if not (args.vast or args.vast_url or os.environ.get("VAST_SERVER_URL")):
            parser.error("--require-vast needs --vast or a configured Vast server URL.")
        os.environ["REQUIRE_VAST_TTS"] = "1"

    # Ensure audio assets (ambience beds, sfx) are present
    try:
        ensure_assets()
    except Exception as e:
        print(f"note: could not verify all procedural audio assets: {e}", file=sys.stderr)

    if args.voices:
        print(format_voices_table())
        return 0

    if args.samples:
        return generate_voice_samples()

    if args.text:
        import soundfile as sf
        engine = Voice()
        voice_name = args.voice or "hf_beta"
        sp = float(args.speed) if args.speed is not None else 1.0
        out_path = Path(args.out or ROOT / "out" / "test_audio.wav").resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Synthesising test line (Voice: {voice_name}, Speed: {sp:.2f}, Whisper: {args.whisper}):")
        print(f"Text: {args.text}")
        master_cfg = {**DEFAULT_MASTER, "whisper": args.whisper}
        audio_data, words = synthesise_line(engine, args.text, voice_name, sp, 0.3, master=master_cfg)
        sf.write(str(out_path), audio_data, SAMPLE_RATE, subtype="PCM_16")
        dur = len(audio_data) / SAMPLE_RATE
        print(f"[SUCCESS] Audio saved to: {out_path} ({dur:.2f}s, {len(words)} words)")
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
    # Parler-TTS joins sentences very tightly; give the Indic narrator a clearer breath
    # between sentences so adjacent lines do not run together as one continuous stream.
    if is_indic_voice(voice_default) and pause < 0.18:
        pause = 0.18
    master = script.get("master") or {}
    characters = script.get("characters") or {}

    scenes_in = raw.get("scenes") or []
    if not scenes_in:
        print("error: script.json has no scenes", file=sys.stderr)
        return 1

    scenes = [merge(DEFAULT_SCENE, s) for s in scenes_in]
    if is_indic_voice(voice_default):
        # A clear breath at every scene boundary, so one scene's closing line does not
        # run straight into the next scene's opening line.
        for scene in scenes:
            scene["padBefore"] = max(float(scene.get("padBefore") or 0.0), 0.12)
            scene["padAfter"] = max(float(scene.get("padAfter") or 0.0), 0.35)
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

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    engine = Voice()

    # Local IndicF5 is an explicit fallback: when Vast is configured, use its batch API
    # below rather than loading the local 3.11 model process unnecessarily.
    if is_indicf5_local_voice(voice_default) and not remote_tts_enabled() and not args.no_vast_batch:
        if os.environ.get("REQUIRE_VAST_TTS", "0") == "1":
            print(
                "error: remote IndicF5 is required but no Vast.ai server is configured.",
                file=sys.stderr,
            )
            return 1
        try:
            indicf5_cached = prebuild_indicf5_batch(
                scenes, characters, voice_default, speed, args.dynamic_speed
            )
            if indicf5_cached:
                print(f"[IndicF5 Local] {indicf5_cached} segment(s) ready")
        except Exception as batch_err:  # noqa: BLE001 - per-sentence synthesis still works
            print(
                f"[IndicF5 Local Warning] pre-fetch failed ({batch_err}); "
                "using per-sentence synthesis.",
                file=sys.stderr,
            )

    # One bulk round trip for the whole narration when remote TTS is selected; the
    # per-sentence path below then reads those segments from memory.
    if remote_tts_enabled() and not args.no_vast_batch:
        try:
            batch_cached = prebuild_vast_batch(scenes, characters, voice_default, speed, args.dynamic_speed)
            if batch_cached:
                print(f"[Vast TTS Batch] {batch_cached} segment(s) ready from the GPU")
        except Exception as batch_err:  # noqa: BLE001 - never block synthesis on the pre-fetch
            print(
                f"[Vast TTS Batch Warning] pre-fetch failed ({batch_err}); "
                "using per-sentence requests.",
                file=sys.stderr,
            )

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
            voice, line_speed, character = line_voice_and_speed(
                scene, line, characters, voice_default, speed, args.dynamic_speed
            )

            suffix = f"-{line_index + 1}" if len(scene["lines"]) > 1 else ""
            audio_path = AUDIO_DIR / f"{scene['id']}{suffix}.wav"

            line_master = dict(master)
            if args.whisper or scene.get("whisper") or character.get("whisper") or "whisper" in str(voice).lower():
                line_master["whisper"] = True

            meta = build_line(
                engine,
                audio_path=audio_path,
                text=text,
                voice=voice,
                speed=line_speed,
                pause=pause,
                master=line_master,
                rhubarb=rhubarb,
                force=args.force,
                label=f"[{scene_index + 1}.{line_index + 1}] {audio_path.name}",
            )
            if meta["duration"] <= 0:
                continue

            built.append(
                {
                    "characterId": character_id,
                    "audio": f"audio/voice/{audio_path.name}",
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
