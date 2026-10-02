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
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PUBLIC = ROOT / "public"
AUDIO_DIR = PUBLIC / "audio"
GENERATED = ROOT / "src" / "generated"

SAMPLE_RATE = 24000  # Kokoro's native rate

# Kokoro v1.0 English voices. Any voice id from hexgrad/Kokoro-82M also works.
EN_VOICES = {
    "af_heart": "American female, warm (highest rated)",
    "af_bella": "American female, bright",
    "af_nicole": "American female, soft",
    "af_sarah": "American female, clear",
    "af_sky": "American female, light",
    "af_alloy": "American female",
    "af_aoede": "American female",
    "af_jessica": "American female",
    "af_kore": "American female",
    "af_nova": "American female",
    "af_river": "American female",
    "am_michael": "American male, calm narration",
    "am_adam": "American male",
    "am_echo": "American male",
    "am_eric": "American male",
    "am_fenrir": "American male",
    "am_liam": "American male",
    "am_onyx": "American male, deep",
    "am_puck": "American male",
    "am_santa": "American male, older",
    "bf_alice": "British female",
    "bf_emma": "British female",
    "bf_isabella": "British female",
    "bf_lily": "British female",
    "bm_george": "British male, documentary",
    "bm_lewis": "British male",
    "bm_daniel": "British male",
    "bm_fable": "British male",
}

# Rhubarb's Hanna-Barbera mouth shape set.
MOUTH_SHAPES = {"A", "B", "C", "D", "E", "F", "G", "H", "X"}

# Conservative vowel set covering Kokoro's ARPAbet-derived IPA. It is only used
# to weight words against each other, so an occasional miss just softens the
# estimate rather than breaking it.
VOWELS = set("aeiouæɑɒɔəɜɛɪʊʌɐɨɵɘʉɞɶøœyAOIᵻᵊ")
STRESS = set("ˈˌ")
IGNORED = set(".,!?;:'\"()[]-—… \t\n")

DEFAULT_SCRIPT = {
    "title": "Untitled",
    "voice": "am_michael",
    "speed": 1.0,
    "fps": 30,
    "width": 1920,
    "height": 1080,
    "tailPaddingSeconds": 1.2,
    "sentencePauseSeconds": 0.18,
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
    """Split narration into sentences, keeping common abbreviations intact."""
    text = " ".join(text.split())
    if not text:
        return []

    sentences: list[str] = []
    for part in re.split(r"(?<=[.!?])\s+", text):
        if sentences:
            tail = re.split(r"[\s.]+", sentences[-1].rstrip())[-1].lower().strip(".")
            if tail in ABBREVIATIONS:
                sentences[-1] = f"{sentences[-1]} {part}"
                continue
        sentences.append(part)
    return sentences


def phoneme_weight(phonemes: str) -> float:
    """Relative duration of a word's pronunciation.

    Vowels carry most of a syllable's length; consonants are shorter. Stress and
    length marks stretch the syllable they attach to.
    """
    weight = 0.0
    for char in phonemes:
        if char in IGNORED:
            continue
        if char in STRESS or char == "ː":
            weight += 0.35
        elif char in VOWELS:
            weight += 1.0
        else:
            weight += 0.6
    return max(weight, 0.4)


# --------------------------------------------------------------------------
# Speech synthesis
# --------------------------------------------------------------------------


class Voice:
    """Wraps the Kokoro pipeline, loading one per language code."""

    def __init__(self) -> None:
        self._pipelines: dict[str, object] = {}
        self._g2p = None

    def _pipeline(self, lang_code: str):
        if lang_code not in self._pipelines:
            from kokoro import KPipeline

            self._pipelines[lang_code] = KPipeline(lang_code=lang_code)
        return self._pipelines[lang_code]

    def _tokeniser(self):
        if self._g2p is None:
            from misaki import en

            self._g2p = en.G2P()
        return self._g2p

    def tokens(self, text: str) -> list:
        """Word tokens carrying per-word IPA phonemes."""
        _phonemes, tokens = self._tokeniser()(text)
        return [t for t in tokens if (getattr(t, "text", "") or "").strip()]

    def say(self, text: str, voice: str, speed: float):
        """Synthesise one sentence; returns a float32 mono numpy array."""
        import numpy as np

        lang_code = voice[0] if voice[:1] in {"a", "b"} else "a"
        chunks = [
            np.asarray(audio, dtype=np.float32)
            for _g, _p, audio in self._pipeline(lang_code)(text, voice=voice, speed=speed)
        ]
        if not chunks:
            raise RuntimeError(f"Kokoro produced no audio for: {text[:60]!r}")
        return np.concatenate(chunks)


def synthesise_line(engine: Voice, text: str, voice: str, speed: float, pause: float):
    """Synthesise a whole line, returning (audio, words).

    Each sentence is rendered separately so its duration is exact, then that
    duration is spread across the sentence's words by phoneme weight.
    """
    import numpy as np

    sentences = split_sentences(text)
    if not sentences:
        return np.zeros(0, dtype=np.float32), []

    pieces: list = []
    words: list[dict] = []
    offset = 0.0
    silence = np.zeros(int(SAMPLE_RATE * pause), dtype=np.float32)

    for index, sentence in enumerate(sentences):
        audio = engine.say(sentence, voice, speed)
        duration = len(audio) / SAMPLE_RATE

        tokens = engine.tokens(sentence)
        if tokens:
            weights = [phoneme_weight(getattr(t, "phonemes", "") or "") for t in tokens]
            total = sum(weights) or 1.0
            cursor = offset
            for token, weight in zip(tokens, weights):
                span = duration * weight / total
                words.append(
                    {
                        "text": getattr(token, "text", ""),
                        "start": round(cursor, 4),
                        "end": round(cursor + span, 4),
                    }
                )
                cursor += span

        pieces.append(audio)
        offset += duration

        if index < len(sentences) - 1:
            pieces.append(silence)
            offset += pause

    return np.concatenate(pieces), words


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
    rhubarb: Path | None,
    force: bool,
    label: str,
) -> dict:
    meta_path = audio_path.with_suffix(".json")
    key = hashlib.sha256(
        json.dumps([voice, round(speed, 4), round(pause, 4), text]).encode("utf-8")
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

    audio, words = synthesise_line(engine, text, voice, speed, pause)
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
    parser.add_argument("--force", action="store_true", help="re-synthesise cached audio")
    parser.add_argument("--no-lipsync", action="store_true", help="skip Rhubarb mouth shapes")
    parser.add_argument("--voices", action="store_true", help="list voices and exit")
    args = parser.parse_args()

    if args.voices:
        print("Kokoro English voices (use the id in script.json or a scene's `voice`):\n")
        for name, description in EN_VOICES.items():
            print(f"  {name:<14} {description}")
        return 0

    script_path = Path(args.script).resolve()
    if not script_path.exists():
        print(f"error: {script_path} not found", file=sys.stderr)
        return 1

    raw = json.loads(script_path.read_text(encoding="utf-8"))
    script = merge(DEFAULT_SCRIPT, raw)
    speed = rate_to_speed(script)
    voice_default = script["voice"]
    pause = float(script["sentencePauseSeconds"])
    characters = script.get("characters") or {}

    scenes_in = raw.get("scenes") or []
    if not scenes_in:
        print("error: script.json has no scenes", file=sys.stderr)
        return 1

    scenes = [merge(DEFAULT_SCENE, s) for s in scenes_in]
    for index, scene in enumerate(scenes):
        scene["id"] = slugify(scene.get("id") or "", f"scene-{index + 1}")
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
            voice = character.get("voice") or voice_default
            line_speed = float(character.get("speed") or speed)

            suffix = f"-{line_index + 1}" if len(scene["lines"]) > 1 else ""
            audio_path = AUDIO_DIR / f"{scene['id']}{suffix}.wav"

            meta = build_line(
                engine,
                audio_path=audio_path,
                text=text,
                voice=voice,
                speed=line_speed,
                pause=pause,
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
                "text": word["text"],
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

    minutes, seconds = divmod(cursor, 60)
    wpm = total_words / (cursor / 60) if cursor else 0
    print(
        f"\nTimeline : {int(minutes)}m {seconds:04.1f}s | {total_words} words | {wpm:.0f} wpm"
    )
    print(f"Wrote    : {(GENERATED / 'project.json').relative_to(ROOT)}")
    if wpm > 185:
        print("Note: over 185 wpm is fast for narration; consider fewer words per scene.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
