"""Check whether Kokoro can narrate Hindi, and how well.

Hindi is the critical unknown: the channels referenced are Hindi horror and
manhwa-recap narration. Kokoro v1.0 ships Hindi voices (h*_*) and misaki needs
espeak-ng for Hindi phonemisation, so this verifies the whole chain rather than
assuming it works.

Run with:  .venv\\Scripts\\python.exe scripts\\probe_hindi.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

HINDI_TEXT = (
    "रात के दो बजे थे। गाँव के बाहर वाली पुरानी हवेली से आवाज़ आ रही थी। "
    "कोई नहीं जानता था कि उस घर में कौन रहता है।"
)

ENGLISH_SAMPLE = "The pressure at four thousand meters would crush a steel submarine."


def main() -> int:
    print("--- all voices matching Hindi locales ---")
    from kokoro import KPipeline

    # Kokoro's voice list is fetched from the model repo's VOICES.md.
    try:
        import huggingface_hub as hf

        path = hf.hf_hub_download("hexgrad/Kokoro-82M", "voices.json")
        import json

        voices = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"  could not fetch voices.json: {type(exc).__name__}: {exc}")
        voices = {}

    hindi = {k: v for k, v in voices.items() if k.startswith("h")}
    if hindi:
        for name, meta in sorted(hindi.items()):
            language = meta.get("language") if isinstance(meta, dict) else meta
            print(f"  {name:<12} {language}")
    else:
        print("  (none found in voices.json)")

    print()
    print("--- misaki Hindi phonemiser ---")
    try:
        from misaki import espeak

        phonemizer = espeak.EspeakG2P(language="hi")
        phonemes, _ = phonemizer(HINDI_TEXT)
        print(f"  OK  {phonemes[:70]}")
    except Exception as exc:
        print(f"  FAIL {type(exc).__name__}: {exc}")
        print("       Hindi narration is not usable until this works.")
        return 1

    print()
    print("--- synthesising Hindi ---")
    import numpy as np
    import soundfile as sf

    for voice in ("hm_omega", "hm_psi", "hf_alpha", "hf_beta"):
        pipeline = KPipeline(lang_code="h")
        started = time.time()
        chunks = [
            np.asarray(audio, dtype=np.float32)
            for _g, _p, audio in pipeline(HINDI_TEXT, voice=voice, speed=1.0)
        ]
        elapsed = time.time() - started
        if not chunks:
            print(f"  {voice:<9} FAILED (no audio)")
            continue

        audio = np.concatenate(chunks)
        duration = len(audio) / 24000
        rtf = elapsed / duration if duration else 0
        out = ROOT / "out" / f"_hindi_{voice}.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out), audio, 24000)
        print(
            f"  {voice:<9} OK  {duration:5.2f}s audio in {elapsed:5.2f}s "
            f"(rtf {rtf:.2f})  peak {np.abs(audio).max():.3f}  -> {out.name}"
        )

    print()
    print("--- mixed Hindi/English (Hinglish), common in these videos ---")
    hinglish = "रात के दो बजे थे। The pressure was unbearable. कोई नहीं जानता था।"
    try:
        pipeline = KPipeline(lang_code="h")
        chunks = [
            np.asarray(audio, dtype=np.float32)
            for _g, _p, audio in pipeline(hinglish, voice="hm_omega", speed=1.0)
        ]
        if chunks:
            audio = np.concatenate(chunks)
            print(f"  OK  {len(audio) / 24000:.2f}s audio produced for mixed script")
        else:
            print("  FAILED (no audio)")
    except Exception as exc:
        print(f"  FAIL {type(exc).__name__}: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
