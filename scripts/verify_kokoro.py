#!/usr/bin/env python3
"""Verify Kokoro runs on this machine and measure how fast it is.

Run with:  py scripts/verify_kokoro.py

The first run downloads the model (~350 MB) from Hugging Face, so it will be
noticeably slower than later runs. The realtime factor printed at the end is
what tells us whether a 30-minute video is practical on this CPU.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_RATE = 24000

TEXT = (
    "The ocean floor is not a flat plain. It is a landscape of trenches, "
    "volcanic ridges and abyssal plains that rival anything on land, and most "
    "of it has never been seen by human eyes at all."
)


def main() -> int:
    t0 = time.time()
    from kokoro import KPipeline

    print(f"  import kokoro     : {time.time() - t0:.1f}s")

    t0 = time.time()
    pipeline = KPipeline(lang_code="a")
    print(f"  pipeline init     : {time.time() - t0:.1f}s (includes model download on first run)")

    t0 = time.time()
    chunks: list[np.ndarray] = []
    phonemes = 0
    for graphemes, ps, audio in pipeline(TEXT, voice="af_heart", speed=1.0):
        chunks.append(np.asarray(audio))
        phonemes += len(ps)
        label = graphemes if len(graphemes) <= 44 else graphemes[:41] + "..."
        print(f"    segment | {len(audio):>6} samples | {len(ps):>4} phonemes | {label!r}")
    elapsed = time.time() - t0

    if not chunks:
        print("  ERROR: no audio produced")
        return 1

    audio = np.concatenate(chunks)
    duration = len(audio) / SAMPLE_RATE
    rtf = elapsed / duration
    print(f"  synthesise        : {elapsed:.2f}s wall for {duration:.2f}s audio")
    print(f"  realtime factor   : {rtf:.2f}x  ({duration / elapsed:.1f}x faster than realtime)")
    print(f"  a 30-min video    : ~{30 * rtf:.1f} min of synthesis")

    out = ROOT / "out" / "_kokoro_check.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    import soundfile as sf

    sf.write(str(out), audio, SAMPLE_RATE)
    print(f"  wrote             : {out.relative_to(ROOT)} ({out.stat().st_size / 1024:.0f} KB)")
    print(f"  peak amplitude    : {np.abs(audio).max():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
