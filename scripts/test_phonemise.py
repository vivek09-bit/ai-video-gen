"""Verify per-word phoneme alignment across languages.

Word timings are derived from per-word phonemes, so if the alignment between
input words and phoneme groups is wrong the captions will drift. This measures
the alignment rate rather than assuming it works.

Run with:  .venv\\Scripts\\python.exe scripts\\test_phonemise.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Devanagari cannot be encoded by the default Windows console codepage.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phonemise import Phonemiser, estimate_syllables, resolve_language  # noqa: E402

CASES = [
    ("hi", "रात के दो बजे थे। गाँव के बाहर वाली पुरानी हवेली से आवाज़ आ रही थी।", "hm_omega"),
    ("hi", "कोई नहीं जानता था कि उस घर में कौन रहता है।", "hf_alpha"),
    ("hi", "मैंने दरवाज़ा खोला और अंदर चला गया।", "hm_omega"),
    ("en", "The pressure at four thousand meters would crush a steel submarine.", "am_michael"),
    ("en", "Two thirds of our planet is ocean, and almost all of it is dark.", "bm_george"),
]

engine = Phonemiser()
aligned_total = 0

print(f"{'lang':<5} {'voice':<11} {'words':>5} {'phonemes':>8} {'syll':>5}  status")
print("-" * 78)

for expected_lang, text, voice in CASES:
    lang = resolve_language(voice)
    words = [w for w in text.split() if w]
    phonemes = engine.word_phonemes(text, voice)
    weights = engine.weights(text, voice)
    syllables = sum(estimate_syllables(w) for w in words)

    ok = phonemes is not None and len(phonemes) == len(words)
    if ok:
        aligned_total += 1
    status = "aligned" if ok else "FALLBACK to syllables"

    print(
        f"{lang:<5} {voice:<11} {len(words):>5} "
        f"{(len(phonemes) if phonemes else 0):>8} {syllables:>5}  {status}"
    )

print()
print("--- worked example (Hindi) ---")
text, voice = CASES[0][1], CASES[0][2]
words = [w for w in text.split() if w]
phonemes = engine.word_phonemes(text, voice)
weights = engine.weights(text, voice)

if phonemes and len(phonemes) == len(words):
    total = sum(weights)
    cursor = 0.0
    duration = 6.0
    for word, phoneme, weight in zip(words, phonemes, weights):
        share = weight / total
        start, end = cursor, cursor + share * duration
        cursor = end
        print(
            f"  {start:5.2f}-{end:5.2f}s  {word:<12} {phoneme:<22} "
            f"weight {weight:4.2f} ({share * 100:4.1f}%)"
        )
    print(f"  (positions for a hypothetical {duration:.0f}s sentence)")

print()
print(f"{aligned_total}/{len(CASES)} sentences reached phoneme alignment")
if aligned_total < len(CASES):
    print("Some fell back to syllable counting, which is still functional.")

# Sanity: weights must be positive and finite.
for _lang, sample, sample_voice in CASES:
    ws = engine.weights(sample, sample_voice)
    n = len([w for w in sample.split() if w])
    assert len(ws) == n, f"weight count {len(ws)} != word count {n}"
    assert all(w > 0 for w in ws), "non-positive weight"
    assert all(w == w for w in ws), "NaN weight"
print("Weight invariants hold (one positive weight per word).")
