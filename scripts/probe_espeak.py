"""Inspect misaki's espeak G2P so Hindi word timing can reuse the English path.

English uses ``misaki.en.G2P``, which returns tokens carrying per-word IPA
phonemes. Hindi cannot use that (it is not an English model), so it goes through
``misaki.espeak.EspeakG2P``. This probe finds out whether that path also
returns phoneme-bearing tokens, because if it does, word timing works exactly
like it does for English.

Run with:  .venv\\Scripts\\python.exe scripts\\probe_espeak.py
"""

from __future__ import annotations

from misaki import espeak

SAMPLE = "रात के दो बजे थे। गाँव के बाहर वाली पुरानी हवेली से आवाज़ आ रही थी।"

print("--- available languages ---")
try:
    voices = espeak.EspeakG2P(language="hi")
    print(f"  EspeakG2P constructed for 'hi'")
except Exception as exc:
    print(f"  FAIL {type(exc).__name__}: {exc}")
    raise SystemExit(1)

phonemes, tokens = voices(SAMPLE)
print()
print(f"--- result ---")
print(f"  phonemes: {phonemes[:90]}")
print(f"  token count: {len(tokens)}")
print()

if tokens:
    attributes = sorted(a for a in dir(tokens[0]) if not a.startswith("_"))
    print(f"--- token attributes ---")
    print(f"  {attributes}")
    print()
    print("--- per token ---")
    for token in tokens:
        text = getattr(token, "text", None)
        ph = getattr(token, "phonemes", None)
        start_ts = getattr(token, "start_ts", None)
        print(f"  text={text!r:<26} phonemes={ph!r:<26} start_ts={start_ts}")

    has_phonemes = any(getattr(t, "phonemes", None) for t in tokens)
    has_timing = any(getattr(t, "start_ts", None) is not None for t in tokens)
    print()
    print(f"  tokens carry phonemes : {has_phonemes}")
    print(f"  tokens carry timings  : {has_timing}")
    print(
        "  -> "
        + (
            "same weighting strategy as English will work"
            if has_phonemes
            else "must fall back to character-count weighting"
        )
    )
else:
    print("  no tokens returned; character-count fallback will be needed")

print()
print("--- a few other codes, for reference ---")
for code in ("hi", "en-us", "es", "fr", "ja"):
    try:
        engine = espeak.EspeakG2P(language=code)
        out, tok = engine("hello world")
        print(f"  {code:<6} OK  {len(tok)} tokens")
    except Exception as exc:
        print(f"  {code:<6} FAIL {type(exc).__name__}")
