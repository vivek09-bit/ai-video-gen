"""Verify the narration prosody and mastering chain.

These are the parts of ``tts.py`` that decide whether a long video sounds like a
narrator or like a machine, so they are measured rather than assumed. Everything
here runs on synthetic audio, which keeps the test fast and independent of
Kokoro; the model-level numbers were measured separately and are recorded in the
README.

Run with:  .venv\\Scripts\\python.exe scripts\\test_audio.py
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import tts  # noqa: E402

TARGET_RMS = tts.DEFAULT_MASTER["targetLoudnessDbfs"]
CEILING = tts.DEFAULT_MASTER["peakCeilingDbfs"]


def dbfs(value: float) -> float:
    return 20 * np.log10(max(abs(float(value)), 1e-12))


def rms_dbfs(x) -> float:
    return dbfs(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def speech_like(level_dbfs: float, seconds: float = 2.0, seed: int = 0):
    """Noise shaped by a syllable-rate envelope, so it varies like speech does."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(tts.SAMPLE_RATE * seconds)) / tts.SAMPLE_RATE
    envelope = 0.4 + 0.6 * np.sin(2 * np.pi * 4.0 * t) ** 2
    x = rng.standard_normal(t.size) * envelope
    return (x * 10 ** (level_dbfs / 20.0) / np.sqrt(np.mean(x**2))).astype(np.float32)


# ---------------------------------------------------------------------------
# Pauses carry the punctuation
# ---------------------------------------------------------------------------
print(f"{'mark':<8} {'meaning':<22} {'pause':>7}  scale")
print("-" * 54)
MARKS = [
    (",", "clause break"),
    (":", "sub-clause"),
    (".", "full stop"),
    ("\u0964", "danda (Hindi stop)"),
    ("\u0965", "double danda"),
    ("!", "exclamation"),
    ("?", "question"),
]
for mark, meaning in MARKS:
    value = tts.pause_after("some words" + mark, 0.30, 0)
    print(f"{mark!r:<8} {meaning:<22} {value:6.3f}s  {value / 0.30:4.2f}")

question = tts.pause_after("x?", 0.30, 0)
stop = tts.pause_after("x.", 0.30, 0)
comma = tts.pause_after("x,", 0.30, 0)
assert question > stop > comma, "punctuation is not shaping the pause"
assert tts.pause_after("x.", 0.30, 5) == tts.pause_after("x.", 0.30, 5), "pause is not deterministic"
wobble = {round(tts.pause_after("x.", 0.30, i), 6) for i in range(12)}
assert len(wobble) >= 8, f"pause wobble collapsed to {len(wobble)} values"
assert max(wobble) / min(wobble) < 1.2, "wobble is large enough to hear as a stutter"
print(f"pauses vary by mark and wobble over {len(wobble)} deterministic values.\n")

# ---------------------------------------------------------------------------
# Fades
# ---------------------------------------------------------------------------
tone = np.ones(tts.SAMPLE_RATE, dtype=np.float32)
faded = tts.apply_fades(tone)
assert abs(faded[0]) < 1e-6 and abs(faded[-1]) < 1e-6, "segment ends are not faded"
assert faded[len(faded) // 2] == 1.0, "fade reached the middle of the segment"
assert np.all(np.diff(faded[: tts.SAMPLE_RATE // 2]) >= -1e-6), "fade is not monotonic"
print("fades: both ends reach zero, midpoint untouched, ramp is monotonic.")

# ---------------------------------------------------------------------------
# Mastering
# ---------------------------------------------------------------------------
print()
print(f"{'input':>8} {'raw rms':>9} {'out rms':>9} {'out peak':>9} {'crest':>7}  status")
print("-" * 62)
OUTPUTS = []
for level in (-30.0, -24.0, -20.0, -14.0, -6.0):
    raw = speech_like(level, seed=int(-level))
    out = tts.master_audio(raw)
    OUTPUTS.append(out)

    peak = dbfs(np.max(np.abs(out)))
    assert not np.isnan(out).any(), f"NaN from level {level}"
    assert peak <= CEILING + 0.05, f"ceiling breached at level {level}: {peak:.2f} dBFS"

    crest = peak - rms_dbfs(out)
    print(f"{level:>7.0f} {rms_dbfs(raw):>9.2f} {rms_dbfs(out):>9.2f} "
          f"{peak:>9.2f} {crest:>7.1f}  limited")

levels = [rms_dbfs(o) for o in OUTPUTS]
spread = max(levels) - min(levels)
print(f"\nloudness across a {max(levels) - min(levels):.2f} dB spread of inputs: "
      f"{min(levels):.2f} to {max(levels):.2f} dBFS")
assert spread < 0.3, f"line-to-line loudness drifts by {spread:.2f} dB"
assert abs(np.mean(levels) - TARGET_RMS) < 0.5, f"loudness target missed: {np.mean(levels):.2f}"

# ---------------------------------------------------------------------------
# The limiter must duck a transient, not scale the whole line.
# Scaling a line to its loudest sample is how loudness drifted between lines, so
# this is the regression guard for that specific bug.
# ---------------------------------------------------------------------------
body = (0.1 * np.sin(2 * np.pi * 300 * np.arange(tts.SAMPLE_RATE) / tts.SAMPLE_RATE)).astype(np.float32)
spiked = body.copy()
spiked[1000] = 0.99

limited = tts.limit_peaks(spiked, 10 ** (CEILING / 20.0))
tail = slice(int(tts.SAMPLE_RATE * 0.9), None)
body_change = dbfs(np.sqrt(np.mean(limited[tail] ** 2))) - dbfs(np.sqrt(np.mean(spiked[tail] ** 2)))

print(f"\ntransient limiting: peak {dbfs(np.max(np.abs(spiked))):.2f} -> "
      f"{dbfs(np.max(np.abs(limited))):.2f} dBFS, "
      f"rest of the line moved {body_change:+.2f} dB")
assert dbfs(np.max(np.abs(limited))) <= CEILING + 0.05, "limiter let the transient through"
assert abs(body_change) < 0.2, f"limiter dragged the whole line down by {body_change:.2f} dB"

# ---------------------------------------------------------------------------
# Compression evens out passage level without flattening transients.
# Loudness is measured around each signal's own speech level, so the reading is
# gain-invariant rather than reflecting how loud the input happened to be.
# ---------------------------------------------------------------------------
def passage_spread(x) -> float:
    block = int(0.02 * tts.SAMPLE_RATE)
    frames = np.asarray(x, dtype=np.float64)[: len(x) // block * block].reshape(-1, block)
    energy = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    db = 20 * np.log10(np.maximum(energy, 1e-12))
    voiced = db[db > np.percentile(db, 90) - 25.0]
    return float(np.std(voiced - np.mean(voiced)))


raw = speech_like(-20.0, seconds=4.0, seed=7)
compressed = tts.compress_segments(raw, tts.DEFAULT_MASTER["thresholdDbfs"], tts.DEFAULT_MASTER["ratio"])
before, after = passage_spread(raw), passage_spread(compressed)
print(f"compression: within-line span {before:.2f} dB -> {after:.2f} dB")
assert after < before, "compressor is not evening out the passage level"
assert compressed.shape == raw.shape, "compressor changed the length"

# ---------------------------------------------------------------------------
# Word timings survive the new gaps, and mastering can be switched off.
# ---------------------------------------------------------------------------
class FakeEngine:
    """Stands in for Kokoro: 0.25s per word, weights from word length.

    Returns low-level noise rather than silence, so the mastering chain has an
    actual signal to work on and the duration still comes from the word count.
    """

    class _Phonemiser:
        def weights(self, text, voice):
            return [float(len(w)) + 1.0 for w in text.split()]

    phonemiser = _Phonemiser()

    def say(self, text, voice, speed):
        length = int(tts.SAMPLE_RATE * 0.25 * len(text.split()))
        rng = np.random.default_rng(len(text))
        return (0.05 * rng.standard_normal(length)).astype(np.float32)


def test_trim_kokoro_edge_padding():
    rng = np.random.default_rng(7)
    speech = (0.05 * rng.standard_normal(2 * tts.SAMPLE_RATE)).astype(np.float32)
    padded = np.concatenate(
        [
            np.zeros(int(0.5 * tts.SAMPLE_RATE), dtype=np.float32),
            speech,
            np.zeros(int(0.4 * tts.SAMPLE_RATE), dtype=np.float32),
        ]
    )
    trimmed = tts.trim_edge_silence(padded)
    duration = len(trimmed) / tts.SAMPLE_RATE
    assert 2.01 <= duration <= 2.03, f"edge padding not trimmed cleanly: {duration:.3f}s"
    assert len(tts.trim_edge_silence(np.zeros(2400, dtype=np.float32))) == 2400


test_trim_kokoro_edge_padding()


TEXT = "\u092a\u0939\u0932\u093e \u0935\u093e\u0915\u094d\u092f \u092f\u0939\u093e\u0901 \u0939\u0948\u0964 \u0926\u0942\u0938\u0930\u093e \u0935\u093e\u0915\u094d\u092f \u092f\u0939\u093e\u0901 \u0939\u0948?"
spoken = TEXT.split()
audio, words = tts.synthesise_line(FakeEngine(), TEXT, "hm_omega", 1.0, 0.30)

assert len(words) == len(spoken), f"{len(words)} timings for {len(spoken)} words"
assert all(w["start"] < w["end"] for w in words), "empty word span"
assert all(words[i]["end"] <= words[i + 1]["start"] + 1e-6 for i in range(len(words) - 1)), "words overlap"
assert words[0]["start"] == 0.0, "first word does not start at zero"
assert abs(rms_dbfs(audio) - TARGET_RMS) < 1.0, "line was not mastered"

expected = 0.25 * len(spoken) + tts.pause_after("पहला वाक्य यहाँ है।", 0.30, 0)
actual = len(audio) / tts.SAMPLE_RATE
assert abs(actual - expected) < 0.02, f"danda pause missing: {actual:.3f}s vs {expected:.3f}s"

plain = tts.synthesise_line(FakeEngine(), TEXT, "hm_omega", 1.0, 0.30, {"enabled": False})[0]
assert rms_dbfs(plain) < rms_dbfs(audio) - 1.0, "mastering cannot be switched off"

print(f"word timings: {len(words)} words, no overlaps, danda pause of "
      f"{tts.pause_after('पहला वाक्य यहाँ है।', 0.30, 0):.2f}s inserted, "
      f"line {actual:.2f}s.")
print("mastering can be disabled and returns the raw level.\n")

# ---------------------------------------------------------------------------
# Hindi Number Expansion & Phonetics Pre-normalization
# ---------------------------------------------------------------------------
assert tts.num_to_hindi(3) == "तीन"
assert tts.num_to_hindi(50) == "पचास"
assert tts.num_to_hindi(1990) == "एक हज़ार नौ सौ नब्बे"
assert tts.expand_hindi_numbers("रात ३ बजे ५० साल पहले") == "रात तीन बजे पचास साल पहले"
assert tts.expand_hindi_numbers("रात 3 बजे 50 साल पहले") == "रात तीन बजे पचास साल पहले"
assert "पेहला" in tts.normalize_hindi_phonetics("पहला")
assert "वो" in tts.normalize_hindi_phonetics("वह")
print("Hindi numbers and phonetic normalization verified.")

# ---------------------------------------------------------------------------
# Corrupt-audio guards (silent / NaN / DC -> rejected, never cached)
# ---------------------------------------------------------------------------
import io  # noqa: E402

import soundfile as sf  # noqa: E402


def wav_bytes(samples) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.asarray(samples), tts.SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()


dc = np.full(tts.SAMPLE_RATE, -32768, dtype=np.int16)  # NaN encoded by libsndfile as DC
nan = np.full(tts.SAMPLE_RATE, np.nan, dtype=np.float32)
silence = np.zeros(tts.SAMPLE_RATE, dtype=np.float32)

assert tts.audio_is_bad(dc) is True, "constant -32768 (DC) must be flagged bad"
assert tts.audio_is_bad(nan) is True, "all-NaN must be flagged bad"
assert tts.audio_is_bad(silence) is True, "silence must be flagged bad"
assert tts.audio_is_bad(np.zeros(0, dtype=np.float32)) is True, "empty must be flagged bad"
assert tts.audio_is_bad(speech_like(-20)) is False, "real audio must not be flagged bad"

# A corrupt segment must never reach the disk cache...
bad_path = tts.gpu_cache_path("guard-test-bad", "indicf5_aarav", 1.0)
bad_path.unlink(missing_ok=True)
tts.write_gpu_cache("guard-test-bad", "indicf5_aarav", 1.0, wav_bytes(dc))
assert not bad_path.exists(), "silent segment was written to the gpu cache"

# ...and an already-cached corrupt segment must be discarded on read.
bad_path.write_bytes(wav_bytes(dc))
assert tts.read_gpu_cache("guard-test-bad", "indicf5_aarav", 1.0) is None
assert not bad_path.exists(), "corrupt cached segment was not discarded"

# A good segment survives the write/read round trip.
ok_path = tts.gpu_cache_path("guard-test-ok", "indicf5_aarav", 1.0)
ok_path.unlink(missing_ok=True)
tts.write_gpu_cache("guard-test-ok", "indicf5_aarav", 1.0, wav_bytes(speech_like(-20, seconds=1.0)))
assert ok_path.exists(), "a valid segment was not cached"
assert tts.read_gpu_cache("guard-test-ok", "indicf5_aarav", 1.0) is not None
ok_path.unlink(missing_ok=True)
print("Corrupt-audio guards verified: silent/NaN/DC rejected on cache write and read.")

print("Audio chain verified: punctuation-aware pauses, fades, gentle compression,")
print(f"loudness held to {TARGET_RMS:.0f} dBFS with peaks under {CEILING:.1f} dBFS.")
