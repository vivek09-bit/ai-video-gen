# Implementation Plan — Audio Guards + Cache Purge

**Goal:** stop a *silent-but-valid* narration WAV from ever reaching the render, and make
the pipeline recover on its own when the Vast.ai Indic Parler-TTS server returns corrupt audio.

**One-line summary of the bug:** the GPU returned WAVs made of constant `−32768` (NaN
encoded as DC). Nothing validated the audio, so the silent result was cached on disk and
reused by every later run — hence "no voice, not even in any audio generated".

---

## Root cause (evidence, 2026-10-13)

| Layer | Finding |
|---|---|
| `public/audio/voice/scene-001.wav` | 23.58 s, peak −1.5 dBFS, but per‑100 ms profile is ~all zeros with isolated single‑sample spikes → **silence** (mastering removed the DC, leaving step edges). |
| `out/gpu_audio_cache/*.wav` | **16 of 17** segments have `peak == RMS == 32768` → every sample is `−32768` (`0x8000`, raw bytes `00 80 00 80 …`). One file is fine. |
| `scripts/vast_server.py` | `handle_audio_synthesize*` calls `infer_process(...)` then `sf.write(..., subtype="PCM_16")` with **no finiteness check**. libsndfile encodes NaN as `−32768` → a "valid" DC WAV. |
| `scripts/vast_manager.py` | `synthesize_tts*` only checks `status == "success"`, then `base64.b64decode`s. **No amplitude/silence check.** |
| `scripts/tts.py` | `write_gpu_cache` stores the bytes verbatim; `prebuild_vast_batch` skips any sentence whose cache file exists; key = `sha256(AUDIO_PIPELINE_VERSION, voice, speed, text)` so the poison is reused forever. |

**Why it renders silent:** the mastering chain (`highpass 85 Hz` + compressor + peak
normalize) strips the DC; the leftover edges at segment joins become the spikes. The
Kokoro fallback never fires because the remote call **succeeded** — it just returned junk.

---

## How to use this file

| Mark | Meaning |
|---|---|
| `[ ]` | Todo — not implemented |
| `[~]` | In progress |
| `[x]` | Done — skip this step |
| `[!]` | Blocked (note why underneath) |

Do the phases in order. **Phase 1 recovers today's broken video immediately**; Phases 2–5
are the permanent guards. Every step is independently shippable.

### Implementation status (2026-10-13)

| Phase | Status |
|---|---|
| 1 — purge cache + version bump | ✅ **done** — `AUDIO_PIPELINE_VERSION` 11 → 12; 187 files purged |
| 2 — shared `audio_is_bad()` | ✅ **done** — `scripts/tts.py` |
| 3 — server guard | ✅ **done** — `vast_server._audio_is_bad`, both audio handlers |
| 4 — client transport guard | ✅ **done** — `vast_manager._wav_is_bad`, `synthesize_tts*` |
| 5 — cache guards | ✅ **done** — `write_gpu_cache`, `read_gpu_cache`, `prebuild_vast_batch`, `synthesize_indic_parler` |
| 6 — regression tests | ✅ **done** — `test_audio.py` + `test_providers.py` fixture fixed; all 4 suites green |
| 7 — `--purge-audio-cache` | ✅ **done** — `tts.py` + `new_video.py` pass-through |
| 8 — fp16 NaN root cause | ⬜ **pending** (follow-up; guards make it non-urgent) |
| 9 — docs + repo memory | ✅ **done** — README + `/memories/repo/vast-gotchas.md` |

Guard rule used throughout — a waveform is **bad** if any of:

1. empty (`size == 0`), or
2. `not np.all(np.isfinite(x))` (NaN / inf), or
3. `peak = max(|x|) < 1e-4` (silence), or
4. `std(x) < 1e-5` (constant / pure DC).

Real narration peaks well above `1e-4` (master targets `−18 dBFS`), so these thresholds
won't reject good audio.

---

## Phase 1 — Recover now: purge the poisoned cache + bump the pipeline version

**What:** make every future run ignore the corrupt GPU segments.

**Where:** `scripts/tts.py`

**Change:**
- Bump `AUDIO_PIPELINE_VERSION = 11` → `12` (line ~83). Both the GPU segment key
  (`gpu_cache_key`) and the scene sidecar key (`build_line`, ~line 2298) include this, so
  old keys no longer match and every scene is re-synthesised.
- Delete the corrupt store: remove all `out/gpu_audio_cache/*.wav`.
- (Belt-and-braces) delete the stale scene sidecars `public/audio/voice/*.json` so nothing
  can short-circuit the rebuild even if a sidecar survived.

**Command (one-off, safe to re-run):**

```powershell
Remove-Item -Recurse -Force out\gpu_audio_cache -ErrorAction SilentlyContinue
Remove-Item -Force public\audio\voice\*.json -ErrorAction SilentlyContinue
```

**Accept:** `out/gpu_audio_cache` is gone; `tts.py` reports `AUDIO_PIPELINE_VERSION = 12`.

**Verify:**

```powershell
.venv\Scripts\python.exe -c "import sys; sys.path.insert(0,'scripts'); import tts; print(tts.AUDIO_PIPELINE_VERSION)"
# -> 12
```

> Note: bumping the version alone is *not* enough on its own — the corrupt files would
> linger on disk. Purge is required so disk usage doesn't keep 100s of MB of dead WAVs.

---

## Phase 2 — Shared bad-audio detector

**What:** one canonical helper so every guard agrees on what "bad" means.

**Where:** `scripts/tts.py`, near the other speech helpers (after
`is_permanent_remote_tts_error`, ~line 1670).

**Change:** add

```python
def audio_is_bad(wav) -> bool:
    """True when a waveform is empty, non-finite, silent or pure DC.

    Guards the whole narration path: the Vast.ai server can return a WAV that is
    structurally valid but full of constant -32768 (NaN encoded as DC). Catching it
    here keeps silent audio out of the disk cache and out of the final render.
    """
    import numpy as np
    x = np.asarray(wav, dtype=np.float64).reshape(-1)
    if x.size == 0 or not np.all(np.isfinite(x)):
        return True
    if np.max(np.abs(x)) < 1e-4:   # silence
        return True
    if np.std(x) < 1e-5:           # constant / DC
        return True
    return False
```

**Accept:** `audio_is_bad(np.full(1000, -32768, dtype=np.int16))` → `True`;
`audio_is_bad(speech_like_float)` → `False`.

**Verify:** covered by Phase 6 tests.

---

## Phase 3 — Server-side guard (stop producing the corrupt WAV)

**What:** refuse to emit a WAV whose samples are non-finite/constant; report an error so
the client retries or falls back instead of caching junk.

**Where:** `scripts/vast_server.py`

**Change:**
- Add a small local `_audio_is_bad()` (numpy only — the server must not depend on `tts.py`
  imports succeeding) mirroring the Phase 2 rule.
- In `handle_audio_synthesize` (~line 494): after `audio_np = engine.synthesize(...)`,
  before `sf.write`, add

  ```python
  audio_np = np.asarray(audio_np, dtype=np.float32)
  if _audio_is_bad(audio_np):
      self._send_json(
          {"status": "error",
           "error": "Indic Parler-TTS produced non-finite/constant audio (possible fp16 NaN)"},
          status=500,
      )
      return
  ```
- In `handle_audio_synthesize_batch` (~line 553), wrap per item: if `_audio_is_bad`, append
  `{"id": entry_id, "error": "non-finite/constant audio"}` and `continue` (the client treats
  it as a missing segment and retries it individually).

**Accept:** a NaN/DC generation returns HTTP 500 (single) or a per-item `error` (batch),
never a base64 WAV.

**Verify:** unit-call the helper; then a live re-run should show batch warnings instead of
silent segments.

---

## Phase 4 — Client transport guard

**What:** validate at the boundary so a bad payload is rejected *before* it is written to
the disk cache.

**Where:** `scripts/vast_manager.py` (`synthesize_tts` ~1645, `synthesize_tts_batch` ~1672)

**Change:**
- Add a module-level `_wav_is_bad(wav_bytes: bytes) -> bool` that decodes with
  `soundfile` into float32 and applies the Phase 2 rule.
- `synthesize_tts`: after `audio_bytes = base64.b64decode(...)`, if `_wav_is_bad(audio_bytes)`
  → `raise RuntimeError("Remote Indic Parler-TTS returned silent/non-finite audio")` (this makes
  the existing retry/fallback logic in `tts.synthesize_indic_parler` engage).
- `synthesize_tts_batch`: in the results loop, if `_wav_is_bad(decoded)` → append to
  `errors` and **omit** the entry from `out` (caller then retries it individually and, on
  repeat failure, falls back to Kokoro).

**Accept:** corrupt segments raise (single) or are omitted + counted in the warning (batch).

**Verify:** Phase 6 test feeds a `−32768` WAV through `_wav_is_bad`.

---

## Phase 5 — Cache guards in `tts.py`

**What:** the disk cache is the reason the bug persisted. Make it refuse bad entries on
both write and read.

**Where:** `scripts/tts.py`

**Change:**
- `write_gpu_cache` (~1707): decode `wav_bytes`; if `audio_is_bad` → print a
  `[Indic Parler Cache Warning]` and **return without writing**.
- `read_gpu_cache` (~1692): after `sf.read`, if `audio_is_bad(wav_np)` → delete the file
  (`path.unlink(missing_ok=True)`), warn, and `return None`.
- `prebuild_vast_batch` (~1794): before `write_gpu_cache(sentence, …)`, decode and check
  `audio_is_bad`; if bad, append `sentence` to `missing` (retry individually) and skip the
  cache write.
- `synthesize_indic_parler` remote branch (~1095): after `sf.read(...)`, if `audio_is_bad(wav_np)`
  → do **not** cache; raise/continue so the loop retries, and ultimately return `None`
  (→ Kokoro fallback). Guarding here means a bad segment can never populate
  `_VAST_AUDIO_CACHE` either.

**Accept:** no bad WAV is ever written to `out/gpu_audio_cache`, and any pre-existing bad
entry is deleted on first read.

**Verify:** Phase 6 tests + a live run leaves `out/gpu_audio_cache` populated only with
`audio_is_bad == False` files.

---

## Phase 6 — Regression tests

**What:** lock the behaviour so a "successful but silent" synthesis fails CI instead of
shipping a muted video.

**Where:** `scripts/test_audio.py`

**Change:** add a block (before the final summary) that:

1. Asserts `tts.audio_is_bad()` is `True` for:
   - constant `−32768` (`np.full(2400, -32768, np.int16)`),
   - all-NaN (`np.full(2400, np.nan, np.float32)`),
   - all-zero silence,
   - empty array.
2. Asserts it is `False` for `speech_like(-20)` (existing helper, 2 s of speech-shaped noise).
3. Asserts `gpu_cache_path(...)` + `write_gpu_cache(...)` with a `−32768` WAV leaves **no**
   file on disk.
4. Asserts `read_gpu_cache(...)` on a file hand-written with `−32768` returns `None` **and
   deletes it**.

**Accept:** `scripts/test_audio.py` exits 0 and the new asserts run.

**Verify:**

```powershell
npm run test          # or: .venv\Scripts\python.exe scripts\test_audio.py
```

Also confirm the other three suites stay green (`test_providers.py`, `test_phonemise.py`,
`test_story.py`) and `get_errors` is clean on the edited files.

---

## Phase 7 — One-command purge (optional but recommended)

**What:** make recovery a first-class command instead of a manual `Remove-Item`.

**Where:** `scripts/tts.py` (argparse in `main`) + `scripts/new_video.py` (pass-through) +
`package.json` / `.vscode/tasks.json`.

**Change:**
- Add `--purge-audio-cache` to `tts.py`: deletes `out/gpu_audio_cache/*.wav` and the
  `public/audio/voice/*.json` sidecars, prints the count, exits 0.
- Add the same flag to `new_video.py` so `npm run new -- … --purge-audio-cache` purges then
  rebuilds in one go.
- Optional: when `audio_is_bad` rejects ≥1 segment in a run, print a hint:
  `"run again with --purge-audio-cache"`.

**Accept:** `npm run voice -- --purge-audio-cache` clears both stores.

**Verify:** run it, then confirm `out/gpu_audio_cache` is empty.

---

## Phase 8 — Root cause: fp16 NaN in the zero-shot model (follow-up)

**What:** the guards stop the bleeding; this removes the source of the NaN.

**Where:** `scripts/vast_server.py` (`load_indic_parler_engine`, `SimpleIndicParler.synthesize`)

**Change (candidates, test one at a time):**
- Load the CFM model in `float32` instead of `float16` on CUDA (line ~257:
  `torch.float16 if DEVICE == "cuda"`), or
- Keep fp16 but re-run `nfe_step`/`cfg_strength` with a finite-value retry: if
  `_audio_is_bad`, retry once in fp32 before failing.
- Clamp text length / split very long sentences (OOD inputs make F5-TTS unstable).

**Accept:** repeated batch runs produce zero `_audio_is_bad` segments.

**Verify:** run a 1‑minute narration 3× and confirm every returned segment is finite and
non-constant.

> This is a **likely** fix, not a certain one — measure it. Phases 2–6 make the pipeline
> safe regardless of whether Phase 8 fully eliminates the NaN.

---

## Phase 9 — Documentation

**What:** record the failure mode so it isn't re-discovered.

**Where:** `/memories/repo/vast-gotchas.md`, `README.md` (troubleshooting section)

**Change:**
- Add a gotcha entry: "The old fp16 F5-style model can emit NaN → `−32768` DC WAV; guards are
  `tts.audio_is_bad`, `vast_manager._wav_is_bad`, `vast_server._audio_is_bad`; recover with
  `--purge-audio-cache`."
- README: a one-line "no voice in the output" troubleshooting pointer.

**Accept:** a future reader can go from symptom → cause → fix in one place.

**Verify:** read both edits back.

---

## End-to-end verification (after Phases 1–6)

```powershell
# 1. purge + version bump already done in Phase 1
# 2. rebuild the same topic
npm run new -- --topic "पुरानी हवेली का राज़" --minutes 1

# 3. assert the narration actually contains speech
.venv\Scripts\python.exe -c "import wave,array;w=wave.open('public/audio/voice/scene-001.wav','rb');a=array.array('h');a.frombytes(w.readframes(w.getnframes()));print('peak',max(abs(x) for x in a))"
#    -> peak should be > 0 AND not the *only* non-zero value; RMS should be meaningfully > 0

# 4. inspect the rendered track
C:\ffmpeg\ffprobe.EXE -v error -show_entries stream=codec_type -of csv out\video.mp4
C:\ffmpeg\ffmpeg.EXE -hide_banner -i out\video.mp4 -af volumedetect -f null NUL 2>&1 | Select-String "mean_volume"
#    -> mean_volume must be well above the ~-90 dB floor and the speech must be audible
```

**Pass criteria**
- [ ] `out/gpu_audio_cache/*.wav` contains only segments where `tts.audio_is_bad() is False`.
- [ ] `public/audio/voice/scene-001.wav` is speech (not silence + spikes).
- [ ] The rendered MP4 has clearly audible narration.
- [ ] All four test suites green; `get_errors` clean.

---

## Rollback

All changes are additive guards + a version bump. To revert to the old behaviour:
- set `AUDIO_PIPELINE_VERSION` back to `11`, and
- remove the `audio_is_bad` calls (the helpers can stay unused).

The purge is destructive only to regenerable cache, so no data is lost.

---

## Change map (files touched)

| Phase | File | Symbol |
|---|---|---|
| 1 | `scripts/tts.py` | `AUDIO_PIPELINE_VERSION`; `out/gpu_audio_cache` |
| 2 | `scripts/tts.py` | `audio_is_bad()` (new) |
| 3 | `scripts/vast_server.py` | `_audio_is_bad()`, `handle_audio_synthesize`, `handle_audio_synthesize_batch` |
| 4 | `scripts/vast_manager.py` | `_wav_is_bad()`, `synthesize_tts`, `synthesize_tts_batch` |
| 5 | `scripts/tts.py` | `write_gpu_cache`, `read_gpu_cache`, `prebuild_vast_batch`, `synthesize_indic_parler` |
| 6 | `scripts/test_audio.py` | new asserts |
| 7 | `scripts/tts.py`, `scripts/new_video.py`, `package.json`, `.vscode/tasks.json` | `--purge-audio-cache` |
| 8 | `scripts/vast_server.py` | `load_indic_parler_engine`, `SimpleIndicParler.synthesize` |
| 9 | `README.md`, repo memory | troubleshooting note |
