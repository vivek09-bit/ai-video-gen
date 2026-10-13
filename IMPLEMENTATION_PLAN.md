# Implementation Plan — One-Command Pipeline

**Goal:** a single command that runs

> research → keywords → write script → build image/video prompts →
> *(if IndicF5)* rent Vast RTX 3090 → load model → generate voice → store locally →
> render the final video on CPU.

**Usage target (after all steps are done):**

```powershell
npm run new -- --topic "पुरानी हवेली का राज़" --minutes 10 `
  --audio-model indicf5 --voice indicf5_aarav --imageprovider local
```

---

## How to use this file

Mark each step as you finish it so later runs can skip it:

| Mark | Meaning |
|---|---|
| `[ ]` | Todo — not implemented |
| `[~]` | In progress |
| `[x]` | **Done — skip this step** |
| `[!]` | Blocked (note why underneath) |

Every step has **What / Where / Change / Accept / Verify** so it is self-contained and
re-runnable. Do the steps in order; each is independently shippable.

---

## Implementation status (2026-10-13)

| Phase | Status |
|---|---|
| 1 — persist research | ✅ **done** |
| 2 — keywords → visual prompts | ✅ **done** |
| 3 — keywords → SEO tags | ✅ **done** |
| 4 — `--stop-after audio` / `metadata` | ✅ **done** |
| 4B — boot-stall policy | ✅ **Option B** chosen, documented and locked by a test |
| 5 — theme → visual prompts | ✅ **done** |
| 6 — Vast SSH path | ✅ **resolved — HTTP-only documented** (no SSH keys on this account) |
| 7 — prove the flow live | 🟡 **partial** — `--stop-after audio` (Kokoro) verified; GPU run pending |
| 8 — docs / canonical command | ✅ **done** (README flags + render-engine note + batch docstring) |

All four suites are green (`npm test`); `test_providers.py` now also runs
`test_vast_boot_stall_policy()`.

### What is left (code-checked 2026-10-13)

**Done in this pass**
1. **Phase 8 (docs).** ✅ README "Useful flags" now has `--stop-after` and `--render-on` rows,
   plus a render-engine note (Remotion / bundled ffmpeg on CPU); `batch.py` docstring updated.
2. **Phase 5 (theme → prompts).** ✅ `THEME_PROMPT_STYLE` + `theme_style_fragment()` in
   `generate_images.py`; `make_prompt(..., theme=…)`; both generators pass `script["theme"]`.
3. **Phase 6 (SSH).** ✅ Resolved as **HTTP-only** — README states `--render-on vast` / repair /
   voice-upload cannot run without Vast SSH keys on this account.
4. **Batch refactor** — deliberately skipped (documented as available in the docstring).

**Still to verify (no code)**
5. **Phase 7 (live GPU run)** — rent 3090 → IndicF5 → audio → local render, and confirm the
   instance finishes **stopped (not terminated)**. Needs a Vast account + credit + LLM key.
6. `--stop-after metadata` not yet exercised at runtime.
7. Phase 1/3 with research **enabled** (needs an LLM key): confirm `script.json` gains
   `research` + `keywords`, `out/research.json` exists, and `out/metadata.json` tags include
   research terms.

**Verified**
- `--no-research --stop-after write` runs green offline and writes `script.json`.
- `--stop-after audio` on local Kokoro stops after narration (no render) — **verified**.
- All four suites green; `get_errors` clean on all edited files.
- `theme_style_fragment()` returns the right fragment per theme (and `''` for auto/random).

> Line numbers quoted elsewhere in this file predate the Phase 1-4 edits and have shifted
> (e.g. the `build_script` call, the `--stop-after` argparse block, the metadata call site).
> Treat them as approximations — the status notes here are authoritative.

---

---

## Legend of the current architecture (verified by reading the code)

| Concern | File(s) | Key symbols |
|---|---|---|
| Entry commands | `scripts/new_video.py`, `scripts/batch.py` | `main()`, `run_step()`, batch phase functions |
| Research / keywords | `scripts/research.py` | `fetch_youtube_suggestions`, `fetch_google_suggestions`, `generate_keyword_clusters`, `research_viral_angles`, `format_research_report` |
| Script writing | `scripts/new_video.py`, `scripts/beats.py` | `build_prompt(research=…)`, `parse_written`, `build_script`, `resolve_llm_writer`, `ask_llm` |
| Visual prompts | `scripts/generate_images.py`, `scripts/generate_videos.py` | `extract_script_visual_keywords`, `build_search_queries`, `make_prompt`, `generate_images_for_script`, `generate_videos_for_script` |
| GPU / IndicF5 | `scripts/vast_manager.py`, `scripts/vast_server.py`, `scripts/tts.py` | `VastManager.provision_instance`, `find_existing_instance`, `require_gpu`, `prebuild_vast_batch`, `synthesize_indicf5` |
| Metadata / SEO | `scripts/metadata.py` | `build_seo_tags`, `generate_metadata`, `generate_metadata_procedural` |
| Render | `package.json`, `scripts/vast_render.py` | `npx remotion render Main …`, `render_on_vast` |
| Tests | `scripts/test_*.py` | `test_providers.py`, `test_audio.py`, `test_story.py`, `test_phonemise.py` |

### Already implemented — **do not rebuild**

- ✅ Research + live keyword clusters (`research.py`) and default invocation from `new_video.py`.
- ✅ `--auto-topic` / `--research-only` flows.
- ✅ Research + concepts injected into the writer brief (`build_prompt(research=…)`).
- ✅ Visual prompt factories and multi-provider image generation.
- ✅ LTX-Video generation for `--visual-mode video`.
- ✅ IndicF5 auto-rent of RTX 3090 (4090 only for video mode), budget caps, bad-host
  blacklist, warm reuse, onstart install (`uv` + torch/numpy/pyarrow/torchaudio pins) +
  HF prefetch of `rsolanki1822/IndicF5-mirror`.
- ✅ Remote TTS API (`/health`, `/v1/audio/synthesize`, `/v1/audio/synthesize_batch`) and
  local disk cache `out/gpu_audio_cache` keyed by `(text, voice, speed)`.
- ✅ `batch.py` holds the GPU only for the audio phase, stops once, renders on CPU.
- ✅ CPU render via Remotion (bundled ffmpeg). `--render-on vast` exists (unverified).
- ✅ **End-of-run GPU is STOPPED, not terminated** (see the invariant below).

---

## GPU lifecycle invariant (hard requirement)

> **After execution the GPU instance must be STOPPED, never terminated (destroyed).**

**Status: `[x]` already implemented — verify only, do not rebuild.**

Why it matters: a *stopped* instance keeps its container filesystem, so the dependency
install it already paid for (torch + IndicF5 + HF weights, ~20-30 min) survives for cheap
reuse. Destroying it throws that away and the next run re-pays the full setup. A stopped
50GB instance bills storage (~$0.77/day), which is far less than re-running setup.

Where this is implemented (verified):
- `scripts/new_video.py` → `stop_vast_instance()` (L1702-1709) calls `vast_mgr.stop_instance(...)`
  (a **stop**, not a destroy) on every exit path: `--stop-after write` (L1860), TTS failure
  (L1908), `--no-render` (L1926), render failure, and normal completion. The stop at L1967
  runs **before** the upload block, so a later upload failure cannot leave the GPU running.
- `--vast-autostop` defaults to **True** (`new_video.py` ~L1281-1287); `--no-vast-autostop`
  leaves it running on purpose.
- `scripts/batch.py` → `stop()` (~L255) uses `VastManager().stop_instance()`; the GPU is
  stopped once after the audio phase.
- `scripts/vast_manager.py` → `_safe_stop()` uses `stop_instance()`.

**Destroy is only allowed for genuinely unusable capacity** (and must stay that way):
- broken container / Docker pull failure / bad host → destroy + blacklist
  (`provision_instance` ~L986, `find_existing_instance(destroy_broken=True)` ~L1055).
- over-budget instance detected after creation (~L931).
- `prune --idle-hours N` for instances stopped too long.
- A merely **slow** host is stopped and **kept** (~L993), never destroyed.
- **Mere boot/setup silence is NOT a destroy reason** — a stalled boot is *stopped*, not
  destroyed (see Phase 4B). Only explicit error signals may destroy.

**Accept:** after any normal run (success, `--no-render`, or `--stop-after`), the instance
appears **stopped** in `vastai show instances` — never absent (absent = terminated).

**Verify:**
```powershell
npm run new -- --topic "..." --minutes 1 --audio-model indicf5 --voice indicf5_aarav --imageprovider local --stop-after audio
.venv\Scripts\vastai.exe show instances   # expect status "stopped", the id still listed
# do NOT see the instance disappear (that would mean it was terminated)
```

**Do not** add any `destroy_instance` call to an end-of-run / success path. Only failure
modes above may destroy.

---

## Phase 0 — Preflight (environment)

- [ ] **0.1** Node 18+ and `npm install` done.
- [ ] **0.2** `.venv` exists (`npm run setup`) and `.venv\Scripts\vastai.exe` is present.
- [ ] **0.3** `.env` has an LLM key (`DEEPSEEK_API_KEY` preferred) — `research`/writing need it.
- [ ] **0.4** Vast.ai account has API key + credit; `python scripts/_authcheck.py` (or
      `vastai show instances`) returns without error.
- [ ] **0.5** Baseline tests green: `npm test` (all four suites exit 0).

*Verify:* `npm test` → exit 0.

---

## Phase 1 — Persist research output (Gap #1) — ✅ **DONE**

> **Status: `[x]` implemented** in `new_video.py` right after `build_script(...)`:
> `script["research"]`, a flat `script["keywords"]`, and a best-effort `out/research.json`.
> Guarded by `if research_report:` so an offline / `--no-research` run is unaffected.

**What:** the research report currently lives only in the local variable `research_report`
and is thrown away after the writer prompt is built.

**Where:** `scripts/new_video.py`
- `main()` research block (L1402-1418), the `build_script(...)` call (**L1631**), and
  `SCRIPT_PATH.write_text(...)` (L1850).

**Change:**
1. After research succeeds, set `script["research"] = research_report`.
2. Also add a flat convenience field, e.g.
   `script["keywords"] = research_report["keywords"]` when present.
3. Write a sidecar `out/research.json` with the full report (best-effort; never fatal).
4. Guard for `research_report is None` (offline / `--no-research`).

**Accept:** `script.json` contains a `research` key with `keywords` + `ai_synthesis`
(seed, youtube_trending, google_search, questions_and_curiosity, long_tail_keywords,
concepts, seo_tags, recommended_hashtags). `out/research.json` exists after a run.

**Verify:**
```powershell
npm run new -- --topic "aadha din aadhi raat" --minutes 1 --dry-run   # sanity
npm run new -- --topic "aadha din aadhi raat" --minutes 1 --narration examples\aadha-din-aadhi-raat.txt --no-render
.venv\Scripts\python.exe -c "import json;s=json.load(open('script.json'));print(sorted(s.get('research',{}).keys()))"
```

---

## Phase 2 — Keywords feed visual prompts (Gap #2a) — ✅ **DONE**

> **Status: `[x]` implemented.** `extract_script_visual_keywords`, `build_search_queries`,
> `make_search_keywords`, `make_prompt`, `fetch_free_related_image` and both generators now
> take `script_keywords` (read from `script_data["keywords"]`), applied as PRIORITY 4 and
> only when the scene has no explicit `visualKeywords`.

**What:** `generate_images.py` only uses `scene.visualKeywords` + headline; research
keywords never reach the image/video prompt.

**Where:** `scripts/generate_images.py`
- `extract_script_visual_keywords(scene, topic, genre)` (~L511-608)
- its caller `generate_images_for_script(...)` (~L1610+)
- `scripts/generate_videos.py` reuses `make_prompt`.

**Change:**
1. Thread an optional `script_keywords: list[str] | None` (or `research: dict | None`) from
   the script dict into `extract_script_visual_keywords` / `make_prompt`.
2. Append a small number (e.g. top 5) of `long_tail_keywords` / `google_search` terms to
   `scene_keywords` **only when the scene has no explicit `visualKeywords`** (keep scene
   intent dominant; never let research terms override the shot description).
3. Keep it deterministic and capped (avoid prompt bloat).

**Accept:** image prompts for scenes without `visualKeywords` include research terms;
scenes with explicit `visualKeywords` are unchanged. `--visual-mode video` also benefits
(same factory).

**Verify:**
```powershell
python -c "import json;s=json.load(open('script.json'));print(s.get('keywords',{}).get('long_tail_keywords'))"
npm run new -- --topic "..." --minutes 1 --imageprovider local --no-render   # prompt preview logs
python scripts/test_providers.py
```

---

## Phase 3 — Keywords feed metadata / SEO (Gap #2b) — ✅ **DONE**

> **Status: `[x]` implemented.** `_research_seo_extras()` in `metadata.py` folds
> `seo_tags` + `recommended_hashtags` into both the procedural and AI YouTube tag lists,
> read straight from `script_data["research"]` (no new `new_video.py` call-site plumbing).

**What:** `metadata.build_seo_tags(topic, genre, language, headlines)` ignores research
`seo_tags` / `recommended_hashtags`.

**Where:** `scripts/metadata.py` — `build_seo_tags` (L520), `generate_metadata_procedural`
(L574), `generate_metadata_ai` (L834), `generate_metadata` (L951).
Call site in `new_video.py` is **L1942** (not ~L1920).

**Change:**
1. Accept optional `research: dict | None` in `build_seo_tags` and the two generators.
2. Merge `research["ai_synthesis"]["seo_tags"]` + `recommended_hashtags` into the tag list,
   de-duplicated and capped (e.g. ≤ 15 YouTube tags).
3. Pass `script.get("research")` from `new_video.py` at the metadata call site (L1942).

**Accept:** generated `out/metadata.json` YouTube `tags` include research-derived tags;
behaviour unchanged when research is absent.

**Verify:**
```powershell
npm run new -- --topic "..." --minutes 1 --narration examples\aadha-din-aadhi-raat.txt --no-render --generate-metadata
python -c "import json;m=json.load(open('out/metadata.json'));print(m['youtube']['tags'])"
```

---

## Phase 4 — `--stop-after audio` (Gap #3) — ✅ **DONE**

> **Status: `[x]` implemented.** `--stop-after` is now `("write", "audio", "metadata")`.
> `audio` returns right after narration (GPU stopped, no metadata/render); `metadata`
> returns after metadata, before the render. README flag table still to update (Phase 8).

**What:** there is no stop point after narration is stored; only `--stop-after write`,
`--no-render`, or `batch --phase audio`.

**Where:** `scripts/new_video.py`
- argparse `--stop-after` choices `("write",)` (~L1174-1177)
- early return for `== "write"` (~L1858)
- the post-`tts_succeeded` block (before metadata/render, ~L1900).

**Change:**
1. Extend choices to `("write", "audio", "metadata")`.
2. After `tts_succeeded` and (optionally) metadata, if `--stop-after audio` → call
   `stop_vast_instance()` and `return 0` **before** the render step.
3. Ensure the GPU stops (autostop) on this path, and that audio is already persisted to
   `out/gpu_audio_cache` + `public/audio/voice/` (it is — `tts.py` writes as it goes).
4. Update the help text and `README.md` flag table.

**Accept:** one command rents the 3090, writes script + narration + cache, stops the GPU,
and exits **without** rendering.

**Verify:**
```powershell
npm run new -- --topic "..." --minutes 1 --audio-model indicf5 --voice indicf5_aarav --imageprovider local --stop-after audio
# expect: script.json + public/audio/voice/*.wav + out/gpu_audio_cache/*, no out/*.mp4
```

---

## Phase 4B — Boot-stall watchdog: 5 min no-progress → STOP — ✅ **Option B chosen**

> **Decision: Option B (recorded 2026-10-13).** No new timeout knob was added. Policy is
> now documented in the `vast_manager.py` module header and **locked by a test**:
> `test_vast_boot_stall_policy()` in `scripts/test_providers.py` drives `wait_until_ready()`
> with a fake clock and asserts stalled-but-alive → `InstanceSlow` (stop-and-keep) while a
> provably empty install → `InstanceBroken` (destroy).

**Requirement (user):** if no execution happens during container setup/boot within
**5 minutes**, the instance must be **killed = STOPPED** (kept, per the invariant), not
left billing and not terminated.

> ⚠️ **Reality check (verified in `vast_manager.py`) — the first draft of this phase was
> wrong.** There is not one clock; there are **three**, and one *already fires at 5 min* —
> but it **destroys** rather than stops. The defect is the **action**, not the duration.

| Knob (env) | Default | When | Fires as |
|---|---|---|---|
| `VAST_CONTAINER_TIMEOUT` (L1272) | 600 s | image pull (extended while docker text changes) | `InstanceBroken` → **destroy** |
| `VAST_INSTALL_GRACE_SECONDS` (L1290) | **300 s** | after container is `running` | `InstanceBroken` → **destroy + blacklist** |
| `VAST_SERVER_STALL_TIMEOUT` (L1273) | 600 s | install in progress | `InstanceSlow` → stop; but `InstanceBroken`→**destroy** if `peak_disk < VAST_INSTALL_FLOOR_GB` (1.5 GB) |

So a **5-minute "no execution" window already exists** (`VAST_INSTALL_GRACE_SECONDS=300`);
adding `VAST_BOOT_STALL_TIMEOUT=300` would just duplicate it.

**Why two paths destroy (deliberate — do not silently undo):** the comments at L1389/L1409
say that if `peak_disk < 1.5 GB` then *nothing was installed*, so stopping would preserve an
**empty** install while still billing storage; destroying frees the disk. The user's rule
("no execution in 5 min → stop") therefore **conflicts** with that design — decide it.

**Where:** `scripts/vast_manager.py` — `wait_until_ready()` (L1251-1440) two-phase wait;
knobs at L1270-1290; raise sites L1310/L1375/L1409/L1392; `_safe_stop` (L435) /
`_safe_destroy` (L428); `_record_idle` via `stop_instance` (L1116-1121).

**Change (choose one, and record the choice):**
- **Option A — user's rule, literal:** in the "installed nothing" / "never reached running"
  paths use `_safe_stop()` + `mark_bad_host()` **instead of** `_safe_destroy()`. Instance is
  stopped and kept, resumable via `--vast-instance`. *Cost:* a stopped empty 50 GB box keeps
  billing ~$0.77/day until `prune --idle-hours N` reclaims it.
- **Option B — recommended (cost-safe):** keep destroying only when *truly nothing installed*
  (`peak_disk < VAST_INSTALL_FLOOR_GB`) or the image never pulled, and guarantee
  stop-and-keep for every "slow but alive" case (already true: `InstanceSlow` → `_safe_stop`).

**Also (independent of the choice):**
1. **Do not** add a new `VAST_BOOT_STALL_TIMEOUT`; reuse the existing `VAST_INSTALL_GRACE_SECONDS`
   (already 300 s) and document the three knobs together in the module header.
2. Keep `fatal_status` (`error`/`offline`/`"error response from daemon"`) → `InstanceBroken`
   → destroy. Silence alone must never destroy.
3. `.vast_idle.json` is **already written** by `stop_instance()` → `_record_idle()` (L1116-1121),
   so item "record stop time" needs **no new work** — `_safe_stop()` covers it.
4. If the API reports no disk figure (`progress_signal` False), never stop/destroy on it
   (hard cap only) — already handled.
5. If a CLI override is added, thread it through `provision_instance` (guarded by
   `test_vast_call_sites()`).

**Accept:** a slow/stuck boot is **stopped at ~5 min** (resumable); only an explicit error
signal — or, under Option B, a provably empty install — may destroy.

**Verify:** extend `test_vast_setup_resilience()` in `scripts/test_providers.py` to stub
`get_instance_info`/`_disk_used_gb` and assert, for the chosen option, whether
`stop_instance` or `destroy_instance` is called. Then `python scripts/test_providers.py` exits 0.

---

## Phase 5 — Theme → visual prompts (Gap #4) — ✅ **DONE**

**What:** `theme` only drives Remotion colors/fonts; it never reaches image prompts.

**Where:** `scripts/generate_images.py` `make_prompt` (~L655-706),
`scripts/beats.py` theme presets (`list_themes`, `get_theme`).

**Change:** add an optional theme style fragment (palette / mood words) to the prompt
prefix when `script["theme"]` is set and not `auto`/`random`.

**Accept:** setting `--theme cyber_neon` visibly changes the prompt text; unset theme keeps
current behaviour.

**Verify:** compare `--theme` on/off prompt logs for one scene.

---

## Phase 6 — Vast SSH path (Gap #5) — ✅ **HTTP-only (resolved)**

**What:** the account has no SSH keys, so `--render-on vast` (ssh+scp), reference-voice
upload, and `repair_dependencies()` cannot run. HTTP endpoints are the live path.

**Where:** `scripts/vast_manager.py` (`ssh_target`, `run_ssh`, `push_reference_voices`),
`scripts/vast_render.py`.

**Change (option A — enable SSH):**
```powershell
.venv\Scripts\vastai.exe create ssh-key      # create a key
.venv\Scripts\vastai.exe attach ssh <instance_id> <key_id>
```
Then verify `ssh`/`scp` connectivity before relying on `--render-on vast`.

**Change (option B — accept HTTP-only):** document that remote render/repair are
unavailable and keep local rendering. (Recommended until remote render is proven.)

**Accept:** either SSH works end-to-end, or the docs explicitly mark remote render as
"HTTP-only / not supported on this account".

---

## Phase 7 — Prove the flow live (Gap #6) — ⬜ **PENDING (verification only)**

**What:** rent 3090 → boot → IndicF5 → audio → local render has never been run end-to-end.

**Change:** none (verification only). Run a short, cheap job.

**Accept:**
- GPU picks RTX 3090 within budget (`DEFAULT_GPU_BUDGETS`).
- `/health` reports `tts_ready`; `synthesize_batch` returns audio.
- WAVs land in `public/audio/voice/` and `out/gpu_audio_cache/`.
- Instance is **stopped (not terminated)** after the run — still listed, never absent
  (see the GPU lifecycle invariant).
- `out/*.mp4` renders on CPU via Remotion.

**Verify:**
```powershell
npm run new -- --topic "aadha din aadhi raat" --minutes 1 `
  --audio-model indicf5 --voice indicf5_aarav --imageprovider local --out out/indicf5-1min.mp4
```
Then: `npm run verify-final` (or ffprobe/duration check) and confirm the instance is
**stopped, not gone**, with `.venv\Scripts\vastai.exe show instances`.

---

## Phase 8 — Docs & canonical command — ✅ **DONE**

**What:** the README should present the single command and the phase/stop semantics.

**Where:** `README.md` (flags table), `scripts/batch.py` docstring.

**Change:** add the canonical one-liner, document `--stop-after {write,audio,metadata}`,
and note the render engine is **Remotion (bundled ffmpeg) on CPU**, not a raw ffmpeg call.

**Accept:** a new user can copy one command and get a finished video.

---

## Test & regression checklist

- [x] `npm test` (phonemise + audio + story + providers) exits 0. — **verified**
- [x] `python scripts/test_providers.py` passes `test_vast_call_sites()`. — **verified**
- [x] No new `get_errors` / Pylance diagnostics in edited files. — **verified**
- [ ] Non-IndicF5 path still works: `npm run new -- --topic x --minutes 1 ...` (Kokoro). — **NOT verified**
- [x] Offline / `--no-research` still produces a script. — **verified** (`--no-research --stop-after write`)
- [ ] `--stop-after audio` / `--stop-after metadata` exercised at runtime. — **NOT verified**

---

## Ordering & justification

1. ~~Phase 1~~ ✅ · ~~Phases 2 & 3~~ ✅ · ~~Phase 4~~ ✅ · ~~Phase 4B (Option B)~~ ✅.
2. **Phase 7** — prove the flow live (needs a Vast account + an LLM key).
3. **Phase 8** — docs / canonical command.
4. **Phase 5** optional polish; **Phase 6** only if remote render is genuinely wanted.

## Open decisions required before implementing

1. **Phase 4B — stop vs destroy on a stalled boot.** ✅ **RESOLVED — Option B.** Slow-but-alive
   boots are stopped and kept; only a provably empty install (peak disk < 1.5 GB) or a host
   whose image never pulled is destroyed. See Phase 4B.
2. **Phase 1 — shape of persisted keywords.** ✅ **RESOLVED.** Implemented as the raw
   `research` dict **plus** a flat `script["keywords"]: list[str]` for Phase 2 to consume.
3. **Phase 6 — SSH.** ✅ **RESOLVED — HTTP-only.** Documented in the README that
   `--render-on vast`, dependency repair and reference-voice upload cannot run on this
   account until a Vast SSH key is created; the default local render is unaffected.

---

## Deferred / explicitly out of scope

- Writing a **raw ffmpeg** compositor to replace Remotion — large effort, loses captions,
  lip-sync rig and animation. Keep Remotion (bundles ffmpeg, runs on CPU).
- Making research **mandatory** — it must stay best-effort so an offline run still works.
