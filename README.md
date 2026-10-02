# tube2dscript

Turn a written script into a long-form 2D animated video with **characters, motion
and lip sync** — at **zero cost per video**.

Nothing here calls a paid API. Narration is synthesised locally by Kokoro, lip sync
is computed locally by Rhubarb, and the frames are drawn by code with Remotion on
your own machine. The only currency is CPU time.

```
script.json ──► Kokoro (local TTS) ──► narration WAV + word timings
                        │
                        └──► Rhubarb (local) ──► mouth shapes A–H/X
                                  │
                    Remotion (local) ──► out/video.mp4
```

## Why this is free — and why it is not "AI video generation"

AI video generators cost money because you pay per second of *generated pixels*,
and they cannot hold a character consistent: every frame is re-imagined, so faces
drift, poses wander and lip sync is impossible.

A **rig** solves this differently. Draw the character once as vector shapes, then
pose it. Consistency is guaranteed because it is literally the same asset, motion
is keyframe data, and lip sync becomes mouth-shape swapping. The result is
deterministic, re-renderable and free.

| Task | This project | Typical paid stack |
|---|---|---|
| Narration | Kokoro-82M, local CPU | ElevenLabs $$$ |
| Word timing | Kokoro sentence durations + misaki phonemes | Whisper or paid ASR |
| Lip sync | Rhubarb Lip Sync, local CPU | paywalled |
| Characters | Code-drawn vector rig | AI art per frame |
| Encoding | Remotion's bundled FFmpeg | — |

## Licensing (this matters if you monetise)

Everything in the default pipeline is commercially usable:

| Component | License |
|---|---|
| Kokoro-82M weights and code | **Apache-2.0** |
| misaki (grapheme→phoneme) | **Apache-2.0** |
| Rhubarb Lip Sync | **MIT** (explicitly permits commercial works) |
| Your character rig | yours |
| Remotion | free for individuals and companies ≤3 people |

**If you are migrating from `edge-tts`, read this.** `edge-tts` calls an undocumented
Microsoft consumer endpoint using a hardcoded client token, a spoofed Edge user agent
and a fake browser origin. It is not a licensed API, Microsoft's terms prohibit
automated access and reverse engineering, and Microsoft has already broken it twice
(403 errors after a client-auth change, and later quota blocks). Its own maintainer
says it *"shouldn't be used for commercial reasons"*. Kokoro replaces it with
Apache-2.0 weights that run on your machine.

Avoid these for monetised work: **XTTS-v2** (CPML, non-commercial), **F5-TTS**
(weights are CC-BY-NC), **Higgs Audio v3** (non-commercial), **Wav2Lip**
(non-commercial, and photoreal-only anyway).

## Requirements

- **Node.js 18+** (tested on 24)
- **Python** — the setup script provisions its own 3.12 via `uv`, no admin needed
- No FFmpeg install: Remotion bundles its own
- No GPU: everything runs on CPU

The setup script creates a dedicated Python 3.12 environment because Kokoro's
dependency chain has no wheels for Python 3.14 (its `misaki[en]` extra imports
spaCy, and `kokoro` pins `numpy==1.26.4`, neither of which builds without a C
toolchain). `uv` downloads its own interpreter with no administrator rights.

## Quick start

```powershell
npm install
npm run setup     # Python 3.12 env + Kokoro + Rhubarb (~800 MB, one time)
npm run voice     # synthesise narration and lip sync
npm run render    # -> out/video.mp4
npm run studio    # visual editor with live scrubbing
```

`npm run make` chains voice and render. `npm run preview` renders the first 10 seconds.

`npm run voice` caches per line: re-running it only re-synthesises what changed, so
editing one sentence does not re-render the whole narration.

## Performance

Measured on the reference machine — **Intel Core 5 120U, 15 W, CPU only, no GPU**:

```
synthesise       7.12 s wall for 12.72 s of audio
realtime factor  0.56x  (1.8x faster than realtime)
a 30-min video   ~17 min of synthesis
```

Rendering is separate and also CPU-bound. Rhubarb runs multithreaded on as many
cores as you give it.

## Writing the script

Two ways in — an AI agent can produce either.

### 1. Plain text (recommended, ~35% fewer tokens)

```
@title The Deep Sea
@voice bm_george
@speed 1.0

@char maya af_heart bun
@char sam  am_michael short glasses

# The Deep Sea
! gradient: #04101c -> #0f3352
! cast maya 0.5 1.0
> Two researchers, one very large ocean
Two thirds of our planet is ocean, and almost all of it is completely dark.

# Four Thousand Meters Down
! pattern: grid #061321 #153a5c
! cast maya 0.29 0.92 flip
! cast sam 0.71 0.92
sam: At four thousand meters, the pressure would crush a submarine.
maya: And yet something down there is making its own light.
```

```powershell
npm run voice   # or: .venv\Scripts\python.exe scripts\from_txt.py myscript.txt -o script.json
```

Directives: `@title` `@voice` `@speed` `@fps` `@char` · `#` scene · `!` background
or `! cast` · `>` subtext · `*` bullet · `id:` dialogue · `//` comment.

### 2. JSON (`script.json`)

```json
{
  "title": "The Deep Sea",
  "voice": "bm_george",
  "speed": 1.0,
  "characters": {
    "maya": {
      "name": "Maya",
      "voice": "af_heart",
      "hairstyle": "bun",
      "accessory": "none",
      "palette": {
        "skin": "#f0c49a", "skinShade": "#dba87c", "hair": "#2f2a3a",
        "shirt": "#e0653f", "accent": "#ffd166", "outline": "#232a33"
      }
    }
  },
  "scenes": [
    {
      "id": "intro",
      "headline": "The Deep Sea",
      "subtext": "An introduction",
      "cast": [{"id": "maya", "x": 0.5, "scale": 1}],
      "lines": [{"character": "maya", "text": "Two thirds of our planet is ocean."}],
      "background": {"type": "gradient", "from": "#04101c", "to": "#0f3352", "angle": 160}
    }
  ]
}
```

Every field except the text is optional and falls back to a default. The full
default set is `DEFAULT_SCRIPT` and `DEFAULT_SCENE` at the top of
[scripts/tts.py](scripts/tts.py).

## Characters

A character is defined once and reused every episode, which is what makes it
consistent. Fields:

| Field | Values |
|---|---|
| `voice` | any Kokoro voice id (see below) — gives each character a distinct voice |
| `speed` | optional per-character speaking rate |
| `hairstyle` | `short` `bob` `long` `bun` `bald` |
| `accessory` | `none` `glasses` |
| `palette` | `skin` `skinShade` `hair` `shirt` `accent` `outline` |

Placing a character uses `cast`: `x` is the horizontal centre (0 = left, 1 = right),
`scale` sizes them, and `flip` mirrors them so two characters can face each other.

Motion is generated, not hand-keyed: breathing, body sway, head bob, blinking, pupil
drift, and arm gestures that scale with how open the mouth is. Idle characters keep
breathing and blinking. All of it is a pure function of the frame number, so renders
are deterministic and the studio preview matches the final file exactly.

Only the character who owns the currently playing line moves their mouth; everyone
else rests. Dialogue works because each line is synthesised separately — with that
character's voice — and placed at its own point on the timeline.

## Lip sync

[Rhubarb Lip Sync](https://github.com/DanielSWolf/rhubarb-lip-sync) analyses the
generated audio and emits mouth shapes from the **Hanna-Barbera set** — the de-facto
2D standard from Scooby-Doo/The Flintstones, later used by Disney and Warner Bros:

| Shape | Mouth | Sounds |
|---|---|---|
| A | closed | P, B, M |
| B | slightly open, teeth together | EE, K, S, T |
| C | moderately open | EH, AE |
| D | wide open | AA |
| E | rounded | AO, ER |
| F | puckered | UW, OW, W |
| G | teeth on lower lip | F, V |
| H | tongue visible | L |
| X | idle / rest | silence |

Rhubarb runs on CPU only and needs no GPU. It is given the line's text via
`--dialogFile`, which improves accuracy.

If you pass `--no-lipsync`, characters still animate: mouths cycle through shapes off
the clock so a scene is never static.

## Word timing and captions

Kokoro returns audio per *segment*, not per word. To get caption timings, the
narration is split into sentences, each sentence is synthesised separately (so its
duration is exact from the sample count), and that duration is then distributed
across the sentence's words in proportion to their **phoneme counts** from misaki.

So timing is exact at sentence level and approximate word-to-word — well within what
burned-in captions need. Captions are karaoke-style, driven by those timings.

```json
"captions": {
  "enabled": true,
  "wordsPerCue": 4,
  "bottom": 0.06,
  "fontSize": 54,
  "highlight": true
}
```

## Backgrounds

| Type | Example |
|---|---|
| `gradient` | `{"type":"gradient","from":"#04101c","to":"#0f3352","angle":160}` |
| `color` | `{"type":"color","color":"#0b1220"}` |
| `pattern` | `{"type":"pattern","pattern":"grid","bg":"#061321","line":"#153a5c"}` |
| `image` | `{"type":"image","src":"images/sea.jpg","kenBurns":"in","dim":0.25}` |
| `video` | `{"type":"video","src":"video/waves.mp4","dim":0.2}` |

`pattern` (`grid`, `dots`, `diagonal`), `gradient` and `color` need no assets at all,
so you can build a whole video without sourcing a single image. `image` adds a slow
Ken Burns drift (`in`, `out`, `left`, `right`, `none`).

Put files in `public/images/`, `public/video/` and `public/music/` and reference them
by their path **relative to `public/`**. Free sources: Pexels, Pixabay, Unsplash.

Optional background music, mixed under the narration with fades:

```json
"music": {"src": "music/ambient.mp3", "volume": 0.07, "fadeOutSeconds": 3}
```

## Kokoro voices

`npm run voices` lists them. English voices:

| Id | Character | Id | Character |
|---|---|---|---|
| `af_heart` | American female, warm (top rated) | `am_michael` | American male, calm narration |
| `af_bella` | American female, bright | `am_onyx` | American male, deep |
| `af_nicole` | American female, soft | `am_adam` | American male |
| `af_sarah` | American female, clear | `bm_george` | British male, documentary |
| `bf_emma` | British female | `bm_lewis` | British male |

Also available: `af_alloy` `af_aoede` `af_jessica` `af_kore` `af_nova` `af_river`
`af_sky` `am_echo` `am_eric` `am_fenrir` `am_liam` `am_puck` `am_santa` `bf_alice`
`bf_isabella` `bf_lily` `bm_daniel` `bm_fable`.

Set `"speed"` (0.5–2.0) to pace narration; `1.0` is natural, `0.92` reads slower.

## Rendering long-form video

Render time scales with frame count: a 30-minute video at 30 fps is 54,000 frames.

- `remotion.config.ts` sets `Config.setConcurrency(4)` — raise it on a fast machine,
  lower it if rendering makes the machine unusable.
- Frames render in parallel, so a long video is CPU-bound, not memory-bound.

```powershell
npx remotion render Main out/video.mp4 --concurrency=8
npx remotion render Main out/preview.mp4 --frames=0-300   # 10 seconds
npx remotion still Main out/thumb.png --frame=120          # a thumbnail
```

## Project layout

```
setup.ps1                    one-time Python 3.12 + Kokoro + Rhubarb setup
script.json                  your script (edit this)
scripts/tts.py               Kokoro narration, word timings, Rhubarb lip sync
scripts/from_txt.py          plain text -> script.json
scripts/verify_kokoro.py     check Kokoro works here and measure its speed
src/Root.tsx                 composition; duration derived from the audio
src/Main.tsx                 scene sequencing, per-line audio, progress bar
src/components/SceneView.tsx scene composition: background, cast, text, captions
src/components/character/    the rig: Character.tsx and the A-H/X Mouth.tsx
src/components/Background.tsx / Captions.tsx
public/audio/                generated narration (gitignored)
public/images|video|music/   your media
src/generated/project.json   generated; read by the renderer
out/                         rendered videos
```

`npm run voice` must run before `studio` or `render`: `src/generated/project.json` is
imported at bundle time and tells Remotion how long the video is.

## Tips for long-form storytelling

- **Never let an agent read `src/generated/project.json`.** It stores a
  `{text, start, end}` object per word — roughly 47 tokens per narration word, so a
  30-minute script would cost ~214k tokens to read. It is generated output.
- **Write plain text, not JSON** — about 35% cheaper in tokens (2.00 vs 3.07 tokens
  per narration word).
- **Use longer scenes.** Scene overhead is pure tax; 35–60 word scenes are a good
  balance.
- **Pace**: `npm run voice` prints words-per-minute and warns above 185 wpm.
  140–165 wpm keeps long videos watchable.
- **One idea per scene.** Each scene gets its own audio files, so a mistake only
  costs one cached re-synthesis.
- **Repairing audio**: `npm run voice -- --force` re-synthesises everything.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Kokoro produced no audio` | Voice id wrong. Run `npm run voices`. |
| No mouth movement | Rhubarb missing — re-run `npm run setup`. |
| Captions drift from the voice | Stale cache; `npm run voice -- --force`. |
| `No module named 'kokoro'` | Wrong interpreter. Use `npm run voice`, or `.venv\Scripts\python.exe`. |
| Render is very slow | Lower `Config.setConcurrency` or render at 1080p instead of 4K. |
| `spacy` or `numpy` build errors | Something is using the system Python 3.14; use the `.venv` one. |
