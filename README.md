# tube2dscript

Turn a written script into a long-form 2D video — **with no AI video-generation bill**.

The trick: nothing here generates pixels with a paid model. An AI agent (or you) writes
a `script.json`, free text-to-speech narrates it, and [Remotion](https://remotion.dev)
renders the frames deterministically on your own machine. The only cost is your CPU time.

```
script.json ──► edge-tts ──► narration MP3 + per-word timestamps
                    │
                    └──► Remotion ──► out/video.mp4
```

## Why this is free

| Task | This project | Typical paid stack |
|---|---|---|
| Narration | `edge-tts` neural voices, no API key | ElevenLabs $$ |
| Caption timing | `edge-tts` reports **word boundaries** | Whisper (free) or paid ASR |
| Visuals | React/CSS rendered locally | Sora / Runway / Veo ($$$/sec) |
| Encoding | Remotion's bundled FFmpeg | — |

Long-form is the hard case for AI video generators because cost scales with seconds.
Here, a 30-minute video and a 30-second video cost the same: zero.

## Requirements

- **Node.js 18+** (tested on 24)
- **Python 3.9+** with `edge-tts` — `py -m pip install edge-tts`
- No FFmpeg install needed. Remotion ships its own.

## Quick start

```powershell
npm install
npm run voice     # narrate script.json -> public/audio/*.mp3 + src/generated/project.json
npm run render    # -> out/video.mp4
```

Or in one shot: `npm run make`

Other commands:

```powershell
npm run studio    # visual editor with live preview and scrubbing
npm run preview   # narrate, then render only the first 10 seconds
py scripts/tts.py --voices   # list the free voices
```

`npm run voice` caches per scene: an unchanged scene is never re-synthesised, so
tweaking one paragraph does not re-download the whole narration.

## Writing the script

Two ways in. Pick whichever suits you — an AI agent can produce either.

### 1. Plain JSON (`script.json`)

```json
{
  "title": "The Deep Sea",
  "voice": "en-US-GuyNeural",
  "rate": "+6%",
  "fps": 30,
  "captions": {"enabled": true, "wordsPerCue": 4, "highlight": true},
  "theme": {"accent": "#38bdf8", "text": "#ffffff"},
  "scenes": [
    {
      "id": "intro",
      "headline": "The Deep Sea",
      "subtext": "An introduction",
      "bullets": ["Roughly 71% of the surface"],
      "narration": "Two thirds of our planet is ocean, and almost all of it is dark.",
      "background": {"type": "gradient", "from": "#04101c", "to": "#0f3352", "angle": 160}
    }
  ]
}
```

Every field except `narration` and `background` is optional and falls back to a default.
The full default set lives in `DEFAULT_SCRIPT` / `DEFAULT_SCENE` at the top of
[scripts/tts.py](scripts/tts.py).

### 2. Plain text

Write a readable script and let [scripts/from_txt.py](scripts/from_txt.py) convert it:

```
@title The Deep Sea
@voice en-US-GuyNeural
@rate +6%

# The Deep Sea
! gradient: #04101c -> #0f3352
> An introduction
* Roughly 71% of the surface
Two thirds of our planet is ocean, and almost all of it is dark.
```

```powershell
py scripts/from_txt.py myscript.txt -o script.json
npm run voice
```

## Backgrounds

| Type | Example |
|---|---|
| `gradient` | `{"type":"gradient","from":"#04101c","to":"#0f3352","angle":160}` |
| `color` | `{"type":"color","color":"#0b1220"}` |
| `pattern` | `{"type":"pattern","pattern":"grid","bg":"#061321","line":"#153a5c"}` |
| `image` | `{"type":"image","src":"images/sea.jpg","kenBurns":"in","dim":0.25}` |
| `video` | `{"type":"video","src":"video/waves.mp4","dim":0.2}` |

`pattern` (`grid`, `dots` or `diagonal`) needs no assets at all, which makes it the
fastest way to get a clean look. `image` adds a slow Ken Burns drift
(`in`, `out`, `left`, `right`, `none`).

Put files in `public/images/`, `public/video/` and `public/music/`, then reference them
by their path **relative to `public/`**. Free sources: Pexels, Pixabay, Unsplash.

Background music is optional and mixes itself under the narration:

```json
"music": {"src": "music/ambient.mp3", "volume": 0.07, "fadeOutSeconds": 3}
```

## Themes and captions

```json
"theme": {
  "accent": "#38bdf8",
  "text": "#ffffff",
  "muted": "#cbd5e1",
  "panel": "rgba(3, 7, 18, 0.55)"
},
"captions": {
  "enabled": true,
  "wordsPerCue": 4,
  "bottom": 0.07,
  "fontSize": 54,
  "uppercase": false,
  "highlight": true
}
```

`highlight` turns on karaoke-style word highlighting, driven by the real word
timestamps from the TTS engine. `wordsPerCue` controls how many words appear at once —
4 is a comfortable default for 1920×1080.

Fonts use CSS font stacks (`theme.headlineFont` / `theme.bodyFont`), so nothing needs
downloading. Defaults are Georgia for headlines and Segoe UI for captions.

## Rendering long-form video

Rendering time is roughly proportional to frame count. A 30-minute video at 30 fps is
54,000 frames, so expect a real encode rather than a few seconds.

- `remotion.config.ts` sets `Config.setConcurrency(4)`. Raise it on a fast machine with
  many cores; lower it if the render makes your PC unusable.
- Render to H.264 with `Config.setCrf(18)` for good quality at a sane file size.
- Frames render in parallel, so a long video is CPU-bound, not memory-bound.

Useful flags:

```powershell
npx remotion render Main out/video.mp4 --concurrency=8
npx remotion render Main out/preview.mp4 --frames=0-300     # 10 seconds
npx remotion still Main out/thumb.png --frame=120            # a thumbnail
```

## Project layout

```
script.json              your script (edit this)
scripts/tts.py           narration + word timings
scripts/from_txt.py      plain-text -> script.json
src/Root.tsx             composition, duration derived from the audio
src/Main.tsx             scene sequencing, audio, progress bar
src/components/          Background, SceneView, Captions
public/audio/            generated narration (gitignored)
public/images|video|music/   your media
src/generated/project.json   generated; read by the renderer
out/                     rendered videos
```

`npm run voice` must run before `npm run studio` or `npm run render`, because
`src/generated/project.json` is imported at bundle time and is what tells Remotion how
long the video is.

## Tips for faceless long-form

- **Pace**: `npm run voice` prints words-per-minute and warns above 185 wpm.
  Somewhere around 140–165 wpm keeps long videos watchable.
- **One idea per scene.** Each scene gets its own headline and its own audio file, so
  a broken sentence only costs you one cached re-synthesis.
- **Keep scenes 8–20 seconds.** Shorter scenes give the visuals more variety and make
  it easy to re-render just part of the audio.
- **Use `padAfter`** (default 0.55s) to leave breathing room between scenes.
