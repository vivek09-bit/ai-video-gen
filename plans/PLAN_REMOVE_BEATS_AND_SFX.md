# Implementation Plan: Remove Beats and SFX (Voice Audio & Images Only in Final Render)

## 1. Overview & Objective
Eliminate all **beats** (musical beat-matching, background music/beats) and **SFX** (sound effects) from the project codebase. The rendering pipeline will exclusively combine:
- **Visuals**: Scene images, background art, character visual sequences, and framing.
- **Audio**: Spoken voiceover/narration audio only (no background tracks, no sound effect cues).

---

## 2. File & Component Modifications

| Area | Target Files | Implementation Action |
| :--- | :--- | :--- |
| **Remotion Composition** | `src/Main.tsx`<br>`src/Root.tsx` | Delete all `<Audio>` elements targeting background beats, ambient music, and SFX tracks. Keep only the narration/speech `<Audio>` layer. |
| **Scene Components** | `src/components/SceneView.tsx`<br>`src/components/horror/Effects.tsx`<br>`src/components/horror/Stage.tsx` | Remove SFX playback triggers and beat-synced shake/flash events. Transition all visual animations to standard duration-based timelines. |
| **Data Types & Schemas** | `src/types.ts`<br>`src/generated/project.json` | Remove or make optional `beats`, `sfx`, `soundEffects`, and `bgm` fields. Ensure the renderer accepts scenes without beat/sfx metadata. |
| **Beat & Pipeline Scripts** | `scripts/beats.py`<br>`scripts/story.py`<br>`scripts/new_video.py` | Disable beat detection, onset analysis, and SFX cue allocation. Calculate scene durations strictly from voiceover audio lengths. |
| **Render Orchestration** | `scripts/generate_videos.py`<br>`scripts/batch.py` | Strip out SFX fetching/synthesis and beat processing steps. Execute Remotion render with only speech audio and image sequences. |

---

## 3. Direct Implementation Steps

### Step 1: Remotion Composition Cleanup (`src/`)
1. **Strip SFX & Beat Audio in `src/Main.tsx`**:
   - Locate and delete all `<Audio>` tags corresponding to background music (`bgm`), beats, and sound effects (`sfx`).
   - Retain only the primary dialogue/voice narration `<Audio>` element mapped to each scene.
2. **Decouple Visual Effects from Beats (`src/components/`)**:
   - In `src/components/horror/Effects.tsx` and `Stage.tsx`, remove hooks and event listeners triggered by `beat` or `sfx` timestamps.
   - Refactor visual effects and shot transitions to use linear scene time or speech-based keyframes rather than musical beats.
3. **Ensure Null Safety**:
   - Ensure scene rendering operates unconditionally when `scene.beats` or `scene.sfx` are empty or omitted.

### Step 2: Types & Schema Updates (`src/types.ts`)
1. **Update Scene and Project Types**:
   - Remove or mark as optional `beats?: Beat[]` and `sfx?: SfxItem[]` in `ProjectConfig` and `Scene` interfaces.
   - Standardize `Scene` schema to strictly:
     ```typescript
     interface Scene {
       id: string;
       image: string;       // Visual asset
       audio: string;       // Voice narration
       durationInFrames: number;
     }
     ```

### Step 3: Python Pipeline Decoupling (`scripts/`)
1. **Neutralize `scripts/beats.py`**:
   - Bypass beat-slicing and SFX injection logic to return a direct 1:1 mapping of voice audio duration.
   - Remove unused beat detection dependencies (e.g., librosa, aubio, onset detection).
2. **Update Story & Asset Generation (`scripts/story.py`, `scripts/new_video.py`)**:
   - Remove prompt instructions that ask LLMs to generate SFX cues (e.g. `[SFX: thunder]`, `[BEAT: drop]`).
   - Remove sound effect downloading, caching, or synthesis routines.
   - Output `project.json` containing only:
     - Spoken voice audio path per scene.
     - Scene image asset path per scene.
     - Scene duration calculated directly from speech audio file duration.
3. **Update Batch & Render Scripts (`scripts/generate_videos.py`, `scripts/batch.py`)**:
   - Remove calls to beat analysis and SFX synthesis in the generation pipeline.
   - Configure Remotion render arguments to process purely voice audio and image layers.

---

## 4. Implementation Checklist
- [x] `src/Main.tsx`: Remove all SFX & Beat audio elements; keep only speech narration.
- [x] `src/components/`: Decouple visual effects and transitions from beats/SFX cues.
- [x] `src/types.ts`: Remove or mark optional all `beats` and `sfx` fields.
- [x] `scripts/beats.py`: Bypass beat-slicing and SFX injection routines.
- [x] `scripts/story.py`: Remove SFX tags, cues, and prompt references.
- [x] `scripts/generate_videos.py`: Configure pipeline to render only speech audio and images.
