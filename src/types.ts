/**
 * Data model for the video.
 *
 * `script.json` (written by you or an AI agent) holds the creative content.
 * `scripts/tts.py` normalises it and adds narration timings, producing
 * `src/generated/project.json`, which is what the renderer actually reads.
 */

export type BackgroundSpec =
  | { type: 'color'; color: string }
  | { type: 'gradient'; from: string; to: string; angle: number }
  | { type: 'pattern'; pattern: 'grid' | 'dots' | 'diagonal'; bg: string; line: string }
  | {
    type: 'image';
    src: string;
    kenBurns: 'in' | 'out' | 'left' | 'right' | 'none';
    dim: number;
  }
  | { type: 'video'; src: string; dim: number };

export type MouthShape = 'A' | 'B' | 'C' | 'D' | 'E' | 'F' | 'G' | 'H' | 'X';

/** Emotional states a character can hold. Drives brows, lids, pupils and mouth. */
export type EmotionName =
  | 'neutral'
  | 'calm'
  | 'worried'
  | 'scared'
  | 'shock'
  | 'angry'
  | 'sad'
  | 'sly';

/** Where a character is looking or holding their head. */
export type Facing = 'left' | 'center' | 'right';

/** Named arm poses. Kept coarse so they read clearly at any scale. */
export type GestureName =
  | 'none'
  | 'explain'
  | 'point'
  | 'reach'
  | 'recoil'
  | 'clutch';

/**
 * How a character performs in a shot.
 *
 * Deliberately mirrors how an animation director briefs a pose: emotion, where
 * the head and eyes are turned, and what the hands are doing. It is all plain
 * data, so it can be written by a planner, an LLM, or by hand, and re-rendered
 * without touching code.
 */
export interface ActingSpec {
  emotion?: EmotionName;
  /** 0-1. Scales the expression and the amount of body movement. */
  intensity?: number;
  head?: Facing;
  eyes?: Facing;
  gesture?: GestureName;
}

/**
 * Horror backdrops. Can be procedural (drawn from code with zero assets) or
 * AI generated background images.
 */
export type VisualSpec =
  | { type: 'image'; src: string; kenBurns?: 'in' | 'out' | 'left' | 'right' | 'none'; dim?: number; prompt?: string }
  | { type: 'void'; creep?: number; tint?: string }
  | { type: 'fog'; density?: number; glow?: number; seed?: number }
  | { type: 'forest'; fog?: number; moonlight?: number }
  | { type: 'figure'; distance?: number; fog?: number; glow?: number }
  | { type: 'house'; fog?: number; lit?: boolean; moonlight?: number }
  | { type: 'corridor'; flicker?: number; depth?: number }
  | { type: 'candle'; glow?: number }
  | { type: 'rain'; intensity?: number; lightning?: boolean }
  | { type: 'eyes'; pairs?: number; spread?: number }
  | { type: 'grave'; fog?: number; moonlight?: number };

/** Per-shot grading and camera treatment. All values are 0..1 unless noted. */
export interface ShotEffects {
  vignette?: number;
  grain?: number;
  dust?: number;
  shake?: number;
  flicker?: number;
  exposureFlicker?: number;
  /** Breathing zoom amount, e.g. 0.03 for a subtle push. */
  zoom?: number;
  drift?: 'in' | 'out' | 'left' | 'right' | 'none';
}

/**
 * One visual beat. Shots carry a relative `weight`, not an absolute duration:
 * they are normalised across the scene, so narration length stays authoritative
 * and shots can never drift out of sync with the audio.
 */
export interface ShotSpec {
  visual: VisualSpec;
  weight?: number;
  effects?: ShotEffects;
  transition?: { type: 'cut' | 'fade' | 'crossfade'; seconds?: number };
}

/** A one-shot sound effect placed at a point in the scene. */
export interface SoundCue {
  /** Seconds from the start of the scene. */
  at: number;
  /** Asset basename produced by `make_assets.py`, e.g. `sfx_stinger`. */
  sfx: string;
  volume?: number;
}

export interface MouthCue {
  start: number;
  end: number;
  value: MouthShape;
}

/** A reusable character. Defined once, reused across every episode. */
export interface CharacterSpec {
  name: string;
  /** TTS voice id, e.g. `af_heart` or `indic_parler_aarav`. */
  voice: string;
  speed?: number;
  palette: {
    skin: string;
    skinShade: string;
    hair: string;
    shirt: string;
    accent: string;
    outline: string;
  };
  hairstyle: 'short' | 'bob' | 'long' | 'bun' | 'bald';
  accessory: 'none' | 'glasses';
}

/** Where a character stands in the frame. */
export interface CastMember {
  id: string;
  /** Horizontal centre, 0 (left) to 1 (right). */
  x: number;
  scale: number;
  flip?: boolean;
  /** Falls back to the scene's default `acting` when omitted. */
  acting?: ActingSpec;
}

/** One spoken line. `character` null means narration with the default voice. */
export interface ScriptLine {
  character: string | null;
  text: string;
}

export interface Scene {
  id: string;
  narration: string;
  lines: ScriptLine[];
  headline: string;
  subtext: string;
  bullets: string[];
  headlinePosition: 'top' | 'center' | 'bottom';
  background: BackgroundSpec;
  cast: CastMember[];
  /** Procedural visual beats. When present these replace `background`. */
  shots: ShotSpec[];
  /** Default performance for every cast member in this scene. */
  acting?: ActingSpec;
  /** Looping ambience bed, e.g. `amb_drone`. Mixed under the narration. */
  ambience: { src: string; volume: number } | null;
  /** One-shot stingers and impacts, timed from the scene start. */
  soundCues: [];
  /** Seconds of silence before narration starts. */
  padBefore: number;
  /** Seconds of silence after narration ends. */
  padAfter: number;
  /** Narration word count, used for pacing checks on long-form scripts. */
  words: number;
}

export interface Theme {
  key?: string;
  name?: string;
  accent: string;
  text: string;
  muted: string;
  panel: string;
  headlineFont: string;
  bodyFont: string;
  bg_color?: string;
}

export interface CaptionOptions {
  enabled: boolean;
  /** How many words appear at once. 2-3 reads as a statement, 4-5 as subtitles. */
  wordsPerCue: number;
  /** Distance from the bottom of the frame, as a fraction of frame height. */
  bottom: number;
  fontSize: number;
  uppercase: boolean;
  highlight: boolean;
  /** Where the text sits. `center` turns subtitles into on-screen statements. */
  position?: 'bottom' | 'center';
  /** Dark outline width in pixels, for legibility over busy backgrounds. */
  outline?: number;
  /** Draw a translucent backdrop panel behind the text. */
  panel?: boolean;
  /** Widest the text block may get, as a fraction of frame width. */
  maxWidth?: number;
  /** How far words rise as they appear, in pixels. */
  rise?: number;
}

export interface ScriptFile {
  title: string;
  topic?: string;
  language?: string;
  genre?: string;
  format?: string;
  aspectRatio?: string;
  /** Default Kokoro voice id for narration. */
  voice: string;
  /** Narration engine selected for this project. */
  audioModel?: 'kokoro' | 'indic_parler' | 'indicf5';
  /** Exactly one visual generation mode is used for generated assets. */
  visualMode?: 'image' | 'video';
  videoModel?: string;
  speed: number;
  fps: number;
  width: number;
  height: number;
  tailPaddingSeconds: number;
  /** Silence inserted between sentences, in seconds. */
  sentencePauseSeconds: number;
  /** Global continuous background ambience spanning the full video duration without interruption. */
  ambience?: null;
  captions: CaptionOptions;
  music: { src: string; volume: number; fadeOutSeconds: number } | null;
  theme: Theme;
  characters: Record<string, CharacterSpec>;
  scenes: Scene[];
}

export interface Word {
  text: string;
  /** Seconds, relative to the start of the scene's audio. */
  start: number;
  end: number;
}

/** One rendered line of audio, optionally spoken by a character. */
export interface LineTiming {
  characterId: string | null;
  /** Path relative to `public/`, or null when there is no audio. */
  audio: string | null;
  /** Seconds from the start of the scene. */
  start: number;
  duration: number;
  words: Word[];
  /** Rhubarb mouth shapes, timed from the start of this line's audio. */
  mouthCues: MouthCue[];
}

export interface SceneTiming {
  id: string;
  index: number;
  /** Absolute start of the scene on the timeline, in seconds. */
  start: number;
  /** Total scene length including padding, in seconds. */
  duration: number;
  /** Seconds from scene start to the first line's audio. */
  audioStart: number;
  /** First line's audio, kept for the single-narrator case. */
  audio: string | null;
  /** All words in the scene, relative to `audioStart`. Used by captions. */
  words: Word[];
  lines: LineTiming[];
}

export interface Timings {
  totalDuration: number;
  totalWords: number;
  scenes: SceneTiming[];
}

/**
 * Kept as a type alias rather than an interface: TypeScript only gives object
 * *aliases* the implicit index signature that Remotion's `Composition` needs
 * for its `Record<string, unknown>` prop constraint.
 */
export type Project = {
  script: ScriptFile;
  timings: Timings;
};
