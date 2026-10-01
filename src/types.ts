/**
 * Data model for the video.
 *
 * `script.json` (written by you or an AI agent) holds the creative content.
 * `scripts/tts.py` normalises it and adds narration timings, producing
 * `src/generated/project.json`, which is what the renderer actually reads.
 */

export type BackgroundSpec =
  | {type: 'color'; color: string}
  | {type: 'gradient'; from: string; to: string; angle: number}
  | {type: 'pattern'; pattern: 'grid' | 'dots' | 'diagonal'; bg: string; line: string}
  | {
      type: 'image';
      src: string;
      kenBurns: 'in' | 'out' | 'left' | 'right' | 'none';
      dim: number;
    }
  | {type: 'video'; src: string; dim: number};

export interface Scene {
  id: string;
  narration: string;
  headline: string;
  subtext: string;
  bullets: string[];
  headlinePosition: 'top' | 'center' | 'bottom';
  background: BackgroundSpec;
  /** Seconds of silence before narration starts. */
  padBefore: number;
  /** Seconds of silence after narration ends. */
  padAfter: number;
  /** Narration word count, used for pacing checks on long-form scripts. */
  words: number;
}

export interface Theme {
  accent: string;
  text: string;
  muted: string;
  panel: string;
  headlineFont: string;
  bodyFont: string;
}

export interface CaptionOptions {
  enabled: boolean;
  wordsPerCue: number;
  /** Distance from the bottom of the frame, as a fraction of frame height. */
  bottom: number;
  fontSize: number;
  uppercase: boolean;
  highlight: boolean;
}

export interface ScriptFile {
  title: string;
  voice: string;
  rate: string;
  pitch: string;
  volume: string;
  fps: number;
  width: number;
  height: number;
  tailPaddingSeconds: number;
  captions: CaptionOptions;
  music: {src: string; volume: number; fadeOutSeconds: number} | null;
  theme: Theme;
  scenes: Scene[];
}

export interface Word {
  text: string;
  /** Seconds, relative to the start of the scene's narration audio. */
  start: number;
  end: number;
}

export interface SceneTiming {
  id: string;
  index: number;
  /** Absolute start of the scene on the timeline, in seconds. */
  start: number;
  /** Total scene length including padding, in seconds. */
  duration: number;
  /** Absolute time the narration audio begins (scene start + padBefore). */
  audioStart: number;
  /** Path relative to the `public/` folder, or null when there is no narration. */
  audio: string | null;
  words: Word[];
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
