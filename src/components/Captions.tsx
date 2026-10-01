import {useVideoConfig} from 'remotion';
import React from 'react';
import {CaptionOptions, Theme, Word} from '../types';

/** Seconds of early reveal before a cue's first word is spoken. */
const LEAD_IN = 0.12;
/** How long a cue may linger after its last word before it is hidden. */
const LINGER = 0.5;

interface Cue {
  words: Word[];
  start: number;
  end: number;
}

/**
 * Groups words into caption cues. Word timings come straight from the TTS
 * engine, so cues stay in sync without any speech recognition step.
 */
export const buildCues = (words: Word[], wordsPerCue: number): Cue[] => {
  const size = Math.max(1, Math.floor(wordsPerCue));
  const groups: Word[][] = [];
  for (let i = 0; i < words.length; i += size) {
    groups.push(words.slice(i, i + size));
  }

  const starts = groups.map((g) => Math.max(0, g[0].start - LEAD_IN));

  return groups.map((g, i) => {
    const last = g[g.length - 1].end;
    // Never overlap the next cue, but stay visible through short pauses.
    const end = i + 1 < groups.length ? Math.min(starts[i + 1], last + LINGER) : last + LINGER;
    return {words: g, start: starts[i], end};
  });
};

export const Captions: React.FC<{
  words: Word[];
  /** Absolute scene time in seconds at the current frame. */
  time: number;
  options: CaptionOptions;
  theme: Theme;
}> = ({words, time, options, theme}) => {
  const {fps, height} = useVideoConfig();

  if (!options.enabled || words.length === 0) {
    return null;
  }

  const cue = buildCues(words, options.wordsPerCue).find((c) => time >= c.start && time < c.end);

  if (!cue) {
    return null;
  }

  // Cues can be as short as half a second, so the entrance has to scale with
  // the cue length: a fixed fade would leave short cues barely legible.
  const ageInFrames = Math.max(0, (time - cue.start) * fps);
  const cueFrames = Math.max(1, (cue.end - cue.start) * fps);
  const enterFrames = Math.max(2, Math.min(8, cueFrames * 0.3));
  const eased = 1 - Math.pow(1 - Math.min(1, ageInFrames / enterFrames), 3);

  return (
    <div
      style={{
        position: 'absolute',
        left: 0,
        right: 0,
        bottom: `${options.bottom * 100}%`,
        display: 'flex',
        justifyContent: 'center',
        padding: `0 ${Math.round(height * 0.08)}px`,
        opacity: eased,
        transform: `translateY(${(1 - eased) * 18}px)`,
      }}
    >
      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          justifyContent: 'center',
          gap: `0 ${Math.round(options.fontSize * 0.32)}px`,
          maxWidth: '100%',
          padding: `${Math.round(options.fontSize * 0.26)}px ${Math.round(options.fontSize * 0.7)}px`,
          borderRadius: Math.round(options.fontSize * 0.35),
          backgroundColor: theme.panel,
          backdropFilter: 'blur(8px)',
        }}
      >
        {cue.words.map((w, i) => {
          const isCurrent = options.highlight && time >= w.start && time < w.end;
          const isSpoken = options.highlight && time >= w.end;

          return (
            <span
              key={`${w.text}-${i}`}
              style={{
                fontFamily: theme.bodyFont,
                fontSize: options.fontSize,
                lineHeight: 1.25,
                fontWeight: isCurrent ? 800 : 700,
                letterSpacing: -0.2,
                color: isCurrent ? theme.accent : theme.text,
                textTransform: options.uppercase ? 'uppercase' : 'none',
                opacity: options.highlight && !isCurrent && !isSpoken ? 0.82 : 1,
                textShadow: '0 2px 12px rgba(0,0,0,0.65)',
                transform: isCurrent ? 'scale(1.04)' : 'scale(1)',
              }}
            >
              {w.text}
            </span>
          );
        })}
      </div>
    </div>
  );
};
