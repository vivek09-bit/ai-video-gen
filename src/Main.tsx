import {
  AbsoluteFill,
  Audio,
  Sequence,
  interpolate,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';
import React from 'react';
import { SceneView } from './components/SceneView';
import { Project } from './types';

const resolveAudioPath = (
  src: string | undefined | null,
  defaultFolder: 'voice' | 'ambience' | 'sfx' | 'music',
  genre?: string,
): string => {
  if (!src) return '';
  const trimmed = src.trim();
  const withExt = trimmed.endsWith('.wav') || trimmed.endsWith('.mp3') ? trimmed : `${trimmed}.wav`;

  if (withExt.startsWith('audio/')) {
    const after = withExt.slice('audio/'.length);
    if (after.includes('/')) {
      return withExt;
    }
    if (genre) {
      if (after.startsWith('amb_')) return `audio/genres/${genre}/ambience/${after}`;
      if (after.startsWith('sfx_')) return `audio/genres/${genre}/sfx/${after}`;
      if (defaultFolder) return `audio/genres/${genre}/${defaultFolder}/${after}`;
    }
    if (after.startsWith('amb_')) return `audio/ambience/${after}`;
    if (after.startsWith('sfx_')) return `audio/sfx/${after}`;
    if (defaultFolder) return `audio/${defaultFolder}/${after}`;
    return withExt;
  }

  if (genre) {
    if (withExt.startsWith('amb_')) return `audio/genres/${genre}/ambience/${withExt}`;
    if (withExt.startsWith('sfx_')) return `audio/genres/${genre}/sfx/${withExt}`;
    if (defaultFolder === 'music') return `audio/genres/${genre}/music/${withExt}`;
    if (defaultFolder === 'ambience') return `audio/genres/${genre}/ambience/${withExt}`;
    if (defaultFolder === 'sfx') return `audio/genres/${genre}/sfx/${withExt}`;
  }

  if (withExt.startsWith('amb_')) return `audio/ambience/${withExt}`;
  if (withExt.startsWith('sfx_')) return `audio/sfx/${withExt}`;
  return `audio/${defaultFolder}/${withExt}`;
};

const ProgressBar: React.FC<{ color: string }> = ({ color }) => {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();
  const progress = Math.min(1, (frame + 1) / durationInFrames);

  return (
    <div
      style={{
        position: "absolute",
        left: "50%",
        bottom: "32%", // Midway between bottom (15-18%) and vertical center (50%)
        transform: "translateX(-50%)",
        width: "88%",
        textAlign: "center",
        zIndex: 20,
      }}
    >
      {/* Caption Text Component */}
    </div>
  );
};

const FrameFade: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();
  const fadeIn = Math.round(fps * 0.5);
  const fadeOut = Math.round(fps * 0.9);

  const opacity = Math.min(
    interpolate(frame, [0, fadeIn], [1, 0], { extrapolateRight: 'clamp' }),
    interpolate(frame, [durationInFrames - fadeOut, durationInFrames], [0, 1], {
      extrapolateLeft: 'clamp',
    }),
  );

  if (opacity <= 0.001) {
    return null;
  }
  return <AbsoluteFill style={{ backgroundColor: '#000', opacity }} />;
};

export const Main: React.FC<Project> = ({ script, timings }) => {
  const { fps, durationInFrames } = useVideoConfig();

  const timingById = new Map(timings.scenes.map((t) => [t.id, t]));
  const sceneStarts = timings.scenes.map((t) => t.start);

  return (
    <AbsoluteFill style={{ backgroundColor: '#000' }}>
      {script.scenes.map((scene, i) => {
        const timing = timingById.get(scene.id);
        if (!timing) {
          return null;
        }
        const from = Math.round(timing.start * fps);
        const until =
          i + 1 < sceneStarts.length
            ? Math.round(sceneStarts[i + 1] * fps)
            : durationInFrames;
        const sceneFrames = Math.max(1, until - from);

        return (
          <Sequence
            key={scene.id}
            from={from}
            durationInFrames={sceneFrames}
            name={scene.headline || scene.id}
          >
            <SceneView
              scene={scene}
              timing={timing}
              durationInFrames={sceneFrames}
              theme={script.theme}
              captions={script.captions}
              characters={script.characters ?? {}}
              isFirstScene={i === 0}
            />
          </Sequence>
        );
      })}

      {/* Narration. Spoken line voiceover only. */}
      {timings.scenes.flatMap((timing) => {
        const lines =
          timing.lines && timing.lines.length > 0
            ? timing.lines
            : timing.audio
              ? [{ audio: timing.audio, start: timing.audioStart }]
              : [];

        return lines.map((line, i) =>
          line.audio ? (
            <Sequence key={`voice-${timing.id}-${i}`} from={Math.round(line.start * fps)}>
              <Audio src={staticFile(resolveAudioPath(line.audio, 'voice'))} />
            </Sequence>
          ) : null,
        );
      })}

      <ProgressBar color={script.theme.accent} />
      <FrameFade />
    </AbsoluteFill>
  );
};
