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
import {SceneView} from './components/SceneView';
import {Project} from './types';

const ProgressBar: React.FC<{color: string}> = ({color}) => {
  const frame = useCurrentFrame();
  const {durationInFrames} = useVideoConfig();
  const progress = Math.min(1, (frame + 1) / durationInFrames);

  return (
    <div
      style={{
        position: 'absolute',
        left: 0,
        bottom: 0,
        height: 6,
        width: `${progress * 100}%`,
        backgroundColor: color,
        opacity: 0.85,
      }}
    />
  );
};

const FrameFade: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps, durationInFrames} = useVideoConfig();
  const fadeIn = Math.round(fps * 0.5);
  const fadeOut = Math.round(fps * 0.9);

  const opacity = Math.min(
    interpolate(frame, [0, fadeIn], [1, 0], {extrapolateRight: 'clamp'}),
    interpolate(frame, [durationInFrames - fadeOut, durationInFrames], [0, 1], {
      extrapolateLeft: 'clamp',
    }),
  );

  if (opacity <= 0.001) {
    return null;
  }
  return <AbsoluteFill style={{backgroundColor: '#000', opacity}} />;
};

export const Main: React.FC<Project> = ({script, timings}) => {
  const {fps, durationInFrames} = useVideoConfig();

  const timingById = new Map(timings.scenes.map((t) => [t.id, t]));
  const sceneStarts = timings.scenes.map((t) => t.start);

  const musicVolume = (frame: number): number => {
    if (!script.music) {
      return 0;
    }
    const fadeIn = Math.round(fps * 1.5);
    const fadeOut = Math.round(script.music.fadeOutSeconds * fps);
    const rampIn = interpolate(frame, [0, fadeIn], [0, 1], {extrapolateRight: 'clamp'});
    const rampOut = interpolate(
      frame,
      [durationInFrames - fadeOut, durationInFrames],
      [1, 0],
      {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'},
    );
    return script.music.volume * Math.min(rampIn, rampOut);
  };

  return (
    <AbsoluteFill style={{backgroundColor: '#000'}}>
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
            />
          </Sequence>
        );
      })}

      {/* Narration. Each spoken line is placed at its own absolute start, which
          is what lets several characters talk in turn inside one scene. */}
      {timings.scenes.flatMap((timing) => {
        const lines =
          timing.lines && timing.lines.length > 0
            ? timing.lines
            : timing.audio
              ? [{audio: timing.audio, start: timing.audioStart}]
              : [];

        return lines.map((line, i) =>
          line.audio ? (
            <Sequence key={`voice-${timing.id}-${i}`} from={Math.round(line.start * fps)}>
              <Audio src={staticFile(line.audio)} />
            </Sequence>
          ) : null,
        );
      })}

      {/* Looping ambience bed per scene, mixed under the narration. */}
      {script.scenes.map((scene) => {
        if (!scene.ambience) {
          return null;
        }
        const timing = timingById.get(scene.id);
        if (!timing) {
          return null;
        }
        return (
          <Sequence
            key={`amb-${scene.id}`}
            from={Math.round(timing.start * fps)}
            durationInFrames={Math.max(1, Math.round(timing.duration * fps))}
          >
            <Audio
              src={staticFile(`audio/${scene.ambience.src}.wav`)}
              loop
              loopVolumeCurveBehavior="extend"
              volume={scene.ambience.volume}
            />
          </Sequence>
        );
      })}

      {/* One-shot stingers, risers and impacts, placed at story beats. */}
      {script.scenes.flatMap((scene) => {
        const timing = timingById.get(scene.id);
        if (!timing) {
          return [];
        }
        return (scene.soundCues ?? []).map((cue, index) => (
          <Sequence
            key={`sfx-${scene.id}-${index}`}
            from={Math.round((timing.start + cue.at) * fps)}
          >
            <Audio src={staticFile(`audio/${cue.sfx}.wav`)} volume={cue.volume ?? 1} />
          </Sequence>
        ));
      })}

      {script.music ? (
        <Audio
          src={staticFile(script.music.src)}
          loop
          // Without 'extend' the volume curve restarts on every loop iteration,
          // which would replay the fade-in in the middle of the video.
          loopVolumeCurveBehavior="extend"
          volume={musicVolume}
        />
      ) : null}

      <ProgressBar color={script.theme.accent} />
      <FrameFade />
    </AbsoluteFill>
  );
};
