import {Composition} from 'remotion';
import React from 'react';
import {Main} from './Main';
import projectJson from './generated/project.json';
import {Project} from './types';

// `src/generated/project.json` is produced by `npm run voice` (scripts/tts.py).
// Run that first if this file is missing.
const project = projectJson as unknown as Project;

const {script, timings} = project;
const durationInFrames = Math.max(
  1,
  Math.ceil(timings.totalDuration * script.fps) +
    Math.round(script.tailPaddingSeconds * script.fps),
);

export const RemotionRoot: React.FC = () => {
  return (
    <Composition
      id="Main"
      component={Main}
      durationInFrames={durationInFrames}
      fps={script.fps}
      width={script.width}
      height={script.height}
      defaultProps={project}
    />
  );
};
