import {Composition} from 'remotion';
import React from 'react';
import {Main} from './Main';
import projectJson from './generated/project.json';
import {Project} from './types';

// `src/generated/project.json` is produced by `npm run voice` (scripts/tts.py).
// Run that first if this file is missing.
const project = projectJson as unknown as Project;

const {script, timings} = project;
const fps = Number(script?.fps) || 30;
const totalDuration = Number(timings?.totalDuration) || 0;
const tailPadding = Number(script?.tailPaddingSeconds ?? 1.2) || 1.2;
const calculatedFrames = Math.ceil(totalDuration * fps) + Math.round(tailPadding * fps);
const durationInFrames = Number.isFinite(calculatedFrames) && calculatedFrames > 0
  ? Math.round(calculatedFrames)
  : 30;

export const RemotionRoot: React.FC = () => {
  return (
    <Composition
      id="Main"
      component={Main}
      durationInFrames={durationInFrames}
      fps={fps}
      width={Number(script?.width) || 1080}
      height={Number(script?.height) || 1920}
      defaultProps={project}
    />
  );
};
