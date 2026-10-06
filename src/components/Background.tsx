import {AbsoluteFill, Img, Video, staticFile, useCurrentFrame} from 'remotion';
import React from 'react';
import {BackgroundSpec} from '../types';

const fill: React.CSSProperties = {width: '100%', height: '100%', objectFit: 'cover'};

const KenBurnsImage: React.FC<{
  src: string;
  mode: 'in' | 'out' | 'left' | 'right' | 'none';
  durationInFrames: number;
  isFirstScene?: boolean;
}> = ({src, mode, durationInFrames, isFirstScene = false}) => {
  const frame = useCurrentFrame();
  const rawP = durationInFrames <= 1 ? 0 : Math.min(1, Math.max(0, frame / (durationInFrames - 1)));
  // Smooth sine easing for natural, cinematic camera movement
  const p = Math.sin((rawP * Math.PI) / 2);
  // Gentle entrance crossfade to prevent visual jar on scene cuts
  const sceneFadeIn = Math.min(1, frame / 7);

  let scale = 1;
  let x = 0;
  let y = 0;
  switch (mode) {
    case 'in':
      scale = 1 + 0.16 * p;
      y = -10 * p;
      break;
    case 'out':
      scale = 1.16 - 0.14 * p;
      y = 8 * p;
      break;
    case 'left':
      scale = 1.15;
      x = 35 - 70 * p;
      break;
    case 'right':
      scale = 1.15;
      x = -35 + 70 * p;
      break;
    case 'none':
      scale = 1.0;
      break;
  }

  return (
    <AbsoluteFill style={{overflow: 'hidden', opacity: isFirstScene ? 1 : sceneFadeIn}}>
      <Img
        src={staticFile(src)}
        style={{
          ...fill,
          transform: `translate3d(${x}px, ${y}px, 0) scale(${scale})`,
        }}
      />
    </AbsoluteFill>
  );
};

export const Background: React.FC<{
  spec: BackgroundSpec;
  durationInFrames: number;
  isFirstScene?: boolean;
}> = ({spec, durationInFrames, isFirstScene = false}) => {
  if (spec.type === 'color') {
    return <AbsoluteFill style={{backgroundColor: spec.color}} />;
  }

  if (spec.type === 'gradient') {
    return (
      <AbsoluteFill
        style={{
          backgroundImage: `linear-gradient(${spec.angle}deg, ${spec.from}, ${spec.to})`,
        }}
      />
    );
  }

  if (spec.type === 'pattern') {
    const size = 48;
    const image =
      spec.pattern === 'dots'
        ? `radial-gradient(${spec.line} 2px, transparent 2px)`
        : spec.pattern === 'diagonal'
          ? `repeating-linear-gradient(45deg, ${spec.line} 0 2px, transparent 2px ${size}px)`
          : `linear-gradient(${spec.line} 1px, transparent 1px), linear-gradient(90deg, ${spec.line} 1px, transparent 1px)`;

    return (
      <AbsoluteFill
        style={{
          backgroundColor: spec.bg,
          backgroundImage: image,
          backgroundSize: spec.pattern === 'grid' ? `${size}px ${size}px` : `${size}px ${size}px`,
        }}
      />
    );
  }

  if (spec.type === 'image') {
    return (
      <>
        <KenBurnsImage src={spec.src} mode={spec.kenBurns} durationInFrames={durationInFrames} isFirstScene={isFirstScene} />
        {spec.dim > 0 ? (
          <AbsoluteFill style={{backgroundColor: `rgba(0,0,0,${spec.dim})`}} />
        ) : null}
      </>
    );
  }

  return (
    <>
      <AbsoluteFill>
        {/* <Video> (not <OffthreadVideo>) is required here: only it supports `loop`. */}
        <Video src={staticFile(spec.src)} muted loop pauseWhenBuffering style={fill} />
      </AbsoluteFill>
      {spec.dim > 0 ? (
        <AbsoluteFill style={{backgroundColor: `rgba(0,0,0,${spec.dim})`}} />
      ) : null}
    </>
  );
};
