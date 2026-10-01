import {AbsoluteFill, Img, Video, staticFile, useCurrentFrame} from 'remotion';
import React from 'react';
import {BackgroundSpec} from '../types';

const fill: React.CSSProperties = {width: '100%', height: '100%', objectFit: 'cover'};

const KenBurnsImage: React.FC<{
  src: string;
  mode: 'in' | 'out' | 'left' | 'right' | 'none';
  durationInFrames: number;
}> = ({src, mode, durationInFrames}) => {
  const frame = useCurrentFrame();
  const p = durationInFrames <= 1 ? 0 : Math.min(1, Math.max(0, frame / (durationInFrames - 1)));

  let scale = 1;
  let x = 0;
  switch (mode) {
    case 'in':
      scale = 1 + 0.14 * p;
      break;
    case 'out':
      scale = 1.14 - 0.14 * p;
      break;
    case 'left':
      scale = 1.16;
      x = 40 - 80 * p;
      break;
    case 'right':
      scale = 1.16;
      x = -40 + 80 * p;
      break;
    case 'none':
      break;
  }

  return (
    <AbsoluteFill style={{overflow: 'hidden'}}>
      <Img
        src={staticFile(src)}
        style={{
          ...fill,
          transform: `translateX(${x}px) scale(${scale})`,
        }}
      />
    </AbsoluteFill>
  );
};

export const Background: React.FC<{
  spec: BackgroundSpec;
  durationInFrames: number;
}> = ({spec, durationInFrames}) => {
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
        <KenBurnsImage src={spec.src} mode={spec.kenBurns} durationInFrames={durationInFrames} />
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
