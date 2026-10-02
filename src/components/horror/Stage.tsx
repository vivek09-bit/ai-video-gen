import {AbsoluteFill, useCurrentFrame, useVideoConfig} from 'remotion';
import React from 'react';
import {FlickeringLight, Rain} from './Effects';
import {hash, noise1} from './noise';
import {VisualSpec} from '../../types';

/**
 * Procedural horror backdrops.
 *
 * Each preset is a stack of cheap layers: a near-black base, a light source,
 * one or more silhouettes, and fog. Silhouettes are near-pure black so anything
 * lit reads immediately, which is what makes this style legible at small sizes.
 *
 * Everything is a pure function of the frame number, so shots are reproducible.
 */

const BLACK = '#04060a';
const SILHOUETTE = '#05070b';

/** Drifting fog banks at several depths. */
const Fog: React.FC<{density: number; seed?: number; tint?: string}> = ({
  density,
  seed = 0,
  tint = '178, 190, 206',
}) => {
  const frame = useCurrentFrame();
  const {fps, width, height} = useVideoConfig();
  const time = frame / fps;

  if (density <= 0) {
    return null;
  }

  const layers = Math.round(2 + density * 4);
  const bands = [];

  for (let i = 0; i < layers; i++) {
    const key = i + seed * 13;
    const speed = 7 + hash(key) * 24;
    const depth = (i + 1) / (layers + 1);
    const alpha = (0.05 + density * 0.13) * (0.45 + hash(key * 3.3) * 0.85);
    const bandHeight = height * (0.28 + depth * 0.3);

    // Wrap the drift so fog is continuous and never pops.
    const span = width * 2.4;
    const drift = ((time * speed * 6) % span) - width * 0.7;

    // Soft multi-stop gradients instead of a CSS blur filter. A large-radius
    // blur on a full-width layer is the single most expensive thing Chromium
    // does here, and it adds nothing: the gradient is already smooth. Dropping
    // it roughly triples render throughput.
    bands.push(
      <div
        key={i}
        style={{
          position: 'absolute',
          left: drift,
          top: height * (0.34 + depth * 0.5) - bandHeight / 2,
          width: span,
          height: bandHeight,
          background: `radial-gradient(ellipse at 50% 50%, rgba(${tint}, ${alpha.toFixed(3)}) 0%, rgba(${tint}, ${(alpha * 0.52).toFixed(3)}) 38%, rgba(${tint}, ${(alpha * 0.18).toFixed(3)}) 64%, rgba(${tint}, 0) 82%)`,
          pointerEvents: 'none',
        }}
      />,
    );
  }

  return <AbsoluteFill style={{pointerEvents: 'none'}}>{bands}</AbsoluteFill>;
};

/** Jagged treeline. Generated once, deterministically. */
const TreeLine: React.FC<{seed?: number; baseline?: number}> = ({seed = 0, baseline = 1.0}) => {
  const {width} = useVideoConfig();
  const trees = [];
  const count = 16;

  for (let i = 0; i < count; i++) {
    const key = i + seed * 31;
    const x = (i / count) * 1400 + hash(key * 5.1) * 30;
    const treeHeight = 90 + hash(key * 9.7) * 190;
    const treeWidth = 34 + hash(key * 2.3) * 44;
    const top = 620 * baseline - treeHeight;

    // A tapered trunk plus two angled boughs reads as a bare tree at a glance.
    trees.push(
      <g key={i}>
        <path
          d={`M ${x} ${620 * baseline} L ${x + treeWidth / 2} ${top + treeHeight * 0.45} L ${x + treeWidth * 0.16} ${top} L ${x - treeWidth * 0.16} ${top + treeHeight * 0.2} L ${x - treeWidth / 2} ${top + treeHeight * 0.5} Z`}
          fill={SILHOUETTE}
        />
        <path
          d={`M ${x} ${top + treeHeight * 0.36} L ${x - treeWidth * 1.5} ${top + treeHeight * 0.16} L ${x - treeWidth * 0.2} ${top + treeHeight * 0.44} Z`}
          fill={SILHOUETTE}
        />
        <path
          d={`M ${x + treeWidth * 0.06} ${top + treeHeight * 0.3} L ${x + treeWidth * 1.7} ${top + treeHeight * 0.04} L ${x + treeWidth * 0.25} ${top + treeHeight * 0.38} Z`}
          fill={SILHOUETTE}
        />
      </g>,
    );
  }

  return (
    <svg
      viewBox="0 0 1400 620"
      preserveAspectRatio="none"
      width={width}
      height="62%"
      style={{position: 'absolute', bottom: 0, left: 0}}
    >
      <rect x={0} y={600 * baseline} width={1400} height={40} fill={SILHOUETTE} />
      {trees}
    </svg>
  );
};

/** A standing human figure, seen from a distance. */
const FigureShape: React.FC<{light: number}> = ({light}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const time = frame / fps;

  // A very slight sway: enough to suggest it is alive, not enough to be obvious.
  const sway = noise1(time * 0.37) * 1.6;

  return (
    <svg viewBox="0 0 200 400" width="100%" height="100%" style={{overflow: 'visible'}}>
      <g transform={`rotate(${sway.toFixed(2)} 100 380)`} fill={SILHOUETTE}>
        <ellipse cx={100} cy={62} rx={27} ry={32} />
        <path d="M 74 96 Q 100 82 126 96 L 134 210 Q 100 224 66 210 Z" />
        <path d="M 74 100 Q 52 130 48 232 L 66 236 Q 72 150 82 118 Z" />
        <path d="M 126 100 Q 148 130 152 232 L 134 236 Q 128 150 118 118 Z" />
        <path d="M 84 212 L 78 372 L 96 372 L 100 232 Z" />
        <path d="M 116 212 L 122 372 L 104 372 L 100 232 Z" />
      </g>
      {/* A faint rim light separates the figure from the fog behind it. */}
      <ellipse
        cx={100}
        cy={62}
        rx={27}
        ry={32}
        fill="none"
        stroke={`rgba(196, 214, 240, ${0.22 * light})`}
        strokeWidth={3}
      />
    </svg>
  );
};

/** A house, optionally with one lit window. */
const HouseShape: React.FC<{lit: boolean}> = ({lit}) => (
  <svg viewBox="0 0 400 300" width="100%" height="100%">
    <path d="M 40 300 L 40 150 L 200 44 L 360 150 L 360 300 Z" fill={SILHOUETTE} />
    <path d="M 20 160 L 200 34 L 380 160 L 366 176 L 200 56 L 34 176 Z" fill={SILHOUETTE} />
    <rect x={60} y={40} width={26} height={110} fill={SILHOUETTE} />
    {lit ? (
      <>
        <rect x={168} y={168} width={64} height={72} fill={`rgba(255, 196, 110, 0.88)`} />
        <rect x={168} y={168} width={64} height={72} fill="none" stroke={SILHOUETTE} strokeWidth={4} />
        <path d="M 200 168 L 200 240" stroke={SILHOUETTE} strokeWidth={4} />
      </>
    ) : null}
  </svg>
);

/** Perspective corridor, with a light at the far end. */
const CorridorShape: React.FC<{depth: number}> = ({depth}) => {
  const rings = [];
  const steps = Math.max(4, Math.round(4 + depth * 7));

  for (let i = 0; i < steps; i++) {
    const t = i / steps;
    const inset = t * 0.44;
    const shade = Math.round(16 - t * 10);
    rings.push(
      <rect
        key={i}
        x={`${inset * 100}%`}
        y={`${inset * 78}%`}
        width={`${(1 - inset * 2) * 100}%`}
        height={`${(1 - inset * 1.56) * 100}%`}
        fill="none"
        stroke={`rgb(${shade}, ${shade + 3}, ${shade + 8})`}
        strokeWidth={i === 0 ? 0 : 2}
      />,
    );
  }

  return (
    <AbsoluteFill>
      <div
        style={{
          position: 'absolute',
          inset: 0,
          background:
            'linear-gradient(90deg, rgba(0,0,0,0.96) 0%, rgba(0,0,0,0.5) 18%, rgba(0,0,0,0.34) 50%, rgba(0,0,0,0.5) 82%, rgba(0,0,0,0.96) 100%)',
        }}
      />
      <svg width="100%" height="100%" viewBox="0 0 100 100" preserveAspectRatio="none">
        {rings}
      </svg>
    </AbsoluteFill>
  );
};

/** Gravestones on a foggy rise. */
const GraveField: React.FC<{seed?: number}> = ({seed = 0}) => {
  const stones = [];
  const count = 11;

  for (let i = 0; i < count; i++) {
    const key = i + seed * 17;
    const x = 60 + i * 118 + hash(key * 3.7) * 26;
    const height = 74 + hash(key * 8.1) * 62;
    const width = 46 + hash(key * 4.9) * 20;
    const y = 470 + hash(key * 6.3) * 22;

    stones.push(
      <path
        key={i}
        d={`M ${x} ${y + height} L ${x} ${y + width / 2} Q ${x + width / 2} ${y - 10} ${x + width} ${y + width / 2} L ${x + width} ${y + height} Z`}
        fill={SILHOUETTE}
      />,
    );
  }

  return (
    <svg viewBox="0 0 1400 620" preserveAspectRatio="none" style={{position: 'absolute', inset: 0, width: '100%', height: '100%'}}>
      <rect x={0} y={540} width={1400} height={80} fill={SILHOUETTE} />
      {stones}
    </svg>
  );
};

/** Pairs of eyes in the dark, blinking out of sync. */
const Eyes: React.FC<{pairs: number; spread: number}> = ({pairs, spread}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const time = frame / fps;

  const items = [];

  for (let i = 0; i < pairs; i++) {
    const key = i * 7.7;
    const cx = 0.5 + (hash(key * 2.1) - 0.5) * spread;
    const cy = 0.34 + hash(key * 6.3) * 0.4;
    const scale = 0.5 + hash(key * 4.7) * 0.9;
    const period = 2.4 + hash(key * 9.1) * 3.4;
    const phase = hash(key * 1.3) * 4;
    const blink = ((time + phase) % period) < 0.14 ? 0.06 : 1;
    const glow = 0.55 + Math.abs(noise1(time * 1.6 + key)) * 0.45;

    items.push(
      <g key={i} transform={`translate(${cx * 1920} ${cy * 1080}) scale(${scale.toFixed(2)})`}>
        <ellipse cx={-30} cy={0} rx={17} ry={11} fill={`rgba(255, 226, 150, ${(glow * blink).toFixed(3)})`} />
        <ellipse cx={30} cy={0} rx={17} ry={11} fill={`rgba(255, 226, 150, ${(glow * blink).toFixed(3)})`} />
        <circle cx={-30} cy={0} r={4.5} fill="#120c04" />
        <circle cx={30} cy={0} r={4.5} fill="#120c04" />
      </g>,
    );
  }

  return (
    <svg viewBox="0 0 1920 1080" width="100%" height="100%">
      {items}
    </svg>
  );
};

export const Stage: React.FC<{spec: VisualSpec}> = ({spec}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const time = frame / fps;

  switch (spec.type) {
    case 'void': {
      const creep = spec.creep ?? 0.5;
      const closing = 1 - Math.min(1, 0.34 + Math.abs(noise1(time * 0.19)) * creep);
      return (
        <AbsoluteFill style={{backgroundColor: spec.tint ?? BLACK}}>
          <AbsoluteFill
            style={{
              background: `radial-gradient(ellipse at 50% 50%, rgba(0,0,0,0) ${(closing * 100).toFixed(1)}%, rgba(0,0,0,0.98) 100%)`,
            }}
          />
        </AbsoluteFill>
      );
    }

    case 'fog':
      return (
        <AbsoluteFill style={{backgroundColor: '#070a0f'}}>
          <FlickeringLight amount={0.2} x={0.5} y={0.45} radius={900} brightness={spec.glow ?? 0.5} cold />
          <Fog density={spec.density ?? 0.6} seed={spec.seed ?? 0} />
        </AbsoluteFill>
      );

    case 'forest':
      return (
        <AbsoluteFill style={{backgroundColor: '#05070c'}}>
          <FlickeringLight amount={0.12} x={0.62} y={0.18} radius={760} brightness={spec.moonlight ?? 0.7} cold />
          <div style={{position: 'absolute', inset: 0, top: '34%', bottom: 0}}>
            <TreeLine seed={2} />
          </div>
          <Fog density={spec.fog ?? 0.55} seed={3} />
        </AbsoluteFill>
      );

    case 'figure': {
      const distance = spec.distance ?? 0.42;
      const size = 0.16 + (1 - distance) * 0.34;
      return (
        <AbsoluteFill style={{backgroundColor: '#05070c'}}>
          <FlickeringLight amount={0.15} x={0.5} y={0.32} radius={820} brightness={spec.glow ?? 0.6} cold />
          <div
            style={{
              position: 'absolute',
              left: '50%',
              bottom: '14%',
              width: `${size * 100}%`,
              height: `${size * 160}%`,
              transform: 'translateX(-50%)',
            }}
          >
            <FigureShape light={1} />
          </div>
          <Fog density={spec.fog ?? 0.62} seed={5} />
        </AbsoluteFill>
      );
    }

    case 'house':
      return (
        <AbsoluteFill style={{backgroundColor: '#04070d'}}>
          <FlickeringLight amount={0.1} x={0.5} y={0.14} radius={880} brightness={spec.moonlight ?? 0.6} cold />
          <div
            style={{
              position: 'absolute',
              left: '50%',
              bottom: '12%',
              width: '46%',
              height: '58%',
              transform: 'translateX(-50%)',
            }}
          >
            <HouseShape lit={spec.lit ?? true} />
          </div>
          <Fog density={spec.fog ?? 0.5} seed={7} />
        </AbsoluteFill>
      );

    case 'corridor': {
      const depth = spec.depth ?? 0.5;
      return (
        <AbsoluteFill style={{backgroundColor: '#03050a'}}>
          <CorridorShape depth={depth} />
          {/* The light at the end of the corridor is what flickers. */}
          <div
            style={{
              position: 'absolute',
              left: '50%',
              top: '50%',
              width: `${(1 - depth * 0.6) * 16}%`,
              height: `${(1 - depth * 0.6) * 22}%`,
              transform: 'translate(-50%, -50%)',
            }}
          >
            <FlickeringLight
              amount={spec.flicker ?? 0.6}
              x={0.5}
              y={0.5}
              radius={300}
              brightness={0.9}
            />
          </div>
          <Fog density={0.3} seed={11} tint="150, 162, 180" />
        </AbsoluteFill>
      );
    }

    case 'candle':
      return (
        <AbsoluteFill style={{backgroundColor: '#08060a'}}>
          <FlickeringLight
            amount={0.75}
            x={0.5}
            y={0.62}
            radius={640}
            brightness={spec.glow ?? 0.9}
          />
          <AbsoluteFill
            style={{
              background: 'radial-gradient(ellipse at 50% 62%, rgba(0,0,0,0) 12%, rgba(0,0,0,0.9) 78%)',
            }}
          />
        </AbsoluteFill>
      );

    case 'rain':
      return (
        <AbsoluteFill style={{backgroundColor: '#04070c'}}>
          <FlickeringLight amount={0.3} x={0.28} y={0.24} radius={720} brightness={0.45} cold />
          <div style={{position: 'absolute', inset: 0, top: '40%', bottom: 0}}>
            <TreeLine seed={4} baseline={0.92} />
          </div>
          <Rain intensity={spec.intensity ?? 0.65} lightning={spec.lightning ?? true} />
        </AbsoluteFill>
      );
    case 'eyes':
      return (
        <AbsoluteFill style={{backgroundColor: '#030406'}}>
          <Eyes pairs={spec.pairs ?? 3} spread={spec.spread ?? 0.7} />
        </AbsoluteFill>
      );

    case 'grave':
      return (
        <AbsoluteFill style={{backgroundColor: '#05070b'}}>
          <FlickeringLight amount={0.1} x={0.5} y={0.2} radius={900} brightness={spec.moonlight ?? 0.55} cold />
          <GraveField seed={1} />
          <Fog density={spec.fog ?? 0.6} seed={13} />
        </AbsoluteFill>
      );

    default:
      return <AbsoluteFill style={{backgroundColor: BLACK}} />;
  }
};
