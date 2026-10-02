import {useCurrentFrame, useVideoConfig} from 'remotion';
import React from 'react';
import {MOUTH_OPENNESS, Mouth, MouthShape} from './Mouth';
import {CharacterSpec} from '../../types';

/**
 * A code-drawn 2D character.
 *
 * There is no rig file to load and no image assets to find: the character is
 * vector shapes positioned by transform, so it renders identically everywhere
 * and stays perfectly consistent across episodes. Reusing one `CharacterSpec`
 * guarantees the same character every time, which is the reason a rig beats
 * regenerating art per frame.
 *
 * All motion is a pure function of the frame number (no randomness), so renders
 * are deterministic and the studio preview matches the final output exactly.
 *
 * Local coordinate space is 400 x 640, character centred on x = 200.
 */

export interface CharacterProps {
  spec: CharacterSpec;
  mouth: MouthShape;
  talking: boolean;
  /** Extra energy for emphatic moments, 0..1. */
  emphasis?: number;
}

const DEFAULT_PALETTE: CharacterSpec['palette'] = {
  skin: '#f2c9a0',
  skinShade: '#dfa97e',
  hair: '#3a2a24',
  shirt: '#4f6f9f',
  accent: '#38bdf8',
  outline: '#232a33',
};

/** Fills in anything the script left out so a partial spec still renders. */
const resolve = (spec: CharacterSpec): CharacterSpec => ({
  ...spec,
  palette: {...DEFAULT_PALETTE, ...(spec.palette ?? {})},
  hairstyle: spec.hairstyle ?? 'short',
  accessory: spec.accessory ?? 'none',
});

/** Stable per-character phase so two characters never animate in lockstep. */
const seedOf = (value: string): number => {
  let hash = 0;
  for (let i = 0; i < value.length; i++) {
    hash = (hash * 31 + value.charCodeAt(i)) % 100000;
  }
  return (hash % 1000) / 1000;
};

const Hair: React.FC<{style: CharacterSpec['hairstyle']; hair: string; outline: string}> = ({
  style,
  hair,
  outline,
}) => {
  if (style === 'bald') {
    return null;
  }

  const cap = (
    <path
      d="M 104 158 Q 110 56 200 52 Q 290 56 296 158 Q 282 96 200 92 Q 118 96 104 158 Z"
      fill={hair}
      stroke={outline}
      strokeWidth={3}
      strokeLinejoin="round"
    />
  );

  if (style === 'short') {
    return cap;
  }

  if (style === 'bun') {
    return (
      <g>
        <circle cx={200} cy={36} r={31} fill={hair} stroke={outline} strokeWidth={3} />
        {cap}
      </g>
    );
  }

  // 'bob' stops at the jaw, 'long' falls past the shoulders.
  const length = style === 'bob' ? 252 : 358;
  return (
    <g>
      <path
        d={`M 100 150 Q 84 206 90 ${length} Q 120 ${length - 20} 126 214 Q 110 170 112 140 Z`}
        fill={hair}
        stroke={outline}
        strokeWidth={3}
        strokeLinejoin="round"
      />
      <path
        d={`M 300 150 Q 316 206 310 ${length} Q 280 ${length - 20} 274 214 Q 290 170 288 140 Z`}
        fill={hair}
        stroke={outline}
        strokeWidth={3}
        strokeLinejoin="round"
      />
      {cap}
    </g>
  );
};

const Arm: React.FC<{
  shoulderX: number;
  rotation: number;
  shirt: string;
  skin: string;
  outline: string;
}> = ({shoulderX, rotation, shirt, skin, outline}) => (
  <g transform={`translate(${shoulderX} 340) rotate(${rotation})`}>
    <path
      d="M -17 0 Q -21 88 -16 166 Q 0 179 16 166 Q 21 88 17 0 Z"
      fill={shirt}
      stroke={outline}
      strokeWidth={3}
      strokeLinejoin="round"
    />
    <circle cx={0} cy={178} r={21} fill={skin} stroke={outline} strokeWidth={3} />
  </g>
);

export const Character: React.FC<CharacterProps> = ({spec, mouth, talking, emphasis = 0}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const time = frame / fps;

  const resolved = resolve(spec);
  const {palette, hairstyle, accessory} = resolved;
  const seed = seedOf(resolved.name || 'character');

  // Body language scales with how open the mouth is, so the pose follows speech.
  const openness = MOUTH_OPENNESS[mouth] ?? 0.1;
  const speech = talking ? Math.max(0.3, openness) : 0;
  const energy = talking ? speech * (1 + emphasis) : 0;

  const breath = Math.sin((time / 3.6) * Math.PI * 2 + seed * 6.28);
  const sway = Math.sin((time / 5.2) * Math.PI * 2 + seed * 4.1);
  const bob = Math.sin((time / 2.9) * Math.PI * 2 + seed * 2.3);

  // Blinks land on an irregular but deterministic schedule.
  const blinkPeriod = 3.2 + seed * 1.6;
  const blinkAt = (time + seed * 1.7) % blinkPeriod;
  const blink = blinkAt < 0.13 ? Math.sin((blinkAt / 0.13) * Math.PI) : 0;

  const headY = -energy * 3.4 + bob * 1.5 + breath * 1.1;
  const headTilt = sway * 2.4 + (talking ? Math.sin(time * 7.5 + seed) * 0.9 * energy : 0);
  const bodyY = breath * 2.4;
  const bodyRotate = sway * 1.2;
  const armSwing = sway * 4.5;
  const gesture = energy * 17;

  const pupilX = Math.sin(time * 0.7 + seed * 3) * 2.2;
  const pupilY = Math.cos(time * 0.5 + seed * 2) * 1.4;

  const armLeft = armSwing + gesture * 0.35;
  const armRight = -(armSwing * 0.4 + gesture);

  const eye = (cx: number) => (
    <g transform={`translate(${cx} 158) scale(1 ${Math.max(0.06, 1 - blink * 0.94)})`}>
      <ellipse
        cx={0}
        cy={0}
        rx={17}
        ry={19}
        fill="#ffffff"
        stroke={palette.outline}
        strokeWidth={2.5}
      />
      <circle cx={pupilX} cy={pupilY + 1} r={9} fill="#2b2b33" />
      <circle cx={pupilX + 3.2} cy={pupilY - 3} r={2.8} fill="#ffffff" opacity={0.92} />
    </g>
  );

  return (
    <svg viewBox="0 0 400 640" width="100%" height="100%" style={{overflow: 'visible'}}>
      <ellipse cx={200} cy={620} rx={118} ry={15} fill="#000000" opacity={0.22} />

      <Arm
        shoulderX={132}
        rotation={armLeft}
        shirt={palette.shirt}
        skin={palette.skin}
        outline={palette.outline}
      />
      <Arm
        shoulderX={268}
        rotation={armRight}
        shirt={palette.shirt}
        skin={palette.skin}
        outline={palette.outline}
      />

      <g transform={`translate(0 ${bodyY}) rotate(${bodyRotate} 200 560)`}>
        <path d="M 180 244 L 220 244 L 220 326 L 180 326 Z" fill={palette.skinShade} />
        <path
          d="M 128 320 Q 200 290 272 320 L 288 566 Q 200 592 112 566 Z"
          fill={palette.shirt}
          stroke={palette.outline}
          strokeWidth={3}
          strokeLinejoin="round"
        />
        <path d="M 176 300 Q 200 336 224 300 L 240 306 Q 200 356 160 306 Z" fill={palette.accent} opacity={0.9} />

        <g transform={`translate(0 ${headY}) rotate(${headTilt} 200 250)`}>
          <ellipse cx={106} cy={184} rx={15} ry={23} fill={palette.skin} stroke={palette.outline} strokeWidth={3} />
          <ellipse cx={294} cy={184} rx={15} ry={23} fill={palette.skin} stroke={palette.outline} strokeWidth={3} />
          <ellipse cx={200} cy={166} rx={96} ry={105} fill={palette.skin} stroke={palette.outline} strokeWidth={3} />

          <Hair style={hairstyle} hair={palette.hair} outline={palette.outline} />

          <ellipse cx={136} cy={198} rx={20} ry={12} fill={palette.accent} opacity={0.2} />
          <ellipse cx={264} cy={198} rx={20} ry={12} fill={palette.accent} opacity={0.2} />

          {eye(166)}
          {eye(234)}

          <path
            d="M 140 122 Q 166 110 190 122"
            fill="none"
            stroke={palette.hair}
            strokeWidth={7}
            strokeLinecap="round"
          />
          <path
            d="M 210 122 Q 234 110 260 122"
            fill="none"
            stroke={palette.hair}
            strokeWidth={7}
            strokeLinecap="round"
          />

          <path
            d="M 200 176 Q 209 192 197 194"
            fill="none"
            stroke={palette.skinShade}
            strokeWidth={3.5}
            strokeLinecap="round"
          />

          {accessory === 'glasses' ? (
            <g fill="none" stroke={palette.outline} strokeWidth={3.5} opacity={0.85}>
              <rect x={140} y={134} width={52} height={48} rx={14} />
              <rect x={208} y={134} width={52} height={48} rx={14} />
              <path d="M 192 156 L 208 156" />
            </g>
          ) : null}

          <g transform="translate(200 214) scale(1.05)">
            <Mouth shape={mouth} outline={palette.outline} />
          </g>
        </g>
      </g>
    </svg>
  );
};
