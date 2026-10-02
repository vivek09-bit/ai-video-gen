import {useCurrentFrame, useVideoConfig} from 'remotion';
import React from 'react';
import {MOUTH_OPENNESS, Mouth, MouthShape} from './Mouth';
import {ActingSpec, CharacterSpec, EmotionName, Facing, GestureName} from '../../types';

/**
 * A code-drawn 2D character that performs.
 *
 * There is no rig file to load and no image assets to find: the character is
 * vector shapes positioned by transform, so it renders identically everywhere
 * and stays consistent across episodes.
 *
 * Performance comes from `ActingSpec` — emotion, head turn, gaze and gesture —
 * rather than from the art. That is deliberate: better art can be dropped in
 * later without touching the acting system, and a character can be directed from
 * plain data by a planner or an LLM.
 *
 * Motion is a pure function of the frame number, so renders are reproducible and
 * the studio preview matches the final output exactly.
 *
 * Local coordinate space is 400 x 640, character centred on x = 200.
 */

export interface CharacterProps {
  spec: CharacterSpec;
  mouth: MouthShape;
  talking: boolean;
  acting?: ActingSpec;
}

const DEFAULT_PALETTE: CharacterSpec['palette'] = {
  skin: '#f2c9a0',
  skinShade: '#dfa97e',
  hair: '#3a2a24',
  shirt: '#4f6f9f',
  accent: '#38bdf8',
  outline: '#232a33',
};

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

/**
 * How each emotion deforms the face.
 *
 * Brow geometry carries most of the signal: raising the inner ends reads as
 * worry or grief, dropping them reads as anger. After that, how far the upper
 * lid covers the eye, and pupil size, do the rest. Numbers are pixel offsets in
 * the 400x640 space, where negative y is up.
 */
interface EmotionShape {
  browInner: number;
  browOuter: number;
  browArch: number;
  browWeight: number;
  /** Vertical scale of the eye opening. */
  eyeOpen: number;
  /** How much of the eye the upper lid covers, 0-1. */
  lidDrop: number;
  pupil: number;
  mouthScaleX: number;
  mouthScaleY: number;
  mouthOffsetY: number;
  cheek: number;
  sweat: number;
}

const NEUTRAL: EmotionShape = {
  browInner: 0,
  browOuter: 0,
  browArch: 0,
  browWeight: 7,
  eyeOpen: 1,
  lidDrop: 0.04,
  pupil: 1,
  mouthScaleX: 1,
  mouthScaleY: 1,
  mouthOffsetY: 0,
  cheek: 0,
  sweat: 0,
};

const EMOTIONS: Record<EmotionName, EmotionShape> = {
  neutral: NEUTRAL,
  calm: {...NEUTRAL, browArch: 2, lidDrop: 0.12, mouthScaleX: 1.04},
  // Inner ends up, outer ends down: the classic worried brow.
  worried: {...NEUTRAL, browInner: -7, browOuter: 4, browArch: -1, pupil: 1.06, mouthOffsetY: 1.5},
  scared: {
    ...NEUTRAL,
    browInner: -8,
    browOuter: -2,
    eyeOpen: 1.16,
    lidDrop: 0,
    pupil: 0.8,
    mouthScaleY: 1.12,
    sweat: 0.5,
  },
  shock: {
    ...NEUTRAL,
    browInner: -10,
    browOuter: -8,
    eyeOpen: 1.3,
    lidDrop: 0,
    pupil: 0.68,
    mouthScaleX: 0.9,
    mouthScaleY: 1.2,
  },
  angry: {
    ...NEUTRAL,
    browInner: 9,
    browOuter: -5,
    browWeight: 8.5,
    lidDrop: 0.3,
    pupil: 0.88,
    mouthScaleX: 0.94,
    cheek: 0.18,
  },
  sad: {
    ...NEUTRAL,
    browInner: -9,
    browOuter: 6,
    lidDrop: 0.24,
    pupil: 1.1,
    mouthScaleX: 0.92,
    mouthOffsetY: 2,
  },
  sly: {...NEUTRAL, browOuter: -6, browArch: -2, lidDrop: 0.42, pupil: 0.95, mouthScaleX: 1.08},
};

const GESTURES: Record<GestureName, {left: number; right: number}> = {
  none: {left: 0, right: 0},
  explain: {left: -10, right: -16},
  point: {left: 8, right: -32},
  reach: {left: -20, right: -38},
  recoil: {left: 16, right: 26},
  clutch: {left: 12, right: -6},
};

/** Horizontal shift of the facial features that suggests the head is turned. */
const FACING_SHIFT: Record<Facing, number> = {left: -7, right: 7, center: 0};
/** Pupil offset for gaze direction. */
const GAZE_SHIFT: Record<Facing, number> = {left: -5.5, right: 5.5, center: 0};

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

const Eye: React.FC<{
  cx: number;
  clipId: string;
  eyeRx: number;
  eyeRy: number;
  pupilR: number;
  pupilX: number;
  pupilY: number;
  blink: number;
  lidDrop: number;
  skin: string;
  outline: string;
}> = ({cx, clipId, eyeRx, eyeRy, pupilR, pupilX, pupilY, blink, lidDrop, skin, outline}) => {
  const squash = Math.max(0.06, 1 - blink * 0.94);
  return (
    <g transform={`translate(${cx} 158)`}>
      <g transform={`scale(1 ${squash})`}>
        <ellipse cx={0} cy={0} rx={eyeRx} ry={eyeRy} fill="#ffffff" stroke={outline} strokeWidth={2.5} />
      </g>
      <defs>
        <clipPath id={clipId}>
          <ellipse cx={0} cy={0} rx={eyeRx} ry={eyeRy * squash} />
        </clipPath>
      </defs>
      <g clipPath={`url(#${clipId})`}>
        <circle cx={pupilX} cy={pupilY + 1} r={pupilR} fill="#2b2b33" />
        <circle cx={pupilX + 3.2} cy={pupilY - 3} r={2.8} fill="#ffffff" opacity={0.92} />
        {/* The upper lid is what separates angry from shocked. */}
        {lidDrop > 0.02 ? (
          <rect
            x={-eyeRx - 2}
            y={-eyeRy - 2}
            width={(eyeRx + 2) * 2}
            height={(eyeRy + 2) * 2 * lidDrop + 2}
            fill={skin}
          />
        ) : null}
      </g>
      <ellipse
        cx={0}
        cy={0}
        rx={eyeRx}
        ry={eyeRy * squash}
        fill="none"
        stroke={outline}
        strokeWidth={2.5}
      />
    </g>
  );
};

export const Character: React.FC<CharacterProps> = ({spec, mouth, talking, acting}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const time = frame / fps;

  const resolved = resolve(spec);
  const {palette, hairstyle, accessory} = resolved;
  const seed = seedOf(resolved.name || 'character');
  const idBase = (resolved.name || 'c').replace(/[^A-Za-z0-9]/g, '');

  const emotion = EMOTIONS[acting?.emotion ?? 'neutral'] ?? NEUTRAL;
  const intensity = Math.max(0, Math.min(1, acting?.intensity ?? 0.6));
  const head = acting?.head ?? 'center';
  const eyes = acting?.eyes ?? 'center';
  const gesture = GESTURES[acting?.gesture ?? 'none'] ?? GESTURES.none;

  // Body language scales with how open the mouth is, so the pose follows speech.
  const openness = MOUTH_OPENNESS[mouth] ?? 0.1;
  const speech = talking ? Math.max(0.3, openness) : 0;
  const energy = talking ? speech * (1 + intensity) : 0;

  const breath = Math.sin((time / 3.6) * Math.PI * 2 + seed * 6.28);
  const sway = Math.sin((time / 5.2) * Math.PI * 2 + seed * 4.1);
  const bob = Math.sin((time / 2.9) * Math.PI * 2 + seed * 2.3);

  // Blinks land on an irregular but deterministic schedule. Fear suppresses them
  // a little, which is a real and useful acting detail.
  const blinkPeriod = 3.2 + seed * 1.6;
  const blinkAt = (time + seed * 1.7) % blinkPeriod;
  const blinkWave = blinkAt < 0.13 ? Math.sin((blinkAt / 0.13) * Math.PI) : 0;
  const blink = blinkWave * (acting?.emotion === 'shock' ? 0.25 : 1);

  // A held gaze drifts far less than a neutral one.
  const gazeDriftScale = head === 'center' && eyes === 'center' ? 1 : 0.25;

  const headY = -energy * 3.4 + bob * 1.5 + breath * 1.1;
  const headTilt = sway * 2.4 + (talking ? Math.sin(time * 7.5 + seed) * 0.9 * energy : 0);
  const bodyY = breath * 2.4;
  const bodyRotate = sway * 1.2 * (1 + intensity * 0.5);

  const pupilX = Math.sin(time * 0.7 + seed * 3) * 2.2 * gazeDriftScale + GAZE_SHIFT[eyes];
  const pupilY = Math.cos(time * 0.5 + seed * 2) * 1.4;
  const faceShiftX = FACING_SHIFT[head];

  // Gestures ride on top of the idle sway, so they never look pasted on.
  const gestureEase = 0.55 + intensity * 0.45;
  const armLeft = sway * 4.5 + gesture.left * gestureEase + energy * 6;
  const armRight = -(sway * 1.8) + gesture.right * gestureEase - energy * 10;

  const eyeRx = 17;
  const eyeRy = 19 * emotion.eyeOpen;
  const pupilR = 9 * emotion.pupil;

  /** Brows carry most of the emotional read, so they are fully parameterised. */
  const Brow: React.FC<{side: 'left' | 'right'}> = ({side}) => {
    const outerX = side === 'left' ? 140 : 260;
    const innerX = side === 'left' ? 190 : 210;
    const outerY = 122 + emotion.browOuter;
    const innerY = 122 + emotion.browInner;
    const midX = (outerX + innerX) / 2;
    const midY = (outerY + innerY) / 2 - 10 - emotion.browArch;
    return (
      <path
        d={`M ${outerX} ${outerY} Q ${midX} ${midY} ${innerX} ${innerY}`}
        fill="none"
        stroke={palette.hair}
        strokeWidth={emotion.browWeight}
        strokeLinecap="round"
      />
    );
  };

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
        <path
          d="M 176 300 Q 200 336 224 300 L 240 306 Q 200 356 160 306 Z"
          fill={palette.accent}
          opacity={0.9}
        />

        <g transform={`translate(0 ${headY}) rotate(${headTilt} 200 250)`}>
          <ellipse cx={106} cy={184} rx={15} ry={23} fill={palette.skin} stroke={palette.outline} strokeWidth={3} />
          <ellipse cx={294} cy={184} rx={15} ry={23} fill={palette.skin} stroke={palette.outline} strokeWidth={3} />
          <ellipse cx={200} cy={166} rx={96} ry={105} fill={palette.skin} stroke={palette.outline} strokeWidth={3} />

          <Hair style={hairstyle} hair={palette.hair} outline={palette.outline} />

          {/* Everything facial shifts together, which is what sells a head turn
              in a purely 2D drawing. */}
          <g transform={`translate(${faceShiftX} 0)`}>
            {emotion.cheek > 0 ? (
              <>
                <ellipse cx={140} cy={200} rx={21} ry={12} fill="#e2707f" opacity={emotion.cheek} />
                <ellipse cx={260} cy={200} rx={21} ry={12} fill="#e2707f" opacity={emotion.cheek} />
              </>
            ) : null}

            <Eye
              cx={166}
              clipId={`${idBase}-l`}
              eyeRx={eyeRx}
              eyeRy={eyeRy}
              pupilR={pupilR}
              pupilX={pupilX}
              pupilY={pupilY}
              blink={blink}
              lidDrop={emotion.lidDrop}
              skin={palette.skin}
              outline={palette.outline}
            />
            <Eye
              cx={234}
              clipId={`${idBase}-r`}
              eyeRx={eyeRx}
              eyeRy={eyeRy}
              pupilR={pupilR}
              pupilX={pupilX}
              pupilY={pupilY}
              blink={blink}
              lidDrop={emotion.lidDrop}
              skin={palette.skin}
              outline={palette.outline}
            />
            <Brow side="left" />
            <Brow side="right" />

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
              <g
                transform={`translate(0 ${emotion.mouthOffsetY}) scale(${emotion.mouthScaleX} ${emotion.mouthScaleY})`}
              >
                <Mouth shape={mouth} outline={palette.outline} />
              </g>
            </g>

            {/* A single sweat bead does a lot of work for fear. */}
            {emotion.sweat > 0 ? (
              <ellipse
                cx={258}
                cy={128}
                rx={5}
                ry={7}
                fill="#9fd8f5"
                opacity={emotion.sweat * (0.6 + Math.sin(time * 3.1) * 0.25)}
              />
            ) : null}
          </g>
        </g>
      </g>
    </svg>
  );
};
