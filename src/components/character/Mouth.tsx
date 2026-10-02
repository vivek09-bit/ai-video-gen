import React from 'react';

/**
 * The Hanna-Barbera mouth shape set that Rhubarb Lip Sync emits.
 * A-F are always produced; G, H and X are Rhubarb's "extended" shapes.
 */
export type MouthShape = 'A' | 'B' | 'C' | 'D' | 'E' | 'F' | 'G' | 'H' | 'X';

/**
 * How open each shape is, 0 (closed) to 1 (widest). Used to drive head bob and
 * arm gestures so the body reacts to speech rather than just the mouth.
 */
export const MOUTH_OPENNESS: Record<MouthShape, number> = {
  A: 0,
  B: 0.2,
  C: 0.5,
  D: 1,
  E: 0.6,
  F: 0.35,
  G: 0.3,
  H: 0.45,
  X: 0.05,
};

const DARK = '#3b1418';
const TEETH = '#fbfbfb';
const TONGUE = '#dd7280';

/**
 * Draws a mouth centred on the origin. The parent applies position and scale,
 * so every shape shares one coordinate space of roughly 84 x 46 units.
 */
export const Mouth: React.FC<{
  shape: MouthShape;
  outline: string;
}> = ({shape, outline}) => {
  switch (shape) {
    // Closed lips, as when saying P, B or M.
    case 'A':
      return (
        <path
          d="M -30 2 Q 0 -6 30 2 Q 0 10 -30 2 Z"
          fill={DARK}
          stroke={outline}
          strokeWidth={3}
          strokeLinejoin="round"
        />
      );

    // Slightly parted, teeth close together: EE, K, S, T.
    case 'B':
      return (
        <g>
          <path d="M -28 0 Q 0 -9 28 0 Q 0 13 -28 0 Z" fill={DARK} />
          <path d="M -26 -1 Q 0 -8 26 -1 L 26 2 Q 0 -2 -26 2 Z" fill={TEETH} />
          <path
            d="M -28 0 Q 0 -9 28 0 Q 0 13 -28 0 Z"
            fill="none"
            stroke={outline}
            strokeWidth={2.5}
          />
        </g>
      );

    // Moderately open: EH, AE.
    case 'C':
      return (
        <g>
          <ellipse cx={0} cy={1} rx={22} ry={14} fill={DARK} />
          <path d="M -20 -5 Q 0 -12 20 -5 L 20 -2 Q 0 -7 -20 -2 Z" fill={TEETH} />
          <ellipse cx={0} cy={1} rx={22} ry={14} fill="none" stroke={outline} strokeWidth={2.5} />
        </g>
      );

    // Widest open, as in AA.
    case 'D':
      return (
        <g>
          <ellipse cx={0} cy={2} rx={30} ry={22} fill={DARK} />
          <path d="M -27 -10 Q 0 -18 27 -10 L 27 -6 Q 0 -13 -27 -6 Z" fill={TEETH} />
          <ellipse cx={0} cy={16} rx={15} ry={8} fill={TONGUE} />
          <ellipse cx={0} cy={2} rx={30} ry={22} fill="none" stroke={outline} strokeWidth={2.5} />
        </g>
      );

    // Rounded, as in AO or ER.
    case 'E':
      return (
        <g>
          <ellipse cx={0} cy={2} rx={20} ry={19} fill={DARK} />
          <path d="M -17 -11 Q 0 -17 17 -11 L 17 -8 Q 0 -13 -17 -8 Z" fill={TEETH} />
          <ellipse cx={0} cy={2} rx={20} ry={19} fill="none" stroke={outline} strokeWidth={2.5} />
        </g>
      );

    // Puckered, as in UW, OW or W.
    case 'F':
      return (
        <g>
          <ellipse cx={0} cy={1} rx={11} ry={12} fill={DARK} />
          <ellipse cx={0} cy={1} rx={11} ry={12} fill="none" stroke={outline} strokeWidth={3} />
        </g>
      );

    // Upper teeth resting on the lower lip: F or V.
    case 'G':
      return (
        <g>
          <path d="M -24 2 Q 0 -6 24 2 Q 0 14 -24 2 Z" fill={DARK} />
          <path d="M -22 -1 Q 0 -7 22 -1 L 22 3 Q 0 -1 -22 3 Z" fill={TEETH} />
          <path
            d="M -24 2 Q 0 -6 24 2 Q 0 14 -24 2 Z"
            fill="none"
            stroke={outline}
            strokeWidth={2.5}
          />
        </g>
      );

    // Tongue visible against the ridge: L.
    case 'H':
      return (
        <g>
          <ellipse cx={0} cy={2} rx={22} ry={17} fill={DARK} />
          <path d="M -19 -8 Q 0 -15 19 -8 L 19 -5 Q 0 -11 -19 -5 Z" fill={TEETH} />
          <ellipse cx={0} cy={6} rx={11} ry={9} fill={TONGUE} />
          <ellipse cx={0} cy={2} rx={22} ry={17} fill="none" stroke={outline} strokeWidth={2.5} />
        </g>
      );

    // Idle / rest: a relaxed smile.
    case 'X':
    default:
      return (
        <g>
          <path d="M -27 -1 Q 0 14 27 -1 Q 0 6 -27 -1 Z" fill={DARK} opacity={0.85} />
          <path
            d="M -27 -1 Q 0 14 27 -1"
            fill="none"
            stroke={outline}
            strokeWidth={3}
            strokeLinecap="round"
          />
        </g>
      );
  }
};
