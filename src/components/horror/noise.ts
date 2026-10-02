/**
 * Deterministic noise helpers.
 *
 * Every effect in the horror renderer is a pure function of the frame number,
 * so a render is reproducible and the studio preview matches the final file.
 * Nothing here uses Math.random.
 */

/** Hash of an integer into [0, 1). Stable across runs and platforms. */
export const hash = (n: number): number => {
  const value = Math.sin(n * 127.1) * 43758.5453123;
  return value - Math.floor(value);
};

const smooth = (t: number): number => t * t * (3 - 2 * t);

/** Smooth 1D value noise in [-1, 1]. Good for drift and camera sway. */
export const noise1 = (x: number): number => {
  const i = Math.floor(x);
  const f = x - i;
  const u = smooth(f);
  return (hash(i) * (1 - u) + hash(i + 1) * u) * 2 - 1;
};

/** Fractal noise: several octaves, for a more organic result than one layer. */
export const fbm = (x: number, octaves = 3): number => {
  let total = 0;
  let amplitude = 1;
  let frequency = 1;
  let norm = 0;
  for (let o = 0; o < octaves; o++) {
    total += noise1(x * frequency + o * 37.3) * amplitude;
    norm += amplitude;
    amplitude *= 0.5;
    frequency *= 2.07;
  }
  return total / norm;
};

/**
 * Light flicker in [floor, 1].
 *
 * Two noise layers give both a slow sizzle and fast sputtering, plus rare deep
 * dips that read as a failing bulb rather than a smooth sine wave.
 */
export const flicker = (time: number, floor = 0.55): number => {
  const slow = fbm(time * 1.6, 2) * 0.5 + 0.5;
  const fast = noise1(time * 17.0) * 0.5 + 0.5;
  const dip = noise1(time * 0.7) > 0.86 ? 0.4 : 1;
  const value = slow * 0.62 + fast * 0.38;
  return Math.max(floor, Math.min(1, value * dip));
};

/** Stepped noise that changes every `interval` seconds. Used for film grain. */
export const steppedNoise = (time: number, interval: number, channel = 0): number =>
  hash(Math.floor(time / interval) * 31.7 + channel * 71.3) * 2 - 1;
