import {AbsoluteFill, interpolate, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import React from 'react';
import {flicker, hash, noise1, steppedNoise} from './noise';

/**
 * Visual effects layered over every horror shot.
 *
 * All animation is derived from the frame number, so renders are reproducible
 * and the studio preview matches the final file. Nothing here uses randomness
 * at render time.
 */

/**
 * Film grain.
 *
 * Real grain updates at the film frame rate, not the video frame rate, so the
 * tile is only stepped ~14 times a second. That reads as grain rather than TV
 * static, and it means the texture is looked up rarely rather than every frame.
 */
export const FilmGrain: React.FC<{amount: number}> = ({amount}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();

  if (amount <= 0) {
    return null;
  }

  const step = Math.floor((frame / fps) * 14);
  const offsetX = hash(step * 1.7) * 512;
  const offsetY = hash(step * 3.1) * 512;

  return (
    <AbsoluteFill
      style={{
        backgroundImage: `url(${staticFile('textures/grain.png')})`,
        backgroundRepeat: 'repeat',
        backgroundPosition: `${-offsetX}px ${-offsetY}px`,
        opacity: Math.min(0.75, amount * 0.55),
        mixBlendMode: 'overlay',
        pointerEvents: 'none',
      }}
    />
  );
};

/** Heavy darkened edges, with a slow breathing pulse. */
export const Vignette: React.FC<{strength: number; breathe?: number}> = ({
  strength,
  breathe = 0,
}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const time = frame / fps;

  if (strength <= 0) {
    return null;
  }

  const pulse = breathe ? 1 + noise1(time * 0.22) * breathe : 1;
  const inner = Math.max(12, 62 - strength * 30);
  const edge = Math.min(1, 0.42 + strength * 0.5) * pulse;

  return (
    <AbsoluteFill
      style={{
        background: `radial-gradient(ellipse at 50% 48%, rgba(0,0,0,0) ${inner}%, rgba(0,0,0,${edge.toFixed(3)}) 100%)`,
        pointerEvents: 'none',
      }}
    />
  );
};

/** Floating dust motes, catching an unseen light. */
export const Dust: React.FC<{density: number; seed?: number}> = ({density, seed = 0}) => {
  const frame = useCurrentFrame();
  const {fps, width, height} = useVideoConfig();
  const time = frame / fps;

  if (density <= 0) {
    return null;
  }

  const count = Math.round(density * 44);
  const motes = [];

  for (let i = 0; i < count; i++) {
    const key = i + seed * 977;
    const startX = hash(key * 12.9);
    const startY = hash(key * 78.2);
    const size = 1 + hash(key * 45.1) * 3.2;
    const speed = 6 + hash(key * 91.3) * 16;
    const swayAmount = 10 + hash(key * 23.7) * 34;
    const phase = hash(key * 67.1) * Math.PI * 2;
    const alpha = 0.12 + hash(key * 51.3) * 0.4;

    // Motes drift upward and wrap, so the field never empties out.
    const y = (((startY * height - time * speed) % height) + height) % height;
    const x = startX * width + Math.sin(time * 0.6 + phase) * swayAmount;

    motes.push(
      <div
        key={i}
        style={{
          position: 'absolute',
          left: x,
          top: y,
          width: size,
          height: size,
          borderRadius: '50%',
          backgroundColor: `rgba(255, 246, 224, ${alpha})`,
          filter: y < height * 0.4 ? 'blur(1px)' : undefined,
        }}
      />,
    );
  }

  return <AbsoluteFill style={{pointerEvents: 'none'}}>{motes}</AbsoluteFill>;
};

/** Falling rain, with occasional lightning. */
export const Rain: React.FC<{intensity: number; lightning?: boolean}> = ({
  intensity,
  lightning = false,
}) => {
  const frame = useCurrentFrame();
  const {fps, width, height} = useVideoConfig();
  const time = frame / fps;

  if (intensity <= 0) {
    return null;
  }

  const count = Math.round(intensity * 130);
  const drops = [];
  const angle = 0.13;

  for (let i = 0; i < count; i++) {
    const startX = hash(i * 31.7);
    const startY = hash(i * 57.3);
    const length = 24 + hash(i * 11.3) * 70;
    const speed = 1300 + hash(i * 83.9) * 1500;
    const alpha = 0.1 + hash(i * 29.1) * 0.3;

    const y = (((startY * height - time * speed) % (height + 200)) + height + 200) % (height + 200);
    const x = startX * width + y * angle;

    drops.push(
      <div
        key={i}
        style={{
          position: 'absolute',
          left: x,
          top: y - length,
          width: 1.4,
          height: length,
          background: `linear-gradient(rgba(210,230,255,0), rgba(210,230,255,${alpha}))`,
          transform: `rotate(${angle * 57.3}deg)`,
        }}
      />,
    );
  }

  // Lightning: a rare double-flash, not a steady strobe.
  let flash = 0;
  if (lightning) {
    const period = 11;
    const local = time % period;
    if (local < 0.22) {
      flash = (1 - local / 0.22) * (local < 0.1 ? 1 : 0.55);
    }
  }

  return (
    <AbsoluteFill style={{pointerEvents: 'none'}}>
      {drops}
      {flash > 0 ? (
        <AbsoluteFill style={{backgroundColor: `rgba(198, 216, 255, ${flash * 0.45})`}} />
      ) : null}
    </AbsoluteFill>
  );
};

/**
 * A light source that behaves like a failing bulb or a guttering flame.
 * Renders as a radial glow; `cold` switches from candlelight to fluorescent.
 */
export const FlickeringLight: React.FC<{
  amount: number;
  x: number;
  y: number;
  radius: number;
  cold?: boolean;
  brightness?: number;
}> = ({amount, x, y, radius, cold = false, brightness = 1}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const time = frame / fps;

  const level = amount > 0 ? flicker(time, 1 - amount * 0.7) : 1;
  const colour = cold ? '190, 214, 255' : '255, 196, 118';
  const strength = level * brightness;

  return (
    <AbsoluteFill
      style={{
        background: `radial-gradient(circle ${radius}px at ${x * 100}% ${y * 100}%, rgba(${colour}, ${(0.5 * strength).toFixed(3)}), rgba(${colour}, ${(0.13 * strength).toFixed(3)}) 45%, rgba(0,0,0,0) 72%)`,
        mixBlendMode: 'screen',
        pointerEvents: 'none',
      }}
    />
  );
};

/** Overall exposure dip, for a bulb cutting out or the picture dying. */
export const ExposureFlicker: React.FC<{amount: number; floor?: number}> = ({
  amount,
  floor = 0.55,
}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const time = frame / fps;

  if (amount <= 0) {
    return null;
  }

  const level = flicker(time, floor);
  const darkness = (1 - level) * amount;

  return (
    <AbsoluteFill
      style={{
        backgroundColor: `rgba(0, 0, 0, ${Math.min(0.92, darkness).toFixed(3)})`,
        pointerEvents: 'none',
      }}
    />
  );
};

/**
 * Camera shake. Wraps a shot so the whole picture jolts, which is what sells a
 * scare beat. Two noise channels keep the motion from looking like a sine wave.
 */
export const Shake: React.FC<{amount: number; children: React.ReactNode}> = ({
  amount,
  children,
}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const time = frame / fps;

  if (amount <= 0) {
    return <>{children}</>;
  }

  const magnitude = amount * 22;
  const x = noise1(time * 9.5) * magnitude + steppedNoise(time, 1 / 30, 1) * magnitude * 0.35;
  const y = noise1(time * 8.3 + 41) * magnitude * 0.7;
  const rotation = noise1(time * 7.1 + 17) * amount * 0.9;

  return (
    <AbsoluteFill
      style={{
        transform: `translate(${x.toFixed(2)}px, ${y.toFixed(2)}px) rotate(${rotation.toFixed(3)}deg)`,
      }}
    >
      {children}
    </AbsoluteFill>
  );
};

/** Slow parallax-free drift and breathing zoom, so nothing ever sits perfectly still. */
export const LivingFrame: React.FC<{
  drift?: 'in' | 'out' | 'left' | 'right' | 'none';
  zoom?: number;
  children: React.ReactNode;
}> = ({drift = 'in', zoom = 0.03, children}) => {
  const frame = useCurrentFrame();
  const {fps, durationInFrames} = useVideoConfig();
  const time = frame / fps;
  const progress = durationInFrames <= 1 ? 0 : frame / (durationInFrames - 1);

  const breathe = 1 + noise1(time * 0.13) * zoom;
  let scale = breathe;
  let x = 0;

  switch (drift) {
    case 'in':
      scale = breathe * (1 + progress * zoom * 2.2);
      break;
    case 'out':
      scale = breathe * (1.06 - progress * zoom * 2.2);
      break;
    case 'left':
      scale = breathe * 1.05;
      x = (0.5 - progress) * 2.2 * (zoom * 40);
      break;
    case 'right':
      scale = breathe * 1.05;
      x = (progress - 0.5) * 2.2 * (zoom * 40);
      break;
    case 'none':
      break;
  }

  return (
    <AbsoluteFill
      style={{
        transform: `translateX(${x.toFixed(2)}px) scale(${scale.toFixed(4)})`,
      }}
    >
      {children}
    </AbsoluteFill>
  );
};

/**
 * Opening Visual Hook Intro Effect:
 * - Subtle initial atmospheric opening flash (clean white/glow without red tint)
 */
export const HookIntroEffect: React.FC = () => {
  return null;
};

/**
 * Pattern Interrupts
 */
export const PatternInterruptEffect: React.FC = () => {
  return null;
};
