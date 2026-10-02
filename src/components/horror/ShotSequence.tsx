import {AbsoluteFill, Sequence, interpolate, useCurrentFrame, useVideoConfig} from 'remotion';
import React from 'react';
import {Dust, ExposureFlicker, FilmGrain, LivingFrame, Shake, Vignette} from './Effects';
import {Stage} from './Stage';
import {ShotEffects, ShotSpec} from '../../types';

/**
 * Turns a list of weighted shots into a sequence across the scene.
 *
 * Shots carry relative weights rather than absolute durations, so narration
 * length stays authoritative: change the script and the shots automatically
 * re-tile, with no chance of drifting out of sync with the audio.
 *
 * Transitions work by overlap. A crossfade keeps the outgoing shot on screen and
 * fades the incoming one in on top of it, which composites to a true crossfade.
 * Fading the outgoing shot out as well would darken the middle of the
 * transition, because both layers would be partly transparent at once.
 */

const DEFAULT_EFFECTS: Required<ShotEffects> = {
  vignette: 0.78,
  grain: 0.4,
  dust: 0.22,
  shake: 0,
  flicker: 0,
  exposureFlicker: 0,
  zoom: 0.03,
  drift: 'in',
};

/** Budget for a shot, in frames. */
interface Slot {
  shot: ShotSpec;
  from: number;
  to: number;
  fadeIn: number;
}

const buildSlots = (shots: ShotSpec[], totalFrames: number, fps: number): Slot[] => {
  const weights = shots.map((s) => Math.max(0.01, s.weight ?? 1));
  const total = weights.reduce((sum, w) => sum + w, 0);

  const starts: number[] = [];
  const lengths: number[] = [];
  let cursor = 0;

  for (let i = 0; i < shots.length; i++) {
    const length = i === shots.length - 1 ? totalFrames - cursor : Math.round((weights[i] / total) * totalFrames);
    starts.push(cursor);
    lengths.push(length);
    cursor += length;
  }

  return shots.map((shot, i) => {
    const type = shot.transition?.type ?? 'cut';
    const seconds = shot.transition?.seconds ?? 0.9;
    const fadeIn = i > 0 && type !== 'cut' ? Math.round(seconds * fps) : 0;

    return {
      shot,
      from: starts[i],
      to: starts[i] + lengths[i],
      fadeIn,
    };
  });
};

/** Pushes the outgoing shot past its boundary when the next one cross-fades in. */
const extendForCrossfade = (shots: ShotSpec[], index: number, fps: number): number => {
  const next = shots[index + 1];
  if (!next) {
    return 0;
  }
  const type = next.transition?.type ?? 'cut';
  if (type !== 'crossfade') {
    return 0;
  }
  const seconds = next.transition?.seconds ?? 0.9;
  return Math.round(seconds * fps);
};

const ShotContent: React.FC<{shot: ShotSpec; fadeIn: number}> = ({shot, fadeIn}) => {
  const frame = useCurrentFrame();
  const {durationInFrames} = useVideoConfig();
  const effects = {...DEFAULT_EFFECTS, ...(shot.effects ?? {})};

  const opacity =
    fadeIn > 0
      ? interpolate(frame, [0, fadeIn], [0, 1], {extrapolateRight: 'clamp'})
      : 1;

  return (
    <AbsoluteFill style={{opacity, backgroundColor: '#000'}}>
      <Shake amount={Math.max(0, effects.shake)}>
        <LivingFrame drift={effects.drift} zoom={Math.max(0, effects.zoom)}>
          <Stage spec={shot.visual} />
        </LivingFrame>
      </Shake>

      <ExposureFlicker amount={Math.max(0, effects.exposureFlicker)} />
      <Dust density={Math.max(0, effects.dust)} />
      <Vignette strength={Math.max(0, effects.vignette)} breathe={0.06} />
      <FilmGrain amount={Math.max(0, effects.grain)} />
      <span style={{display: 'none'}}>{frame + durationInFrames}</span>
    </AbsoluteFill>
  );
};

export const ShotSequence: React.FC<{shots: ShotSpec[]}> = ({shots}) => {
  const {fps, durationInFrames} = useVideoConfig();

  if (shots.length === 0) {
    return null;
  }

  if (shots.length === 1) {
    return <ShotContent shot={shots[0]} fadeIn={0} />;
  }

  const slots = buildSlots(shots, durationInFrames, fps);

  return (
    <>
      {slots.map((slot, i) => {
        const extension = extendForCrossfade(shots, i, fps);
        const from = Math.max(0, slot.from);
        const to = Math.min(durationInFrames, slot.to + extension);
        const length = Math.max(1, to - from);

        return (
          <Sequence key={i} from={from} durationInFrames={length} name={slot.shot.visual.type}>
            <ShotContent shot={slot.shot} fadeIn={slot.fadeIn} />
          </Sequence>
        );
      })}
    </>
  );
};
