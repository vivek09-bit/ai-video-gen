import {AbsoluteFill, interpolate, spring, useCurrentFrame, useVideoConfig} from 'remotion';
import React from 'react';
import {Background} from './Background';
import {Captions} from './Captions';
import {Character} from './character/Character';
import {MouthShape} from './character/Mouth';
import {ShotSequence} from './horror/ShotSequence';
import {HookIntroEffect, PatternInterruptEffect} from './horror/Effects';
import {
  CaptionOptions,
  CharacterSpec,
  LineTiming,
  Scene,
  SceneTiming,
  Theme,
  Word,
} from '../types';

/** Shrinks long headlines so they always fit on two lines. */
const headlineSize = (text: string, base: number): number => {
  const len = text.trim().length;
  if (len <= 24) {
    return base;
  }
  return Math.max(Math.round(base * 0.52), Math.round(base * Math.pow(24 / len, 0.55)));
};

const JUSTIFY: Record<Scene['headlinePosition'], string> = {
  top: 'flex-start',
  center: 'center',
  bottom: 'flex-end',
};

/** Fallback mouth cycle for when lip-sync data is unavailable. */
const FALLBACK_MOUTHS: MouthShape[] = ['C', 'D', 'E', 'C', 'B', 'D'];

/**
 * Works out what a character's mouth should be doing right now.
 *
 * Only the character who owns the currently playing line speaks; everyone else
 * rests. If Rhubarb produced no cues (for example `--no-lipsync` was used) the
 * mouth still moves, cycling shapes off the clock so the scene is not static.
 */
const mouthAt = (
  characterId: string | null,
  time: number,
  lines: LineTiming[],
): {shape: MouthShape; talking: boolean} => {
  for (const line of lines) {
    if (time < line.start || time >= line.start + line.duration) {
      continue;
    }
    if (line.characterId !== characterId) {
      return {shape: 'X', talking: false};
    }

    const local = time - line.start;
    if (line.mouthCues.length === 0) {
      const index = Math.floor(local * 11) % FALLBACK_MOUTHS.length;
      return {shape: FALLBACK_MOUTHS[index], talking: true};
    }

    const cue = line.mouthCues.find((c) => local >= c.start && local < c.end);
    return {shape: cue?.value ?? 'X', talking: true};
  }
  return {shape: 'X', talking: false};
};

export const SceneView: React.FC<{
  scene: Scene;
  timing: SceneTiming;
  durationInFrames: number;
  theme: Theme;
  captions: CaptionOptions;
  characters: Record<string, CharacterSpec>;
  isFirstScene?: boolean;
}> = ({
  scene,
  timing,
  durationInFrames,
  theme,
  captions,
  characters,
  isFirstScene = false,
}) => {
  const frame = useCurrentFrame();
  const {fps, width, height} = useVideoConfig();
  const time = timing.start + frame / fps;

  const words: Word[] = timing.words.map((w) => ({
    text: w.text,
    start: timing.audioStart + w.start,
    end: timing.audioStart + w.end,
  }));

  const enter = spring({
    frame,
    fps,
    config: {damping: 200, mass: 0.6},
    durationInFrames: 18,
  });

  const hasBody = scene.subtext.length > 0 || scene.bullets.length > 0;
  const headlinePx = headlineSize(scene.headline, Math.round(height * 0.088));
  const usingShots = (scene.shots?.length ?? 0) > 0;

  return (
    <AbsoluteFill style={{backgroundColor: '#000'}}>
      {/* Procedural horror shots replace the single background when present. */}
      {usingShots ? (
        <ShotSequence shots={scene.shots} />
      ) : (
        <Background spec={scene.background} durationInFrames={durationInFrames} />
      )}

      {/* Scrim: keeps text legible over photos and video without washing them out.
          Horror shots are already dark and graded, so they need far less. */}
      <AbsoluteFill
        style={{
          backgroundImage: usingShots
            ? 'linear-gradient(180deg, rgba(0,0,0,0.55) 0%, rgba(0,0,0,0.05) 45%, rgba(0,0,0,0.7) 100%)'
            : hasBody
              ? 'linear-gradient(180deg, rgba(0,0,0,0.72) 0%, rgba(0,0,0,0.28) 38%, rgba(0,0,0,0.82) 100%)'
              : 'linear-gradient(180deg, rgba(0,0,0,0.55) 0%, rgba(0,0,0,0.15) 40%, rgba(0,0,0,0.75) 100%)',
        }}
      />

      {/* Cast stands above the background but behind the text, so headlines and
          bullets stay legible even when a character overlaps them. */}
      {scene.cast.map((member) => {
        const spec = characters?.[member.id];
        if (!spec) {
          return null;
        }

        const {shape, talking} = mouthAt(member.id, time, timing.lines);
        const charHeight = height * 0.88 * (member.scale ?? 1);
        const charWidth = charHeight * 0.625;
        const left = member.x * width - charWidth / 2;
        const acting = member.acting ?? scene.acting;

        return (
          <React.Fragment key={member.id}>
            {/* A soft pool of light behind the figure. Horror stages are almost
                black, and an unlit character simply disappears into them. */}
            <div
              style={{
                position: 'absolute',
                left: left - charWidth * 0.35,
                top: height - charHeight - charHeight * 0.1,
                width: charWidth * 1.7,
                height: charHeight * 1.15,
                background: `radial-gradient(ellipse at 50% 55%, rgba(150, 176, 210, 0.20), rgba(120, 145, 180, 0.07) 45%, rgba(0,0,0,0) 72%)`,
                mixBlendMode: 'screen',
                pointerEvents: 'none',
              }}
            />
            <div
              style={{
                position: 'absolute',
                left,
                top: height - charHeight,
                width: charWidth,
                height: charHeight,
                transform: member.flip ? 'scaleX(-1)' : undefined,
                filter: 'drop-shadow(0 0 26px rgba(0,0,0,0.75))',
              }}
            >
              <Character spec={spec} mouth={shape} talking={talking} acting={acting} />
            </div>
          </React.Fragment>
        );
      })}

      <AbsoluteFill
        style={{
          display: 'flex',
          flexDirection: 'column',
          justifyContent: JUSTIFY[scene.headlinePosition],
          alignItems: 'center',
          textAlign: 'center',
          padding: `${Math.round(height * 0.1)}px ${Math.round(width * 0.1)}px`,
          paddingBottom: `${Math.round(height * 0.34)}px`,
        }}
      >
        {scene.headline ? (
          <div
            style={{
              opacity: interpolate(enter, [0, 1], [0, 1]),
              transform: `translateY(${interpolate(enter, [0, 1], [26, 0])}px)`,
            }}
          >
            <div
              style={{
                fontFamily: theme.headlineFont,
                fontSize: headlinePx,
                lineHeight: 1.12,
                fontWeight: 800,
                letterSpacing: -1.2,
                color: theme.text,
                textShadow: '0 4px 30px rgba(0,0,0,0.8)',
              }}
            >
              {scene.headline}
            </div>
            <div
              style={{
                width: Math.round(headlinePx * 1.6),
                height: 6,
                marginTop: Math.round(headlinePx * 0.34),
                marginLeft: 'auto',
                marginRight: 'auto',
                borderRadius: 3,
                backgroundColor: theme.accent,
                transform: `scaleX(${interpolate(enter, [0, 1], [0, 1])})`,
              }}
            />
          </div>
        ) : null}

        {scene.subtext ? (
          <div
            style={{
              marginTop: Math.round(height * 0.028),
              maxWidth: Math.round(width * 0.66),
              fontFamily: theme.bodyFont,
              fontSize: Math.round(height * 0.032),
              lineHeight: 1.45,
              fontWeight: 400,
              color: theme.muted,
              opacity: interpolate(enter, [0, 1], [0, 1]),
            }}
          >
            {scene.subtext}
          </div>
        ) : null}

        {scene.bullets.length > 0 ? (
          <div
            style={{
              marginTop: Math.round(height * 0.04),
              display: 'flex',
              flexDirection: 'column',
              gap: Math.round(height * 0.02),
              maxWidth: Math.round(width * 0.64),
            }}
          >
            {scene.bullets.map((bullet, i) => {
              const delayed = spring({
                frame: Math.max(0, frame - 12 - i * 5),
                fps,
                config: {damping: 200, mass: 0.6},
                durationInFrames: 16,
              });
              return (
                <div
                  key={`${bullet}-${i}`}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: Math.round(height * 0.018),
                    padding: `${Math.round(height * 0.014)}px ${Math.round(height * 0.026)}px`,
                    borderRadius: 14,
                    backgroundColor: theme.panel,
                    backdropFilter: 'blur(8px)',
                    fontFamily: theme.bodyFont,
                    fontSize: Math.round(height * 0.03),
                    fontWeight: 600,
                    color: theme.text,
                    textAlign: 'left',
                    opacity: interpolate(delayed, [0, 1], [0, 1]),
                    transform: `translateX(${interpolate(delayed, [0, 1], [-24, 0])}px)`,
                  }}
                >
                  <span
                    style={{
                      width: Math.round(height * 0.014),
                      height: Math.round(height * 0.014),
                      borderRadius: 999,
                      backgroundColor: theme.accent,
                      flexShrink: 0,
                    }}
                  />
                  {bullet}
                </div>
              );
            })}
          </div>
        ) : null}
      </AbsoluteFill>

      <Captions words={words} time={time} options={captions} theme={theme} />
      <PatternInterruptEffect />
      {isFirstScene ? <HookIntroEffect /> : null}
    </AbsoluteFill>
  );
};
