import {AbsoluteFill, interpolate, spring, useCurrentFrame, useVideoConfig} from 'remotion';
import React from 'react';
import {Background} from './Background';
import {Captions} from './Captions';
import {CaptionOptions, Scene, SceneTiming, Theme, Word} from '../types';

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

export const SceneView: React.FC<{
  scene: Scene;
  timing: SceneTiming;
  durationInFrames: number;
  theme: Theme;
  captions: CaptionOptions;
}> = ({scene, timing, durationInFrames, theme, captions}) => {
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

  return (
    <AbsoluteFill style={{backgroundColor: '#000'}}>
      <Background spec={scene.background} durationInFrames={durationInFrames} />

      {/* Scrim: keeps text legible over photos and video without washing them out. */}
      <AbsoluteFill
        style={{
          backgroundImage: hasBody
            ? 'linear-gradient(180deg, rgba(0,0,0,0.72) 0%, rgba(0,0,0,0.28) 38%, rgba(0,0,0,0.82) 100%)'
            : 'linear-gradient(180deg, rgba(0,0,0,0.55) 0%, rgba(0,0,0,0.15) 40%, rgba(0,0,0,0.75) 100%)',
        }}
      />

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
    </AbsoluteFill>
  );
};
