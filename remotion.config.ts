import {Config} from '@remotion/cli/config';

// JPEG frames are dramatically faster to encode for long-form videos where
// backgrounds are photos or gradients. Switch to 'png' if you need alpha.
Config.setVideoImageFormat('jpeg');
Config.setOverwriteOutput(true);
Config.setCodec('h264');
Config.setCrf(18);

// 1x concurrency is safest on Windows to prevent Chromium stack buffer overruns
// (0xC0000409 / exit 3221226505) when loading multi-track audio and high-res canvas.
const defaultConcurrency = process.platform === 'win32' ? 1 : 2;
const concurrency = Number(
  process.env.REMOTION_CONCURRENCY || process.env.CONCURRENCY || defaultConcurrency
);
Config.setConcurrency(concurrency);
