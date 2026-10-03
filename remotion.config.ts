import {Config} from '@remotion/cli/config';

// JPEG frames are dramatically faster to encode for long-form videos where
// backgrounds are photos or gradients. Switch to 'png' if you need alpha.
Config.setVideoImageFormat('jpeg');
Config.setOverwriteOutput(true);
Config.setCodec('h264');
Config.setCrf(18);

// 2x concurrency is stable across high-res portrait (9:16) & landscape renders
// on Windows, preventing browser process stack buffer overruns / OOM crashes.
const concurrency = Number(
  process.env.REMOTION_CONCURRENCY || process.env.CONCURRENCY || 2
);
Config.setConcurrency(concurrency);
