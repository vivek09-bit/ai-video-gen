import {Config} from '@remotion/cli/config';

// JPEG frames are dramatically faster to encode for long-form videos where
// backgrounds are photos or gradients. Switch to 'png' if you need alpha.
Config.setVideoImageFormat('jpeg');
Config.setOverwriteOutput(true);
Config.setCodec('h264');
Config.setCrf(18);

// Long-form renders are frame-count bound; leaving headroom keeps the machine
// usable while a 30-60 minute video encodes.
Config.setConcurrency(4);
