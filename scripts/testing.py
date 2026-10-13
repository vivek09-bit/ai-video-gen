import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"      # skips torch.compile, which breaks on Windows CPU
os.environ["PATH"] += r";C:\ffmpeg"

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModel

torch.set_num_threads(os.cpu_count())

model = AutoModel.from_pretrained(
    "ai4bharat/indic-parler-tts",
    trust_remote_code=True,
    low_cpu_mem_usage=False,
)

audio = model(
    "चार हज़ार बार मैंने उससे बोला है कि तू ही भगवान है लेकिन वो है कि बस मेहनत ही करे जा रहा है।",
    ref_audio_path="ref.wav",
    ref_text="EXACT Devanagari transcript of ref.wav goes here",
)

if audio.dtype == np.int16:
    audio = audio.astype(np.float32) / 32768.0

sf.write("narration.wav", np.array(audio, dtype=np.float32), samplerate=24000)