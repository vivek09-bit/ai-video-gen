import numpy as np, soundfile as sf
from pathlib import Path
from transformers import AutoModel

model = AutoModel.from_pretrained("ai4bharat/IndicF5", trust_remote_code=True)

def speak(voices, text, out="output.wav"):
    d = Path("examples") / voices
    audio = model(
        text,
        ref_audio_path=str(d / "ref.wav"),
        ref_text=(d / "ref.txt").read_text(encoding="utf-8").strip(),
    )
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    sf.write(out, np.array(audio, dtype=np.float32), samplerate=24000)

speak("priya", "नमस्ते, मैं प्रिया बोल रही हूँ।", "priya.wav")
speak("kabir", "नमस्ते, मैं कबीर बोल रहा हूँ।", "kabir.wav")