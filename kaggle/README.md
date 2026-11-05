# Free Kaggle GPU (P100 / T4) for IndicF5 Voice Generation

Kaggle provides **30 hours per week of free NVIDIA GPU (Tesla P100 16GB VRAM)**. This is over **30x faster** than running IndicF5 on CPU (1.5-2 seconds per sentence vs 4-5 minutes per sentence).

---

### Step 1: Open Kaggle Notebook
1. Go to [kaggle.com/code](https://www.kaggle.com/code) and click **New Notebook**.
2. In the right sidebar:
   - **Accelerator**: Select **GPU P100** (or **GPU T4 x 2**).
   - **Internet**: Toggle to **ON** (required to download models & tunnel).

---

### Step 2: Paste and Run this single cell in Kaggle

```python
# 1. Clone repository & install requirements
!git clone https://github.com/vivek09-bit/ai-video-gen.git tube2dscript
%cd tube2dscript

# 2. Run the Kaggle GPU server & Cloudflare tunnel
!python scripts/kaggle_server.py
```

The script will automatically:
- Download `IndicF5` and reference voice models onto Kaggle's GPU.
- Launch the server on CUDA.
- Start a free Cloudflare tunnel and print your public URL:
  ```text
  Public Tunnel URL : https://xyz-sample.trycloudflare.com
  ```

---

### Step 3: Use the URL on your local PC

In your local PowerShell terminal, simply pass `--tts-url` (or set `$env:TTS_SERVER_URL`):

```powershell
.venv\Scripts\python.exe scripts\new_video.py `
  --narration script.txt `
  --topic "कलयुग के अंत के बाद आखिर क्या होगा" `
  --genre documentary `
  --format reel `
  --voice indicf5_aarav `
  --tts-url "https://xyz-sample.trycloudflare.com" `
  --no-vast `
  --images `
  --image-provider cloudflare
```

Or test a single voice sentence directly:
```powershell
.venv\Scripts\python.exe scripts\tts.py `
  --text "कलयुग के अंत के बाद सत्य युग का आरंभ होगा।" `
  --voice indicf5_aarav `
  --tts-url "https://xyz-sample.trycloudflare.com" `
  --out out/test_indic_kaggle.wav
```
