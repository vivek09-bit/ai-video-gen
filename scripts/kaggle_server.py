#!/usr/bin/env python3
"""Kaggle GPU Tunnel Server for IndicF5 & Indic-Parler Voice Generation.

Run this script inside a free Kaggle Notebook with:
- Accelerator: GPU P100 (or GPU T4 x 2)
- Internet: ON

It starts the IndicF5 and Indic-Parler TTS server on the Kaggle GPU and creates a free
Cloudflare Tunnel to give you a public URL (e.g. https://xxx.trycloudflare.com).
You then use this URL on your local PC:
  .venv\\Scripts\\python.exe scripts\\new_video.py --voice indicf5_aarav --tts-url <URL>
  # or
  .venv\\Scripts\\python.exe scripts\\new_video.py --voice indic_parler_aarav --tts-url <URL>
"""

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

PORT = int(os.environ.get("PORT", 8000))

def setup_environment(model: str = "all"):
    print("[0/4] Syncing latest tube2dscript repository code...")
    try:
        subprocess.run(["git", "pull", "origin", "main"], check=False)
    except Exception:
        pass

    print("[1/4] Checking & installing required packages...")
    packages = ["soundfile", "transformers", "vocos", "librosa", "pydub"]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *packages], check=False)

    if model in ("all", "indicf5"):
        print("  -> Installing AI4Bharat IndicF5 TTS...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/AI4Bharat/IndicF5.git"], check=False)

    if model in ("all", "parler"):
        print("  -> Installing AI4Bharat Indic Parler-TTS...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/huggingface/parler-tts.git"], check=False)

    print("[2/4] Setting up Cloudflare tunnel (cloudflared)...")
    cloudflared = Path("cloudflared")
    if not cloudflared.exists() and not shutil.which("cloudflared"):
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
        subprocess.run(["wget", "-q", "-nc", url, "-O", "cloudflared"], check=False)
        if cloudflared.exists():
            subprocess.run(["chmod", "+x", "cloudflared"], check=False)

def start_server_and_tunnel(args=None):
    if args is None:
        import argparse
        parser = argparse.ArgumentParser(description="Kaggle TTS Server (IndicF5 & Indic-Parler)")
        parser.add_argument("--hf-token", help="Hugging Face API token for ai4bharat/IndicF5")
        parser.add_argument("--port", type=int, default=PORT, help="Server port")
        parser.add_argument("--model", choices=["all", "indicf5", "parler"], default="all", help="TTS engines to support (default: all)")
        args, _ = parser.parse_known_args()

    port = args.port or PORT
    model_mode = getattr(args, "model", "all")
    print(f"[3/4] Launching TTS server (mode: {model_mode}) on port {port}...")
    server_cmd = [sys.executable, "scripts/vast_server.py"]
    server_env = dict(
        os.environ,
        PORT=str(port),
        TORCH_COMPILE_DISABLE="1",
        INDICF5_SPEED="0.82",
    )

    # Read Hugging Face token from CLI, env, Kaggle Secrets, or local .env
    if args.hf_token:
        server_env["HF_TOKEN"] = args.hf_token
    elif not server_env.get("HF_TOKEN") and not server_env.get("HUGGINGFACE_API_KEY"):
        try:
            from kaggle_secrets import UserSecretsClient
            secrets = UserSecretsClient()
            hf_tok = secrets.get_secret("HF_TOKEN") or secrets.get_secret("HUGGINGFACE_API_KEY")
            if hf_tok:
                server_env["HF_TOKEN"] = hf_tok
                print("[Kaggle] Loaded Hugging Face token from Kaggle Secrets.")
        except Exception:
            pass
        if not server_env.get("HF_TOKEN"):
            for env_name in (".env", "../.env"):
                p = Path(env_name)
                if p.is_file():
                    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
                        line = line.strip()
                        if line.startswith("HF_TOKEN=") or line.startswith("HUGGINGFACE_API_KEY="):
                            server_env["HF_TOKEN"] = line.split("=", 1)[1].strip().strip('"').strip("'")
                            break
                    if server_env.get("HF_TOKEN"):
                        break

    server_env.setdefault("INDICF5_MODEL", "ai4bharat/IndicF5")
    server_env.setdefault("INDIC_PARLER_MODEL", "RXD03/indic-parler-tts")
    server_proc = subprocess.Popen(server_cmd, env=server_env)

    time.sleep(3)
    if server_proc.poll() is not None:
        print("[Error] Failed to start vast_server.py! Check error output above.", file=sys.stderr)
        return

    print("[4/4] Establishing Cloudflare public tunnel...")
    bin_path = "./cloudflared" if Path("cloudflared").exists() else "cloudflared"
    tunnel_proc = subprocess.Popen(
        [bin_path, "tunnel", "--url", f"http://127.0.0.1:{PORT}"],
        stderr=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )

    tunnel_url = None
    deadline = time.time() + 45
    while time.time() < deadline:
        line = tunnel_proc.stderr.readline()
        if not line:
            continue
        if "trycloudflare.com" in line:
            match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
            if match:
                tunnel_url = match.group(0)
                break

    if not tunnel_url:
        print("[Error] Could not retrieve trycloudflare.com tunnel URL.", file=sys.stderr)
        return

    print("\n" + "=" * 70)
    print("  KAGGLE NVIDIA GPU TTS SERVER IS LIVE! (IndicF5 & Parler-TTS)")
    print("=" * 70)
    print(f"  Public Tunnel URL : {tunnel_url}")
    print("\n  Run on your local PC:")
    print("  # Option 1: IndicF5 voice (zero-shot, 0.82x Sanskritized Hindi)")
    print(f"  .venv\\Scripts\\python.exe scripts\\new_video.py `")
    print(f"    --narration script.txt `")
    print(f"    --topic \"Your Topic\" `")
    print(f"    --genre documentary `")
    print(f"    --format reel `")
    print(f"    --voice indicf5_aarav `")
    print(f"    --tts-url \"{tunnel_url}\" `")
    print(f"    --no-vast `")
    print(f"    --images `")
    print(f"    --image-provider cloudflare")
    print(f"\n  # Option 2: Indic Parler voice (natural description-conditioned)")
    print(f"  .venv\\Scripts\\python.exe scripts\\new_video.py `")
    print(f"    --narration script.txt `")
    print(f"    --topic \"Your Topic\" `")
    print(f"    --genre documentary `")
    print(f"    --format reel `")
    print(f"    --voice indic_parler_aarav `")
    print(f"    --tts-url \"{tunnel_url}\" `")
    print(f"    --no-vast `")
    print(f"    --images `")
    print(f"    --image-provider cloudflare")
    print("=" * 70 + "\n")

    try:
        while True:
            time.sleep(1)
            if server_proc.poll() is not None:
                print("Server exited unexpectedly.")
                break
    except KeyboardInterrupt:
        print("\nShutting down Kaggle server & tunnel...")
    finally:
        server_proc.terminate()
        tunnel_proc.terminate()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Kaggle TTS Server (IndicF5 & Indic-Parler)")
    parser.add_argument("--hf-token", help="Hugging Face API token for ai4bharat/IndicF5")
    parser.add_argument("--port", type=int, default=PORT, help="Server port")
    parser.add_argument("--model", choices=["all", "indicf5", "parler"], default="all", help="TTS engines to support (default: all)")
    args, _ = parser.parse_known_args()

    setup_environment(model=args.model)
    start_server_and_tunnel(args)

if __name__ == "__main__":
    main()
