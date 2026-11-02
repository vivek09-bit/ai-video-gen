"""VoiceStudio local TTS client and CLI utility for tube2dscript.

VoiceStudio (https://github.com/debpalash/VoiceStudio) runs locally
and provides an OpenAI-compatible speech synthesis endpoint.
By default, VoiceStudio backend listens on port 3900 (http://127.0.0.1:3900)
to avoid conflicts with standard port 8000.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import urllib.request
import urllib.error

# Auto-load .env if present
def _load_env_file():
    env_path = Path(".env")
    if env_path.exists():
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip().strip('"').strip("'")
                        if k not in os.environ:
                            os.environ[k] = v
        except Exception:
            pass

_load_env_file()

# Candidates to probe if not explicitly specified
CANDIDATE_URLS = [
    os.environ.get("VOICESTUDIO_BASE_URL", "").strip(),
    "http://127.0.0.1:3900",
    "http://localhost:3900",
    "http://127.0.0.1:8000",
    "http://localhost:8000",
]
CANDIDATE_URLS = [u.rstrip("/") for u in CANDIDATE_URLS if u]

# Default voice and model for VoiceStudio (Kokoro engine)
DEFAULT_VOICE = os.environ.get("VOICESTUDIO_VOICE", "af_heart")
DEFAULT_MODEL = os.environ.get("VOICESTUDIO_MODEL", "kokoro")


def _http_get(url: str, timeout: float = 3.0) -> Tuple[int, bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": "tube2dscript-voicestudio"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read()


def _http_post_json(url: str, payload: dict, timeout: float = 60.0) -> Tuple[int, bytes]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "tube2dscript-voicestudio",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read()


def find_active_server() -> Optional[str]:
    """Find the first responsive VoiceStudio instance from candidate endpoints."""
    # First check explicitly configured VOICESTUDIO_BASE_URL if set
    custom = os.environ.get("VOICESTUDIO_BASE_URL", "").strip().rstrip("/")
    probe_list = [custom] if custom else CANDIDATE_URLS

    for base in probe_list:
        if not base:
            continue
        # VoiceStudio has /health endpoint, and /v1/models
        for path in ["/health", "/v1/models", "/v1/audio/voices"]:
            probe_url = f"{base}{path}"
            try:
                status, _ = _http_get(probe_url, timeout=2.0)
                if status == 200:
                    return base
            except Exception:
                continue
    return None


def check_status(base_url: Optional[str] = None) -> bool:
    target = base_url or find_active_server() or CANDIDATE_URLS[0]
    print(f"[*] Testing VoiceStudio connection at: {target}")

    try:
        # Check /health or /v1/models
        active = False
        health_info = ""
        try:
            status, body = _http_get(f"{target}/health", timeout=3.0)
            if status == 200:
                active = True
                health_info = body.decode("utf-8", errors="ignore")
        except Exception:
            pass

        if not active:
            try:
                status, body = _http_get(f"{target}/v1/models", timeout=3.0)
                if status == 200:
                    active = True
            except Exception:
                pass

        if active:
            print("[+] VoiceStudio Status: ONLINE")
            print(f"[+] Active Endpoint: {target}")
            if health_info:
                print(f"[+] Health: {health_info}")

            # Try listing voices
            try:
                status, body = _http_get(f"{target}/v1/audio/voices", timeout=3.0)
                if status == 200:
                    voices = json.loads(body.decode("utf-8"))
                    print(f"[+] Available Voices: {len(voices.get('voices', voices))}")
            except Exception:
                pass
            return True
        else:
            print("[-] VoiceStudio Status: OFFLINE")
            print("[-] Could not reach VoiceStudio on candidate ports (3900, 8000).")
            print("[-] Tips:")
            print("    1. Make sure VoiceStudio app / python backend is running.")
            print("    2. VoiceStudio by default uses port 3900 (http://127.0.0.1:3900).")
            print("    3. If running on a different port, set VOICESTUDIO_BASE_URL in .env.")
            return False
    except Exception as e:
        print(f"[-] Connection Error: {e}")
        return False


def list_voices(base_url: Optional[str] = None):
    target = base_url or find_active_server() or CANDIDATE_URLS[0]
    print(f"[*] Fetching voices from {target}/v1/audio/voices ...")
    try:
        status, body = _http_get(f"{target}/v1/audio/voices", timeout=5.0)
        data = json.loads(body.decode("utf-8"))
        voices = data.get("voices", data)
        print(f"[+] Found {len(voices)} voices:")
        for v in voices:
            if isinstance(v, dict):
                v_id = v.get("id") or v.get("name") or v.get("voice_id")
                lang = v.get("language", "")
                gender = v.get("gender", "")
                print(f"  - {v_id} ({lang} {gender})")
            else:
                print(f"  - {v}")
    except Exception as e:
        print(f"[-] Failed to fetch voices: {e}")


def synthesize_speech(
    text: str,
    output_path: Path,
    voice: str = DEFAULT_VOICE,
    model: str = DEFAULT_MODEL,
    base_url: Optional[str] = None,
) -> bool:
    target = base_url or find_active_server()
    if not target:
        print("[-] Error: VoiceStudio is not reachable. Is it running?")
        return False

    endpoint = f"{target}/v1/audio/speech"
    payload = {
        "model": model,
        "input": text,
        "voice": voice,
        "response_format": "mp3",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        status, audio_bytes = _http_post_json(endpoint, payload, timeout=120.0)
        if status == 200 and len(audio_bytes) > 0:
            with open(output_path, "wb") as f:
                f.write(audio_bytes)
            print(f"[+] Saved audio to: {output_path} ({len(audio_bytes)} bytes)")
            return True
        else:
            print(f"[-] TTS synthesis returned status {status}")
            return False
    except Exception as e:
        print(f"[-] Synthesis error: {e}")
        return False


def generate_all_scenes(
    script_path: Path = Path("script.json"),
    output_dir: Path = Path("public/audio"),
    voice: str = DEFAULT_VOICE,
    base_url: Optional[str] = None,
):
    if not script_path.exists():
        print(f"[-] Script not found: {script_path}")
        return

    with open(script_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    scenes = data.get("scenes", [])
    if not scenes:
        print("[-] No scenes found in script.json")
        return

    print(f"[*] Generating audio for {len(scenes)} scenes using VoiceStudio (Voice: {voice})...")
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, scene in enumerate(scenes, start=1):
        text = scene.get("narration") or scene.get("text") or scene.get("dialogue") or ""
        if not text:
            continue
        out_file = output_dir / f"scene-{i}.mp3"
        print(f"[*] Scene {i}: {text[:50]}...")
        ok = synthesize_speech(text, out_file, voice=voice, base_url=base_url)
        if not ok:
            print(f"[-] Warning: Failed generating audio for scene {i}")
            break

    print("[+] Audio generation completed.")


def main():
    parser = argparse.ArgumentParser(description="VoiceStudio Local TTS Client")
    parser.add_argument("--check", action="store_true", help="Check VoiceStudio status")
    parser.add_argument("--voices", action="store_true", help="List available voices")
    parser.add_argument("--text", type=str, help="Single text prompt to synthesize")
    parser.add_argument("--out", type=str, default="public/audio/output.mp3", help="Output file path")
    parser.add_argument("--voice", type=str, default=DEFAULT_VOICE, help="Voice name (default: af_heart)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model name (default: kokoro)")
    parser.add_argument("--all-scenes", action="store_true", help="Synthesize audio for all scenes in script.json")
    parser.add_argument("--url", type=str, default=None, help="Custom VoiceStudio base URL")

    args = parser.parse_args()

    if args.check:
        ok = check_status(args.url)
        sys.exit(0 if ok else 1)
    elif args.voices:
        list_voices(args.url)
    elif args.text:
        ok = synthesize_speech(args.text, Path(args.out), voice=args.voice, model=args.model, base_url=args.url)
        sys.exit(0 if ok else 1)
    elif args.all_scenes:
        generate_all_scenes(voice=args.voice, base_url=args.url)
    else:
        # Default behavior: check connection
        check_status(args.url)


if __name__ == "__main__":
    main()
