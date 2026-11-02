#!/usr/bin/env python3
"""Prefetch model weights during Docker build with resume and filtered downloads."""
import os
import sys
import time
from huggingface_hub import snapshot_download

# CRITICAL: Disable XetHub CAS to prevent 401 Unauthorized errors on long downloads.
# Hugging Face CAS session tokens expire after 1 hour, causing failures on large models.
# Disabling Xet forces direct HTTP chunking from Cloudflare/S3 CDN with Bearer auth.
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

token = (os.environ.get("HF_TOKEN") or "").strip().strip("'\"") or None
target = sys.argv[1].lower() if len(sys.argv) > 1 else "all"


def download_with_retry(repo_id: str, allow_patterns=None, ignore_patterns=None, max_attempts=5):
    print(f"\n[Docker build] Prefetching {repo_id} (target: {target})...", flush=True)
    for attempt in range(1, max_attempts + 1):
        try:
            snapshot_download(
                repo_id=repo_id,
                token=token,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                max_workers=4,
                resume_download=True,
            )
            print(f"[Docker build] Successfully prefetched {repo_id}!", flush=True)
            return
        except Exception as err:
            print(
                f"[Docker build Warning] Attempt {attempt}/{max_attempts} failed for {repo_id}: {err}",
                flush=True,
            )
            if attempt == max_attempts:
                raise
            sleep_sec = 5 * attempt
            print(f"[Docker build] Retrying in {sleep_sec}s...", flush=True)
            time.sleep(sleep_sec)


# 1. IndicF5 (~1.4 GB)
if target in ("all", "indicf5"):
    download_with_retry(
        "ai4bharat/IndicF5",
        ignore_patterns=["*.bin", "*.onnx", "*.msgpack", "*.md", "*.git*"],
    )

# 2. FLUX.3 Image Klein 5.6B BF16 (~20.3 GB Diffusers layout)
# Skips 23+ GB of duplicate ComfyUI single-file weights (model.safetensors) and fp8/nvfp4 variants!
if target in ("all", "flux", "flux3"):
    download_with_retry(
        "black-forest-labs/flux-3-image-klein",
        allow_patterns=[
            "*.json",
            "*.txt",
            "transformer/*",
            "text_encoder/*",
            "vae/*",
        ],
    )

print(f"[Docker build] Prefetch step for '{target}' completed successfully.", flush=True)
