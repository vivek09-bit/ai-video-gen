#!/usr/bin/env python3
"""Verify container dependencies at build time."""
import numpy
import torch
import torchaudio
import torchvision
import diffusers
import transformers
import soundfile
import scipy
import vocos
from f5_tts.infer.utils_infer import infer_process
from diffusers import ModularPipeline, FluxPipeline

print(
    f"Container dependencies verified successfully: PyTorch {torch.__version__}, "
    f"Torchaudio {torchaudio.__version__}, Torchvision {torchvision.__version__}, "
    f"Diffusers {diffusers.__version__}",
    flush=True,
)
