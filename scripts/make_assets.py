#!/usr/bin/env python3
"""Generate the textures and audio beds the horror renderer needs.

Everything here is synthesised from scratch, so the project needs no downloaded
assets and no third-party licences: no stock audio, no sample packs, nothing to
attribute. It also means the beds are perfectly loopable and the grain is
identical on every machine.

Outputs
  public/textures/grain.png      film-grain tile, tiled over every shot
  public/audio/amb_*.wav         seamless ambience beds (30 s loops)
  public/audio/sfx_*.wav         one-shot stingers, impacts and risers

Run with:  .venv\\Scripts\\python.exe scripts\\make_assets.py
Re-running is cheap and safe; pass --force to regenerate existing files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

ROOT = Path(__file__).resolve().parent.parent
TEXTURES = ROOT / "public" / "textures"
AUDIO = ROOT / "public" / "audio"

SR = 24000
LOOP_SECONDS = 30.0


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def to_loopable(x: np.ndarray, fade_seconds: float = 2.0) -> np.ndarray:
    """Crossfade the tail of a signal onto its head so it loops seamlessly.

    Without this every loop point clicks, which is very audible on a low drone.
    """
    fade = int(fade_seconds * SR)
    if fade <= 0 or len(x) <= 2 * fade:
        return x

    target = len(x) - fade
    head = x[:fade]
    tail = x[-fade:]
    ramp = np.linspace(0.0, 1.0, fade)
    if x.ndim == 2:
        # Stereo data needs the ramp on its own axis to broadcast correctly.
        ramp = ramp[:, None]

    blended = tail * (1 - ramp) + head * ramp
    out = x[:target].copy()
    out[:fade] = blended
    return out


def normalise(x: np.ndarray, peak: float = 0.9) -> np.ndarray:
    highest = float(np.max(np.abs(x))) or 1.0
    return x * (peak / highest)


def lowpass(x: np.ndarray, cutoff: float, order: int = 2) -> np.ndarray:
    b, a = signal.butter(order, cutoff / (SR / 2), btype="low")
    return signal.lfilter(b, a, x)


def highpass(x: np.ndarray, cutoff: float, order: int = 2) -> np.ndarray:
    b, a = signal.butter(order, cutoff / (SR / 2), btype="high")
    return signal.lfilter(b, a, x)


def bandpass(x: np.ndarray, low: float, high: float, order: int = 2) -> np.ndarray:
    b, a = signal.butter(order, [low / (SR / 2), high / (SR / 2)], btype="band")
    return signal.lfilter(b, a, x)


def slow_lfo(n: int, period_seconds: float, phase: float = 0.0, depth: float = 1.0) -> np.ndarray:
    """A slow sine in [1 - depth, 1 + depth], used to breathe life into a bed."""
    t = np.arange(n) / SR
    return 1.0 + depth * np.sin(2 * np.pi * t / period_seconds + phase)


def stereo(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.stack([left, right], axis=1)


def write(path: Path, data: np.ndarray, *, loop: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if loop:
        data = to_loopable(data)
    sf.write(str(path), np.clip(data, -1.0, 1.0), SR, subtype="PCM_16")
    size = path.stat().st_size / 1024
    duration = len(data) / SR
    print(f"  {path.name:<22} {duration:6.1f}s  {size:8.0f} KB")


# --------------------------------------------------------------------------
# Ambience beds
# --------------------------------------------------------------------------


def bed_drone() -> np.ndarray:
    """Deep, slowly breathing drone. The default under any horror narration."""
    n = int(LOOP_SECONDS * SR)
    t = np.arange(n) / SR
    generator = rng(11)

    tones = np.zeros(n)
    # A minor-ish stack: root, fifth, and a detuned root for slow beating.
    for frequency, amplitude in ((41.2, 0.55), (61.7, 0.30), (82.4, 0.18)):
        tones += amplitude * np.sin(2 * np.pi * frequency * t)
    tones += 0.22 * np.sin(2 * np.pi * 41.9 * t)  # detune -> slow throb

    # Air: filtered noise that gives the drone a sense of a large room.
    air = lowpass(generator.normal(0, 1, n), 320, order=2)
    air *= slow_lfo(n, 23.0, depth=0.55) * 0.30

    mixed = tones * slow_lfo(n, 31.0, depth=0.28) + air
    left = mixed
    right = np.roll(mixed, 97) * 0.97
    return stereo(normalise(left, 0.62), normalise(right, 0.62))


def bed_wind() -> np.ndarray:
    """Hollow wind. Suits outdoor, night, forest and ruins."""
    n = int(LOOP_SECONDS * SR)
    generator = rng(23)

    base = generator.normal(0, 1, n)
    shaped = bandpass(base, 120, 900, order=2)
    # Gusting: several slow modulators at unrelated periods so it never pulses.
    gust = (
        slow_lfo(n, 17.0, depth=0.6)
        * slow_lfo(n, 7.3, phase=1.1, depth=0.35)
        * slow_lfo(n, 3.1, phase=2.4, depth=0.18)
    )
    wind = shaped * gust

    left = wind
    right = bandpass(np.roll(base, 211), 110, 820, order=2) * gust
    return stereo(normalise(left, 0.5), normalise(right, 0.5))


def bed_hum() -> np.ndarray:
    """Electrical hum with a faint buzz. Interiors, basements, hospitals."""
    n = int(LOOP_SECONDS * SR)
    t = np.arange(n) / SR

    hum = (
        np.sin(2 * np.pi * 50 * t)
        + 0.4 * np.sin(2 * np.pi * 100 * t)
        + 0.18 * np.sin(2 * np.pi * 150 * t)
    )
    buzz = bandpass(rng(37).normal(0, 1, n), 1800, 3600, order=2) * 0.05

    mixed = (hum * 0.30 + buzz) * slow_lfo(n, 19.0, depth=0.12)
    return stereo(normalise(mixed, 0.42), normalise(np.roll(mixed, 61), 0.42))


def bed_rain() -> np.ndarray:
    """Steady rain with sparse droplets. Sets dread without any music."""
    n = int(LOOP_SECONDS * SR)
    generator = rng(53)

    hiss = highpass(generator.normal(0, 1, n), 1100, order=2)
    hiss *= slow_lfo(n, 13.0, depth=0.3) * 0.55

    # Sparse droplets: short decaying blips on top of the hiss bed.
    droplets = np.zeros(n)
    count = 900
    positions = generator.integers(0, n - 400, count)
    for start in positions:
        length = int(generator.integers(120, 400))
        envelope = np.exp(-np.linspace(0, 7, length))
        droplets[start : start + length] += envelope * generator.uniform(0.05, 0.22)

    mixed = hiss + bandpass(droplets, 900, 6000, order=2)
    return stereo(normalise(mixed, 0.5), normalise(np.roll(mixed, 143), 0.5))


def bed_heartbeat() -> np.ndarray:
    """Slow heartbeat under a low drone. Builds tension before a reveal."""
    n = int(LOOP_SECONDS * SR)
    beats = 20  # 30 s / 20 = a 1.5 s period, ~40 bpm
    period = n // beats
    out = np.zeros(n)

    for index in range(beats):
        for offset_seconds, amplitude in ((0.0, 0.55), (0.26, 0.34)):
            start = index * period + int(offset_seconds * SR)
            length = int(0.34 * SR)
            if start + length >= n:
                continue
            t = np.arange(length) / SR
            # A thump is a fast pitch drop plus a quick decay.
            frequency = 62 * np.exp(-t * 11) + 28
            thump = np.sin(2 * np.pi * frequency * t) * np.exp(-t * 9)
            out[start : start + length] += thump * amplitude

    drone = lowpass(rng(71).normal(0, 1, n), 160, order=2) * 0.28
    mixed = out + drone
    return stereo(normalise(mixed, 0.6), normalise(np.roll(mixed, 89), 0.6))


# --------------------------------------------------------------------------
# One-shot effects
# --------------------------------------------------------------------------


def sfx_stinger() -> np.ndarray:
    """Sharp dissonant stab, for the moment something appears."""
    duration = 2.2
    n = int(duration * SR)
    t = np.arange(n) / SR
    generator = rng(101)

    # Two detuned sawtooths a semitone apart, so it grates rather than sings.
    saw = signal.sawtooth(2 * np.pi * 220 * t) + signal.sawtooth(2 * np.pi * 233 * t)
    noise = generator.normal(0, 1, n)
    attack = np.minimum(1.0, t / 0.012)
    decay = np.exp(-t * 2.1)
    body = (saw * 0.5 + noise * 0.5) * attack * decay
    return stereo(normalise(body, 0.95), normalise(np.roll(body, 37), 0.95))


def sfx_impact() -> np.ndarray:
    """Low thud, for a door slamming or a body hitting the floor."""
    duration = 1.6
    n = int(duration * SR)
    t = np.arange(n) / SR

    frequency = 120 * np.exp(-t * 9) + 38
    thump = np.sin(2 * np.pi * frequency * t) * np.exp(-t * 5.5)

    crack = highpass(rng(103).normal(0, 1, n), 2200, order=2) * np.exp(-t * 30) * 0.35
    mixed = thump + crack
    return stereo(normalise(mixed, 0.95), normalise(np.roll(mixed, 23), 0.95))


def sfx_riser() -> np.ndarray:
    """Rising tension sweep, for the build before a scare."""
    duration = 4.0
    n = int(duration * SR)
    t = np.arange(n) / SR

    # Frequency sweeps upward across the whole effect.
    sweep = signal.chirp(t, 90, duration, 1500, method="logarithmic", phi=-90)
    tone = sweep * 0.45

    noise = rng(107).normal(0, 1, n)
    # The noise band opens up as it rises, so it brightens with the sweep.
    noise_ramp = np.linspace(0.0, 1.0, n) ** 2
    noisy = bandpass(noise, 300, 4000, order=2) * noise_ramp * 0.5

    envelope = np.minimum(1.0, t / duration) ** 1.5
    mixed = (tone + noisy) * envelope
    return stereo(normalise(mixed, 0.85), normalise(np.roll(mixed, 53), 0.85))


def sfx_whisper() -> np.ndarray:
    """Breathy band-passed noise that reads as a voice just out of earshot."""
    duration = 3.0
    n = int(duration * SR)
    generator = rng(109)

    noise = generator.normal(0, 1, n)
    # Narrow band with a wandering centre frequency sounds voice-like.
    out = np.zeros(n)
    segments = 14
    for index in range(segments):
        start = index * n // segments
        end = (index + 1) * n // segments
        centre = generator.uniform(700, 1900)
        piece = bandpass(noise[start:end], centre * 0.75, centre * 1.25, order=2)
        fade = np.minimum(1.0, np.arange(len(piece)) / 240)
        out[start:end] = piece * fade * generator.uniform(0.4, 1.0)

    return stereo(normalise(out, 0.5), normalise(np.roll(out, 31), 0.5))


def sfx_bell() -> np.ndarray:
    """Distant bell toll. Good for a chapter change or an ending."""
    duration = 5.0
    n = int(duration * SR)
    t = np.arange(n) / SR

    # Inharmonic partials are what make a bell sound like a bell.
    partials = ((1.0, 1.0), (2.76, 0.5), (5.4, 0.28), (8.93, 0.14))
    body = np.zeros(n)
    for ratio, amplitude in partials:
        body += amplitude * np.sin(2 * np.pi * 196 * ratio * t) * np.exp(-t * (0.6 + ratio * 0.22))

    return stereo(normalise(body, 0.7), normalise(np.roll(body, 47), 0.7))


# --------------------------------------------------------------------------
# Textures
# --------------------------------------------------------------------------


def grain_texture(size: int = 512, seed: int = 7) -> Path:
    """A tileable film-grain tile, used with mix-blend-mode: overlay."""
    from PIL import Image

    generator = rng(seed)
    # Slight horizontal blur mimics the grain of real film stock.
    noise = generator.normal(0, 1, (size, size))
    noise = signal.lfilter([0.35, 0.65], [1.0], noise, axis=1)
    noise = np.clip(128 + noise * 46, 0, 255).astype(np.uint8)

    image = Image.fromarray(noise, mode="L").convert("RGB")
    TEXTURES.mkdir(parents=True, exist_ok=True)
    path = TEXTURES / "grain.png"
    image.save(path, optimize=True)
    print(f"  {path.name:<22} {size}x{size}  {path.stat().st_size / 1024:8.0f} KB")
    return path


# --------------------------------------------------------------------------


BEDS = {
    "amb_drone": bed_drone,
    "amb_wind": bed_wind,
    "amb_hum": bed_hum,
    "amb_rain": bed_rain,
    "amb_heartbeat": bed_heartbeat,
}

EFFECTS = {
    "sfx_stinger": sfx_stinger,
    "sfx_impact": sfx_impact,
    "sfx_riser": sfx_riser,
    "sfx_whisper": sfx_whisper,
    "sfx_bell": sfx_bell,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="regenerate existing files")
    args = parser.parse_args()

    print("Textures")
    grain = TEXTURES / "grain.png"
    if grain.exists() and not args.force:
        print(f"  {grain.name} exists (use --force to regenerate)")
    else:
        grain_texture()

    print()
    print("Ambience beds (30 s seamless loops)")
    for name, builder in BEDS.items():
        path = AUDIO / f"{name}.wav"
        if path.exists() and not args.force:
            print(f"  {path.name:<22} exists")
            continue
        write(path, builder(), loop=True)

    print()
    print("One-shot effects")
    for name, builder in EFFECTS.items():
        path = AUDIO / f"{name}.wav"
        if path.exists() and not args.force:
            print(f"  {path.name:<22} exists")
            continue
        write(path, builder(), loop=False)

    print()
    print("Done. These are generated, so the project ships no third-party audio.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
