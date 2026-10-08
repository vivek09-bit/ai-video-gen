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


def bed_ethereal() -> np.ndarray:
    """Dreamy shimmering harmonic pad. Fantasy, magic, fairy tales, and wonder."""
    n = int(LOOP_SECONDS * SR)
    t = np.arange(n) / SR
    gen = rng(61)

    # Chords: C3 (130.81), G3 (196.00), E4 (329.63), B4 (493.88), C5 (523.25)
    chord_freqs = [130.81, 196.00, 329.63, 493.88, 523.25]
    pad = np.zeros(n)
    for i, freq in enumerate(chord_freqs):
        vibrato = 1.0 + 0.003 * np.sin(2 * np.pi * (0.18 + 0.07 * i) * t)
        phase = i * 1.2
        wave = np.sin(2 * np.pi * freq * vibrato * t + phase)
        wave += 0.22 * np.sin(2 * np.pi * freq * 2 * vibrato * t + phase)
        pad += wave * (0.35 / (1.0 + 0.28 * i))

    air = bandpass(gen.normal(0, 1, n), 2200, 7200, order=2)
    air *= slow_lfo(n, 14.0, depth=0.45) * 0.12

    mixed = pad * slow_lfo(n, 21.0, depth=0.25) + air
    left = mixed
    right = np.roll(mixed, 185) * 0.96
    return stereo(normalise(left, 0.55), normalise(right, 0.55))


def bed_cyber() -> np.ndarray:
    """Cybernetic pulse texture with sweeping digital resonances. Sci-fi and cyberpunk."""
    n = int(LOOP_SECONDS * SR)
    t = np.arange(n) / SR
    gen = rng(79)

    pulse = np.sin(2 * np.pi * 55 * t) * 0.6 + 0.3 * np.sin(2 * np.pi * 110 * t)
    pulse *= (0.7 + 0.3 * np.sin(2 * np.pi * 2.0 * t) ** 2)

    raw_noise = gen.normal(0, 1, n)
    stream = bandpass(raw_noise, 450, 3100, order=2) * (0.25 + 0.2 * np.sin(2 * np.pi * 0.12 * t))

    synth_beeps = np.zeros(n)
    beep_freqs = [440, 554.37, 659.25, 880]
    step_samples = int(0.25 * SR)
    for s_idx in range(0, n - step_samples, step_samples):
        b_freq = beep_freqs[(s_idx // step_samples) % len(beep_freqs)]
        seg_t = np.arange(step_samples) / SR
        env = np.exp(-seg_t * 18)
        synth_beeps[s_idx : s_idx + step_samples] += 0.08 * np.sin(2 * np.pi * b_freq * seg_t) * env

    mixed = pulse * 0.45 + stream * 0.35 + synth_beeps * 0.2
    left = mixed
    right = np.roll(mixed, 140) * 0.95
    return stereo(normalise(left, 0.52), normalise(right, 0.52))


def bed_temple() -> np.ndarray:
    """Sacred tanpura and singing bowl resonance. Mythology, Vedic, ancient sacred realms."""
    n = int(LOOP_SECONDS * SR)
    t = np.arange(n) / SR
    gen = rng(83)

    tanpura_notes = [(110.0, 0.35), (146.83, 0.45), (147.2, 0.35), (73.42, 0.5)]
    drone = np.zeros(n)
    for freq, amp in tanpura_notes:
        for harm in (1, 2, 3, 4, 5):
            h_amp = amp / (harm ** 0.85)
            wobble = 1.0 + 0.002 * np.sin(2 * np.pi * 0.2 * t + harm)
            drone += h_amp * np.sin(2 * np.pi * freq * harm * wobble * t)

    bowl = np.zeros(n)
    bowl_interval = int(7.5 * SR)
    bowl_len = int(6.0 * SR)
    for start in range(0, n, bowl_interval):
        end = min(n, start + bowl_len)
        seg_len = end - start
        bt = np.arange(seg_len) / SR
        env = np.exp(-bt * 0.7) * np.minimum(1.0, bt / 0.1)
        b_tone = (
            np.sin(2 * np.pi * 432 * bt) * 0.5
            + np.sin(2 * np.pi * 432 * 2.76 * bt) * 0.3
            + np.sin(2 * np.pi * 432 * 5.4 * bt) * 0.15
        )
        bowl[start:end] += b_tone * env * 0.3

    warm_air = lowpass(gen.normal(0, 1, n), 280, order=2) * 0.2
    mixed = drone * 0.4 + bowl * 0.35 + warm_air * slow_lfo(n, 16.0, depth=0.3)
    left = mixed
    right = np.roll(mixed, 210) * 0.98
    return stereo(normalise(left, 0.55), normalise(right, 0.55))


def bed_warm() -> np.ndarray:
    """Warm acoustic chamber pad with lush strings resonance. Drama, emotion, and romance."""
    n = int(LOOP_SECONDS * SR)
    t = np.arange(n) / SR
    gen = rng(97)

    chord = [87.31, 130.81, 220.00, 329.63]
    pad = np.zeros(n)
    for i, freq in enumerate(chord):
        detune = 1.0 + 0.002 * np.sin(2 * np.pi * (0.15 + 0.05 * i) * t)
        vibrato = np.sin(2 * np.pi * freq * detune * t)
        vibrato += 0.3 * np.sin(2 * np.pi * freq * 2 * detune * t)
        pad += vibrato * (0.4 / (1.0 + 0.25 * i))

    room_air = bandpass(gen.normal(0, 1, n), 300, 1800, order=2) * 0.15
    mixed = pad * slow_lfo(n, 20.0, depth=0.3) + room_air
    left = mixed
    right = np.roll(mixed, 160) * 0.97
    return stereo(normalise(left, 0.5), normalise(right, 0.5))


def bed_tension() -> np.ndarray:
    """Tension bed with subtle clockwork polyrhythm and dark sub drone. Mystery, suspense, documentary."""
    n = int(LOOP_SECONDS * SR)
    t = np.arange(n) / SR
    gen = rng(113)

    drone = np.sin(2 * np.pi * 45 * t) * 0.5 + 0.25 * np.sin(2 * np.pi * 67.5 * t)
    drone *= slow_lfo(n, 18.0, depth=0.25)

    tick_track = np.zeros(n)
    tick_interval = int(1.0 * SR)
    tick_len = int(0.04 * SR)
    for start in range(0, n - tick_len, tick_interval):
        tt = np.arange(tick_len) / SR
        tick = np.sin(2 * np.pi * 1600 * tt) * np.exp(-tt * 180)
        tick_track[start : start + tick_len] += tick * 0.35

    dark_noise = lowpass(gen.normal(0, 1, n), 220, order=2) * 0.25
    mixed = drone * 0.45 + tick_track * 0.25 + dark_noise
    left = mixed
    right = np.roll(mixed, 115) * 0.95
    return stereo(normalise(left, 0.52), normalise(right, 0.52))


def bed_uplift() -> np.ndarray:
    """Bright inspiring ambient chord drone with euphoric swelling pad. Motivational and success."""
    n = int(LOOP_SECONDS * SR)
    t = np.arange(n) / SR
    gen = rng(127)

    chord = [146.83, 220.00, 369.99, 587.33]
    pad = np.zeros(n)
    for i, freq in enumerate(chord):
        lfo = 1.0 + 0.002 * np.sin(2 * np.pi * (0.2 + 0.06 * i) * t)
        wave = np.sin(2 * np.pi * freq * lfo * t)
        wave += 0.22 * np.sin(2 * np.pi * freq * 2 * lfo * t)
        pad += wave * (0.38 / (1.0 + 0.2 * i))

    swell = slow_lfo(n, 15.0, depth=0.35)
    high_air = highpass(gen.normal(0, 1, n), 3000, order=2) * 0.08
    mixed = pad * swell + high_air
    left = mixed
    right = np.roll(mixed, 175) * 0.96
    return stereo(normalise(left, 0.55), normalise(right, 0.55))


def bed_whispers() -> np.ndarray:
    """Multilayered spectral background whispers and shadow murmurs. Horror, mystery, supernatural."""
    n = int(LOOP_SECONDS * SR)
    t = np.arange(n) / SR
    gen = rng(307)

    # Sub-bass dread foundation
    sub = np.sin(2 * np.pi * 38.0 * t) * 0.35 * slow_lfo(n, 14.0, depth=0.4)

    # Wandering formant voice-like whisper layers
    raw_noise = gen.normal(0, 1, n)
    whisper_left = np.zeros(n)
    whisper_right = np.zeros(n)

    # Formant regions (F1: 400-900 Hz, F2: 1200-2400 Hz)
    step = int(0.2 * SR)
    for idx in range(0, n - step, step):
        seg_noise = raw_noise[idx : idx + step]
        f1 = gen.uniform(450.0, 950.0)
        f2 = gen.uniform(1300.0, 2600.0)
        filt1 = bandpass(seg_noise, f1 * 0.75, f1 * 1.35, order=2)
        filt2 = bandpass(seg_noise, f2 * 0.8, f2 * 1.25, order=2) * 0.6
        vocal_grain = (filt1 + filt2) * gen.uniform(0.3, 0.9)

        # Pan back and forth across stereo field
        pan = 0.5 + 0.45 * np.sin(2 * np.pi * (idx / n) * 3.5)
        whisper_left[idx : idx + step] += vocal_grain * pan
        whisper_right[idx : idx + step] += vocal_grain * (1.0 - pan)

    breath_envelope = (0.5 + 0.5 * np.sin(2 * np.pi * 0.18 * t) ** 2) * slow_lfo(n, 8.0, depth=0.3)
    whisper_left *= breath_envelope
    whisper_right *= breath_envelope

    left = sub + whisper_left * 0.7
    right = sub + whisper_right * 0.7
    return stereo(normalise(left, 0.52), normalise(right, 0.52))


def bed_ghost_voices() -> np.ndarray:
    """Eerie supernatural choir and vocal drone pad. Mythology, paranormal, ghostly ruins."""
    n = int(LOOP_SECONDS * SR)
    t = np.arange(n) / SR
    gen = rng(311)

    # Dissonant minor / tritone chord cluster for spine-chilling vocal pad
    # A2 (110), C3 (130.81), D#3 (155.56), E3 (164.81), A3 (220)
    chord = [110.0, 130.81, 155.56, 164.81, 220.0]
    choir = np.zeros(n)
    for i, freq in enumerate(chord):
        vibrato = 1.0 + 0.004 * np.sin(2 * np.pi * (0.2 + 0.05 * i) * t)
        inst_f = freq * vibrato
        # Rich vocal harmonics
        wave = (
            np.sin(2 * np.pi * inst_f * t) * 0.5
            + np.sin(2 * np.pi * inst_f * 2 * t) * 0.3
            + np.sin(2 * np.pi * inst_f * 3 * t) * 0.2
        )
        # Formant shaping (vowel /ah/-/oh/ resonance)
        shaped = bandpass(wave, freq * 0.9, min(SR * 0.45, freq * 4.5), order=2)
        choir += shaped * (0.35 / (1.0 + 0.2 * i))

    spectral_air = bandpass(gen.normal(0, 1, n), 600, 3200, order=2) * 0.12 * slow_lfo(n, 12.0, depth=0.4)
    mixed = choir * slow_lfo(n, 16.0, depth=0.3) + spectral_air
    left = mixed
    right = np.roll(mixed, 185) * 0.96
    return stereo(normalise(left, 0.54), normalise(right, 0.54))


def bed_radio_voices() -> np.ndarray:
    """Muffled CB radio dispatch, police broadcast chatter, and static transmissions. Sci-fi, mystery, urban legends."""
    n = int(LOOP_SECONDS * SR)
    t = np.arange(n) / SR
    gen = rng(317)

    # 60Hz power hum + telephone frequency response (300Hz - 3400Hz)
    hum = np.sin(2 * np.pi * 60.0 * t) * 0.15 + np.sin(2 * np.pi * 120.0 * t) * 0.08
    raw_noise = gen.normal(0, 1, n)
    static = bandpass(raw_noise, 350, 3200, order=2) * 0.25

    # Simulated radio bursts / dispatch vocal formants
    radio_speech = np.zeros(n)
    burst_interval = int(4.5 * SR)
    burst_len = int(2.2 * SR)
    for start in range(0, n - burst_len, burst_interval):
        end = start + burst_len
        seg_t = np.arange(burst_len) / SR
        # Formant chatter carrier
        f_speech = gen.uniform(180.0, 340.0)
        carrier = signal.sawtooth(2 * np.pi * f_speech * seg_t)
        vocal_f = bandpass(carrier, 600, 2400, order=2)
        mod = (0.5 + 0.5 * np.sin(2 * np.pi * gen.uniform(3.5, 8.0) * seg_t)) ** 2
        squelch = gen.normal(0, 1, burst_len) * 0.2
        radio_speech[start:end] += bandpass(vocal_f * mod + squelch, 400, 3000, order=2) * 0.45

    mixed = hum + static + radio_speech
    left = mixed
    right = np.roll(mixed, 95) * 0.94
    return stereo(normalise(left, 0.5), normalise(right, 0.5))


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


def sfx_sparkle() -> np.ndarray:
    """Magical sparkling chime cluster. Fairy tale, fantasy, wonder, discovery."""
    duration = 2.8
    n = int(duration * SR)
    generator = rng(131)
    sparkle = np.zeros(n)

    notes = [1760.0, 2217.46, 2637.02, 3520.0, 4434.92, 5274.04, 3951.07]
    for i in range(24):
        freq = float(generator.choice(notes)) * generator.uniform(0.99, 1.01)
        start_t = generator.uniform(0.0, 1.4)
        start_idx = int(start_t * SR)
        dur = generator.uniform(0.3, 1.1)
        seg_len = min(n - start_idx, int(dur * SR))
        if seg_len <= 0:
            continue
        st = np.arange(seg_len) / SR
        env = np.exp(-st * generator.uniform(4.5, 9.0))
        tone = np.sin(2 * np.pi * freq * st) + 0.3 * np.sin(2 * np.pi * freq * 2 * st)
        sparkle[start_idx : start_idx + seg_len] += tone * env * generator.uniform(0.15, 0.4)

    high_sheen = highpass(generator.normal(0, 1, n), 6000, order=2) * 0.04
    mixed = sparkle + high_sheen * np.exp(-np.arange(n) / (SR * 1.5))
    left = mixed
    right = np.roll(mixed, 75)
    return stereo(normalise(left, 0.8), normalise(right, 0.8))


def sfx_chime() -> np.ndarray:
    """Clear resonant crystal chime / bell. Positive reveals, achievements, clarity."""
    duration = 3.5
    n = int(duration * SR)
    t = np.arange(n) / SR

    partials = ((1.0, 1.0), (2.0, 0.45), (3.01, 0.25), (4.2, 0.12), (5.9, 0.06))
    body = np.zeros(n)
    for ratio, amp in partials:
        freq = 880.0 * ratio
        body += amp * np.sin(2 * np.pi * freq * t) * np.exp(-t * (1.1 + ratio * 0.35))

    left = body
    right = np.roll(body, 60)
    return stereo(normalise(left, 0.85), normalise(right, 0.85))


def sfx_whoosh() -> np.ndarray:
    """Dynamic stereo whoosh / swoosh. Scene cuts, fast action, transitions."""
    duration = 1.2
    n = int(duration * SR)
    t = np.arange(n) / SR
    gen = rng(137)

    noise = gen.normal(0, 1, n)
    sweep_cf = 250.0 + 3200.0 * np.sin(np.pi * (t / duration)) ** 2
    # Apply time-varying bandpass via segmented filtering
    filtered = np.zeros(n)
    step = int(0.02 * SR)
    for idx in range(0, n, step):
        chunk = noise[idx : min(n, idx + step)]
        cf = sweep_cf[min(n - 1, idx + len(chunk) // 2)]
        f_low = max(40.0, cf * 0.5)
        f_high = min(SR * 0.45, cf * 1.6)
        chunk_f = bandpass(chunk, f_low, f_high, order=2) if f_high > f_low + 20 else chunk
        filtered[idx : idx + len(chunk)] = chunk_f

    env = (np.sin(np.pi * (t / duration)) ** 2) * np.minimum(1.0, t / 0.08)
    pan_l = np.cos((np.pi / 2.0) * (t / duration))
    pan_r = np.sin((np.pi / 2.0) * (t / duration))
    left = filtered * env * pan_l
    right = filtered * env * pan_r
    return stereo(normalise(left, 0.9), normalise(right, 0.9))


def sfx_boom() -> np.ndarray:
    """Deep cinematic sub-bass explosion / boom impact. Dramatic turning points and action."""
    duration = 3.2
    n = int(duration * SR)
    t = np.arange(n) / SR
    gen = rng(149)

    freq = 95.0 * np.exp(-t * 6.5) + 26.0
    sub = np.sin(2 * np.pi * freq * t) * np.exp(-t * 1.4)

    noise = lowpass(gen.normal(0, 1, n), 260, order=2) * np.exp(-t * 2.8) * 0.7
    crack = bandpass(gen.normal(0, 1, n), 800, 3200, order=2) * np.exp(-t * 22.0) * 0.5

    mixed = sub * 0.65 + noise * 0.4 + crack * 0.3
    left = mixed
    right = np.roll(mixed, 35)
    return stereo(normalise(left, 0.95), normalise(right, 0.95))


def sfx_glitch() -> np.ndarray:
    """Digital glitch and circuit zap. Cyberpunk, sci-fi, error, hacker aesthetics."""
    duration = 1.0
    n = int(duration * SR)
    t = np.arange(n) / SR
    gen = rng(151)

    carrier = np.sin(2 * np.pi * (1400.0 + 800.0 * np.sign(np.sin(2 * np.pi * 32.0 * t))) * t)
    noise = gen.choice([-1.0, 0.0, 1.0], size=n) * 0.4
    burst = (carrier + noise) * (gen.uniform(0, 1, n) > 0.35)
    env = np.exp(-t * 3.0)

    mixed = burst * env
    left = mixed
    right = np.roll(mixed, 85)
    return stereo(normalise(left, 0.85), normalise(right, 0.85))


def sfx_horn() -> np.ndarray:
    """Ancient brass war horn / temple conch. Mythology, epic reveals, fantasy."""
    duration = 3.8
    n = int(duration * SR)
    t = np.arange(n) / SR

    f0 = 110.0
    vibrato = 1.0 + 0.006 * np.sin(2 * np.pi * 4.8 * t)
    f_inst = f0 * vibrato

    saw1 = signal.sawtooth(2 * np.pi * f_inst * t)
    saw2 = signal.sawtooth(2 * np.pi * f_inst * 1.5 * t)
    brass = saw1 * 0.6 + saw2 * 0.4
    brass = lowpass(brass, 1200, order=2)

    attack = np.minimum(1.0, t / 0.35)
    decay = np.exp(-np.maximum(0.0, t - 2.0) * 1.8)
    body = brass * attack * decay

    left = body
    right = np.roll(body, 120)
    return stereo(normalise(left, 0.88), normalise(right, 0.88))


def sfx_pop() -> np.ndarray:
    """Playful bubbly pop / snap. Comedy, lighthearted moments, whimsical beats."""
    duration = 0.35
    n = int(duration * SR)
    t = np.arange(n) / SR

    freq = 900.0 * np.exp(-t * 26.0) + 180.0
    pop = np.sin(2 * np.pi * freq * t) * np.exp(-t * 22.0)
    click = (rng(163).normal(0, 1, n)) * np.exp(-t * 120.0) * 0.25

    mixed = pop + click
    left = mixed
    right = np.roll(mixed, 15)
    return stereo(normalise(left, 0.9), normalise(right, 0.9))


def sfx_swell() -> np.ndarray:
    """Emotive harmonic string swell. Drama, emotional realization, inspiration."""
    duration = 3.5
    n = int(duration * SR)
    t = np.arange(n) / SR

    chord = [130.81, 196.00, 261.63, 329.63]
    swell = np.zeros(n)
    for i, freq in enumerate(chord):
        detune = 1.0 + 0.003 * np.sin(2 * np.pi * (0.3 + 0.08 * i) * t)
        wave = np.sin(2 * np.pi * freq * detune * t) + 0.25 * np.sin(2 * np.pi * freq * 2 * detune * t)
        swell += wave * (0.4 / (1.0 + 0.2 * i))

    env = (np.sin(np.pi * (t / duration) ** 1.3) ** 2) * np.minimum(1.0, t / 0.5)
    body = swell * env
    left = body
    right = np.roll(body, 90)
    return stereo(normalise(left, 0.88), normalise(right, 0.88))


def sfx_clock_tick() -> np.ndarray:
    """Crisp clock tick-tock sequence. Mystery, suspense, countdown, ticking timer."""
    duration = 1.8
    n = int(duration * SR)
    t = np.arange(n) / SR
    gen = rng(173)

    track = np.zeros(n)
    # Tick at 0.1s, Tock at 0.9s
    for tick_time, freq in [(0.1, 1900.0), (0.9, 1350.0)]:
        start_idx = int(tick_time * SR)
        tick_len = int(0.06 * SR)
        tt = np.arange(tick_len) / SR
        click = np.sin(2 * np.pi * freq * tt) * np.exp(-tt * 140.0)
        noise_tap = gen.normal(0, 1, tick_len) * np.exp(-tt * 200.0) * 0.35
        track[start_idx : start_idx + tick_len] += click + noise_tap

    left = track
    right = np.roll(track, 20)
    return stereo(normalise(left, 0.85), normalise(right, 0.85))


def sfx_cinematic_hit() -> np.ndarray:
    """Blockbuster cinematic impact hit with metallic tail. Thriller, climax, shock."""
    duration = 2.6
    n = int(duration * SR)
    t = np.arange(n) / SR
    gen = rng(181)

    f_sub = 80.0 * np.exp(-t * 9.0) + 32.0
    sub = np.sin(2 * np.pi * f_sub * t) * np.exp(-t * 2.2)
    metal = (np.sin(2 * np.pi * 440 * t) + np.sin(2 * np.pi * 620 * t) + np.sin(2 * np.pi * 930 * t)) * np.exp(-t * 4.5)
    crack = bandpass(gen.normal(0, 1, n), 1200, 4800, order=2) * np.exp(-t * 35.0)

    mixed = sub * 0.6 + metal * 0.35 + crack * 0.45
    left = mixed
    right = np.roll(mixed, 40)
    return stereo(normalise(left, 0.95), normalise(right, 0.95))


def sfx_jumpscare() -> np.ndarray:
    """Brutal sudden shrieking jumpscare hit with terrifying dissonant sub crash."""
    duration = 2.4
    n = int(duration * SR)
    t = np.arange(n) / SR
    gen = rng(193)

    # Dissonant cluster shrieks: tritone + minor 2nd clash
    shriek = (
        signal.sawtooth(2 * np.pi * 666.0 * t) * 0.45
        + signal.sawtooth(2 * np.pi * 707.0 * t) * 0.4
        + signal.sawtooth(2 * np.pi * 1414.0 * t) * 0.3
    )
    shriek = bandpass(shriek, 500, 5500, order=2)

    # Explosive transient burst
    noise = gen.normal(0, 1, n)
    burst = highpass(noise, 1500, order=2) * np.exp(-t * 25.0) * 0.85

    # Bone-rattling sub punch
    f_sub = 110.0 * np.exp(-t * 12.0) + 28.0
    sub = np.sin(2 * np.pi * f_sub * t) * np.exp(-t * 3.2) * 0.8

    env = np.exp(-t * 2.2) * np.minimum(1.0, t / 0.003)
    mixed = (shriek * 0.55 + burst + sub) * env
    left = mixed
    right = np.roll(mixed, 30) * 0.96
    return stereo(normalise(left, 0.98), normalise(right, 0.98))


def sfx_creak() -> np.ndarray:
    """Eerie slow wooden door / floorboard creak in an abandoned room."""
    duration = 2.5
    n = int(duration * SR)
    t = np.arange(n) / SR
    gen = rng(197)

    # FM synthesis for wood resonance friction
    mod = np.sin(2 * np.pi * (18.0 + 8.0 * np.sin(2 * np.pi * 1.5 * t)) * t) * 90.0
    carrier_freq = 320.0 + 140.0 * np.sin(2 * np.pi * 0.8 * t) + mod
    creak_raw = signal.sawtooth(2 * np.pi * np.cumsum(carrier_freq) / SR)
    wood_filter = bandpass(creak_raw, 200, 1800, order=2)

    # Friction texture
    friction = gen.normal(0, 1, n) * (creak_raw > 0.6) * 0.35
    friction_f = bandpass(friction, 400, 2800, order=2)

    env = (np.sin(np.pi * (t / duration) ** 0.8) ** 1.5) * np.minimum(1.0, t / 0.15)
    mixed = (wood_filter * 0.7 + friction_f * 0.3) * env
    left = mixed
    right = np.roll(mixed, 70) * 0.95
    return stereo(normalise(left, 0.85), normalise(right, 0.85))


def sfx_ghostly_groan() -> np.ndarray:
    """Supernatural entity vocal groan / demon breathing from the dark."""
    duration = 3.8
    n = int(duration * SR)
    t = np.arange(n) / SR
    gen = rng(199)

    f_drift = 85.0 - 25.0 * (t / duration) + 6.0 * np.sin(2 * np.pi * 3.5 * t)
    pulse = signal.sawtooth(2 * np.pi * np.cumsum(f_drift) / SR) * 0.6
    vocal = bandpass(pulse, 180, 850, order=2)

    # Breath turbulences
    breath = gen.normal(0, 1, n)
    breath_f = bandpass(breath, 350, 1600, order=2) * (0.4 + 0.3 * np.sin(2 * np.pi * 2.0 * t))

    env = (np.sin(np.pi * (t / duration)) ** 1.8) * np.minimum(1.0, t / 0.4)
    mixed = (vocal * 0.65 + breath_f * 0.35) * env
    left = mixed
    right = np.roll(mixed, 110) * 0.94
    return stereo(normalise(left, 0.88), normalise(right, 0.88))


def sfx_scare_string() -> np.ndarray:
    """Horror string crescendo screech (Psycho / Insidious style shrieking violin scrape)."""
    duration = 3.2
    n = int(duration * SR)
    t = np.arange(n) / SR
    gen = rng(211)

    # Screeching high pitches with intense tremolo & jitter
    freq1 = 1864.66 + 30.0 * np.sin(2 * np.pi * 14.0 * t)  # A#6
    freq2 = 1975.53 + 35.0 * np.sin(2 * np.pi * 16.0 * t)  # B6
    freq3 = 2637.02 + 45.0 * np.sin(2 * np.pi * 19.0 * t)  # E7

    strings = (
        signal.sawtooth(2 * np.pi * np.cumsum(freq1) / SR) * 0.35
        + signal.sawtooth(2 * np.pi * np.cumsum(freq2) / SR) * 0.35
        + signal.sawtooth(2 * np.pi * np.cumsum(freq3) / SR) * 0.30
    )
    bow_scrape = bandpass(gen.normal(0, 1, n), 2200, 7500, order=2) * 0.25

    attack_curve = (t / duration) ** 2.2
    tremolo = 0.7 + 0.3 * np.sin(2 * np.pi * 22.0 * t)
    mixed = (strings + bow_scrape) * attack_curve * tremolo
    left = mixed
    right = np.roll(mixed, 55) * 0.96
    return stereo(normalise(left, 0.92), normalise(right, 0.92))


def sfx_dark_whoosh() -> np.ndarray:
    """Sinister shadowy swoosh / phantom rushing past the camera."""
    duration = 1.6
    n = int(duration * SR)
    t = np.arange(n) / SR
    gen = rng(223)

    raw_noise = gen.normal(0, 1, n)
    sweep_cf = 120.0 + 2400.0 * np.sin(np.pi * (t / duration)) ** 3
    filtered = np.zeros(n)
    step = int(0.015 * SR)
    for idx in range(0, n, step):
        chunk = raw_noise[idx : min(n, idx + step)]
        cf = sweep_cf[min(n - 1, idx + len(chunk) // 2)]
        f_low = max(30.0, cf * 0.4)
        f_high = min(SR * 0.45, cf * 1.5)
        filtered[idx : idx + len(chunk)] = bandpass(chunk, f_low, f_high, order=2) if f_high > f_low + 20 else chunk

    sub_thump = np.sin(2 * np.pi * 45 * t) * np.exp(-((t - 0.7) / 0.3) ** 2) * 0.5
    env = (np.sin(np.pi * (t / duration)) ** 2) * np.minimum(1.0, t / 0.05)
    pan_l = np.cos((np.pi / 2.0) * (t / duration))
    pan_r = np.sin((np.pi / 2.0) * (t / duration))
    left = (filtered * 0.75 + sub_thump) * env * pan_l
    right = (filtered * 0.75 + sub_thump) * env * pan_r
    return stereo(normalise(left, 0.92), normalise(right, 0.92))


def sfx_footstep_wood() -> np.ndarray:
    """Heavy ominous single footstep on hollow old floorboard."""
    duration = 0.9
    n = int(duration * SR)
    t = np.arange(n) / SR
    gen = rng(227)

    f_heel = 90.0 * np.exp(-t * 28.0) + 40.0
    heel = np.sin(2 * np.pi * f_heel * t) * np.exp(-t * 18.0) * 0.65
    wood_creak_tap = bandpass(signal.sawtooth(2 * np.pi * 380 * t), 250, 1400, order=2) * np.exp(-t * 22.0) * 0.35
    grit = bandpass(gen.normal(0, 1, n), 1200, 4200, order=2) * np.exp(-t * 45.0) * 0.25

    mixed = heel + wood_creak_tap + grit
    left = mixed
    right = np.roll(mixed, 25)
    return stereo(normalise(left, 0.9), normalise(right, 0.9))


def sfx_monster_growl() -> np.ndarray:
    """Guttural terrifying demon / beast growl vibrating in chest."""
    duration = 2.8
    n = int(duration * SR)
    t = np.arange(n) / SR
    gen = rng(229)

    f_growl = 42.0 + 12.0 * np.sin(2 * np.pi * 5.0 * t) + 8.0 * np.sin(2 * np.pi * 1.2 * t)
    pulse = signal.sawtooth(2 * np.pi * np.cumsum(f_growl) / SR)
    distorted = np.tanh(pulse * 3.5)

    sub_body = lowpass(distorted, 350, order=2) * 0.65
    snarl = bandpass(distorted, 600, 2200, order=2) * (0.6 + 0.4 * np.sin(2 * np.pi * 18.0 * t)) * 0.4

    env = (np.sin(np.pi * (t / duration) ** 0.9) ** 1.6) * np.minimum(1.0, t / 0.1)
    mixed = (sub_body + snarl) * env
    left = mixed
    right = np.roll(mixed, 65) * 0.95
    return stereo(normalise(left, 0.95), normalise(right, 0.95))


def sfx_heartbeat_single() -> np.ndarray:
    """Isolated high-impact double-thump heartbeat (lub-dub). Panic / sudden terror."""
    duration = 1.2
    n = int(duration * SR)
    t = np.arange(n) / SR

    out = np.zeros(n)
    for offset, amp in ((0.0, 0.85), (0.24, 0.55)):
        start = int(offset * SR)
        length = int(0.32 * SR)
        if start + length > n:
            continue
        seg_t = np.arange(length) / SR
        f = 68.0 * np.exp(-seg_t * 14.0) + 26.0
        thump = np.sin(2 * np.pi * f * seg_t) * np.exp(-seg_t * 9.5) * amp
        out[start : start + length] += thump

    sub_drone = np.sin(2 * np.pi * 32.0 * t) * np.exp(-t * 2.5) * 0.25
    mixed = out + sub_drone
    left = mixed
    right = np.roll(mixed, 30)
    return stereo(normalise(left, 0.95), normalise(right, 0.95))


def sfx_ghost_laugh() -> np.ndarray:
    """Eerie distant supernatural chuckle / giggle echoing in the dark."""
    duration = 3.2
    n = int(duration * SR)
    t = np.arange(n) / SR
    gen = rng(337)

    # Rhythmic staccato formant pulses simulating laughing cadence (ha-ha-ha-ha)
    laugh_track = np.zeros(n)
    pulses = [0.15, 0.45, 0.75, 1.05, 1.38, 1.72]
    base_pitches = [520.0, 560.0, 610.0, 580.0, 510.0, 440.0]

    for p_time, f0 in zip(pulses, base_pitches):
        start_idx = int(p_time * SR)
        pulse_len = int(0.22 * SR)
        if start_idx + pulse_len > n:
            continue
        pt = np.arange(pulse_len) / SR
        # Formant voice tone
        carrier = signal.sawtooth(2 * np.pi * f0 * pt)
        formant = bandpass(carrier, f0 * 0.85, f0 * 2.6, order=2)
        env = (np.sin(np.pi * (pt / 0.22)) ** 2) * np.exp(-pt * 6.0)
        breath = gen.normal(0, 1, pulse_len) * 0.25 * env
        laugh_track[start_idx : start_idx + pulse_len] += (formant * 0.75 + breath) * env

    # Echo reverb tail
    reverb_tail = bandpass(np.roll(laugh_track, int(0.28 * SR)), 400, 2400, order=2) * 0.45
    mixed = laugh_track + reverb_tail
    left = mixed
    right = np.roll(mixed, 85) * 0.95
    return stereo(normalise(left, 0.88), normalise(right, 0.88))


def sfx_phantom_shriek() -> np.ndarray:
    """Terrifying banshee / phantom screech cutting through the dark with resonant wail."""
    duration = 3.0
    n = int(duration * SR)
    t = np.arange(n) / SR
    gen = rng(349)

    # Gliding pitch with chaotic vibrato
    f_shriek = 1400.0 + 850.0 * np.sin(np.pi * (t / duration) ** 0.8) + 90.0 * np.sin(2 * np.pi * 24.0 * t)
    carrier = signal.sawtooth(2 * np.pi * np.cumsum(f_shriek) / SR)
    shriek_vocal = bandpass(carrier, 800, 6500, order=2)

    # Harsh breath turbulence
    noise = gen.normal(0, 1, n)
    shriek_noise = bandpass(noise, 1500, 7000, order=2) * (0.5 + 0.5 * np.sin(2 * np.pi * 16.0 * t))

    env = (np.sin(np.pi * (t / duration) ** 0.7) ** 1.8) * np.minimum(1.0, t / 0.08)
    mixed = (shriek_vocal * 0.65 + shriek_noise * 0.45) * env
    left = mixed
    right = np.roll(mixed, 45) * 0.95
    return stereo(normalise(left, 0.95), normalise(right, 0.95))


def sfx_demonic_whisper() -> np.ndarray:
    """Close-mic binaural demonic growled whisper right next to the listener's ear."""
    duration = 2.6
    n = int(duration * SR)
    t = np.arange(n) / SR
    gen = rng(353)

    # Low guttural vocal cord pulse
    f_pulse = 65.0 + 15.0 * np.sin(2 * np.pi * 4.0 * t)
    throat_pulse = signal.sawtooth(2 * np.pi * np.cumsum(f_pulse) / SR)
    throat_body = bandpass(throat_pulse, 90, 650, order=2)

    # Close whisper rasp
    noise = gen.normal(0, 1, n)
    whisper_rasp = bandpass(noise, 800, 3800, order=2) * (0.6 + 0.4 * np.sin(2 * np.pi * 12.0 * t))

    env = (np.sin(np.pi * (t / duration)) ** 1.5) * np.minimum(1.0, t / 0.12)
    left_sig = (throat_body * 0.55 + whisper_rasp * 0.45) * env
    right_sig = (throat_body * 0.50 + np.roll(whisper_rasp, 70) * 0.50) * env
    return stereo(normalise(left_sig, 0.92), normalise(right_sig, 0.92))


def sfx_eerie_drone_swell() -> np.ndarray:
    """Long ominous sub-harmonic atmospheric drone swell (4.5s)."""
    duration = 4.5
    n = int(duration * SR)
    t = np.arange(n) / SR
    gen = rng(401)

    f_sub1 = 44.0 + 6.0 * np.sin(2 * np.pi * 0.3 * t)
    f_sub2 = 55.0 + 8.0 * np.sin(2 * np.pi * 0.45 * t)
    f_high = 220.0 + 40.0 * (t / duration) ** 2
    drone = (
        np.sin(2 * np.pi * np.cumsum(f_sub1) / SR) * 0.45
        + np.sin(2 * np.pi * np.cumsum(f_sub2) / SR) * 0.35
        + signal.sawtooth(2 * np.pi * np.cumsum(f_high) / SR) * 0.20
    )
    drone_f = lowpass(drone, 450, order=2)
    noise = bandpass(gen.normal(0, 1, n), 200, 1800, order=2) * 0.25
    env = np.sin(np.pi * (t / duration)) ** 1.6
    mixed = (drone_f * 0.75 + noise * 0.25) * env
    left = mixed
    right = np.roll(mixed, 95) * 0.96
    return stereo(normalise(left, 0.90), normalise(right, 0.90))


def sfx_ghost_crying() -> np.ndarray:
    """Eerie distant supernatural wailing / weeping entity vocal (4.2s)."""
    duration = 4.2
    n = int(duration * SR)
    t = np.arange(n) / SR
    gen = rng(409)

    pitch_mod = 420.0 + 120.0 * np.sin(2 * np.pi * 0.85 * t) - 60.0 * (t / duration) + 18.0 * np.sin(2 * np.pi * 6.5 * t)
    carrier = signal.sawtooth(2 * np.pi * np.cumsum(pitch_mod) / SR)
    vocal_formant = bandpass(carrier, 350, 2400, order=2)
    sob_envelope = 0.6 + 0.4 * (np.sin(2 * np.pi * 1.8 * t) ** 2)
    breath = bandpass(gen.normal(0, 1, n), 800, 3200, order=2) * 0.28 * sob_envelope
    env = (np.sin(np.pi * (t / duration)) ** 1.4) * np.minimum(1.0, t / 0.3)
    mixed = (vocal_formant * 0.7 + breath) * env
    left = mixed
    right = np.roll(mixed, 120) * 0.94
    return stereo(normalise(left, 0.88), normalise(right, 0.88))


def sfx_creepy_music_box() -> np.ndarray:
    """Chilling antique music box melody in a minor scale with eerie flutter (4.6s)."""
    duration = 4.6
    n = int(duration * SR)
    t = np.arange(n) / SR
    gen = rng(419)

    notes = [
        (0.1, 523.25),   # C5
        (0.85, 622.25),  # Eb5
        (1.65, 783.99),  # G5
        (2.45, 830.61),  # Ab5
        (3.30, 587.33),  # D5
    ]
    box = np.zeros(n)
    for onset, freq in notes:
        start_idx = int(onset * SR)
        note_len = int(1.4 * SR)
        if start_idx + note_len > n:
            note_len = n - start_idx
        nt = np.arange(note_len) / SR
        flutter = 1.0 + 0.004 * np.sin(2 * np.pi * 5.0 * nt)
        tine = np.sin(2 * np.pi * freq * flutter * nt) * 0.7 + np.sin(2 * np.pi * freq * 2.76 * nt) * 0.3
        decay = np.exp(-nt * 3.2)
        box[start_idx : start_idx + note_len] += tine * decay

    tape_noise = bandpass(gen.normal(0, 1, n), 1200, 5500, order=2) * 0.08
    env = np.minimum(1.0, (duration - t) / 0.4)
    mixed = (box + tape_noise) * env
    left = mixed
    right = np.roll(mixed, 60) * 0.95
    return stereo(normalise(left, 0.88), normalise(right, 0.88))


def sfx_thunder_rumble() -> np.ndarray:
    """Deep atmospheric rolling thunder rumble across the horizon (4.8s)."""
    duration = 4.8
    n = int(duration * SR)
    t = np.arange(n) / SR
    gen = rng(431)

    raw_noise = gen.normal(0, 1, n)
    sub_rumble = lowpass(raw_noise, 85, order=2)
    mid_crack = bandpass(raw_noise, 110, 450, order=2) * np.exp(-((t - 0.8) / 0.6) ** 2) * 0.6
    sub_sine = np.sin(2 * np.pi * (38.0 + 8.0 * np.sin(2 * np.pi * 0.4 * t)) * t) * 0.4
    env = np.sin(np.pi * (t / duration) ** 0.6) ** 1.8
    mixed = (sub_rumble * 0.7 + mid_crack * 0.5 + sub_sine * 0.3) * env
    left = mixed
    right = np.roll(mixed, 150) * 0.95
    return stereo(normalise(left, 0.92), normalise(right, 0.92))


def sfx_haunted_breathing() -> np.ndarray:
    """Chilling entity breathing and chest vibration close to mic (4.0s)."""
    duration = 4.0
    n = int(duration * SR)
    t = np.arange(n) / SR
    gen = rng(439)

    breath_noise = gen.normal(0, 1, n)
    inhale_env = np.exp(-((t - 1.0) / 0.5) ** 2)
    exhale_env = np.exp(-((t - 2.9) / 0.6) ** 2)
    inhale_f = bandpass(breath_noise, 400, 2200, order=2) * inhale_env
    exhale_f = bandpass(breath_noise, 250, 1600, order=2) * exhale_env * 1.2
    chest_sub = np.sin(2 * np.pi * 55.0 * t) * (inhale_env * 0.3 + exhale_env * 0.4)
    mixed = inhale_f + exhale_f + chest_sub
    left = mixed
    right = np.roll(mixed, 80) * 0.95
    return stereo(normalise(left, 0.88), normalise(right, 0.88))


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
    "amb_ethereal": bed_ethereal,
    "amb_cyber": bed_cyber,
    "amb_temple": bed_temple,
    "amb_warm": bed_warm,
    "amb_tension": bed_tension,
    "amb_uplift": bed_uplift,
    "amb_whispers": bed_whispers,
    "amb_ghost_voices": bed_ghost_voices,
    "amb_radio_voices": bed_radio_voices,
}

EFFECTS = {
    "sfx_stinger": sfx_stinger,
    "sfx_impact": sfx_impact,
    "sfx_riser": sfx_riser,
    "sfx_whisper": sfx_whisper,
    "sfx_bell": sfx_bell,
    "sfx_sparkle": sfx_sparkle,
    "sfx_chime": sfx_chime,
    "sfx_whoosh": sfx_whoosh,
    "sfx_boom": sfx_boom,
    "sfx_glitch": sfx_glitch,
    "sfx_horn": sfx_horn,
    "sfx_pop": sfx_pop,
    "sfx_swell": sfx_swell,
    "sfx_clock_tick": sfx_clock_tick,
    "sfx_cinematic_hit": sfx_cinematic_hit,
    "sfx_jumpscare": sfx_jumpscare,
    "sfx_creak": sfx_creak,
    "sfx_ghostly_groan": sfx_ghostly_groan,
    "sfx_scare_string": sfx_scare_string,
    "sfx_dark_whoosh": sfx_dark_whoosh,
    "sfx_footstep_wood": sfx_footstep_wood,
    "sfx_monster_growl": sfx_monster_growl,
    "sfx_heartbeat_single": sfx_heartbeat_single,
    "sfx_ghost_laugh": sfx_ghost_laugh,
    "sfx_phantom_shriek": sfx_phantom_shriek,
    "sfx_demonic_whisper": sfx_demonic_whisper,
    "sfx_eerie_drone_swell": sfx_eerie_drone_swell,
    "sfx_ghost_crying": sfx_ghost_crying,
    "sfx_creepy_music_box": sfx_creepy_music_box,
    "sfx_thunder_rumble": sfx_thunder_rumble,
    "sfx_haunted_breathing": sfx_haunted_breathing,
}


def ensure_assets(force: bool = False, verbose: bool = False) -> None:
    """Ensure all required textures, ambience beds, and SFX files exist in public/."""
    grain = TEXTURES / "grain.png"
    if not grain.exists() or force:
        grain_texture()
    elif verbose:
        print(f"  {grain.name} exists")

    for name, builder in BEDS.items():
        path = AUDIO / f"{name}.wav"
        if not path.exists() or force:
            write(path, builder(), loop=True)
            if verbose:
                print(f"  generated {path.name}")
        elif verbose:
            print(f"  {path.name} exists")

    for name, builder in EFFECTS.items():
        path = AUDIO / f"{name}.wav"
        if not path.exists() or force:
            write(path, builder(), loop=False)
            if verbose:
                print(f"  generated {path.name}")
        elif verbose:
            print(f"  {path.name} exists")


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
    print(f"Ambience beds ({len(BEDS)} 30 s seamless loops)")
    for name, builder in BEDS.items():
        path = AUDIO / f"{name}.wav"
        if path.exists() and not args.force:
            print(f"  {path.name:<22} exists")
            continue
        write(path, builder(), loop=True)

    print()
    print(f"One-shot effects ({len(EFFECTS)} procedural sfx)")
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
