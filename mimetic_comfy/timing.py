"""Lengths and prompt clauses of the official mimetic-a0 tools (mimetic_a0 9f91b67, stable-audio-tools 3241adb). No ComfyUI or torch needed."""

from __future__ import annotations

import math
import re

SAMPLE_RATE = 48000
DOWNSAMPLING = 1920  # the ACE-Step 1.5 autoencoder
SAMPLE_SIZE = 768000  # model_config.json "sample_size": the 16 s window
HEADROOM = 6.0  # generate_diffusion_cond(duration_padding_sec=6.0)
MAX_SECONDS = SAMPLE_SIZE / SAMPLE_RATE
BARS = (1, 2, 4)


def clamp_seconds(seconds: float) -> float:
    """mimetic_a0.generate() caps the length at the 16 s window."""
    return min(float(seconds), MAX_SECONDS)


def latent_length(seconds: float) -> int:
    """Latent frames generate_diffusion_cond samples for `seconds`: the model sets mask_padding_attention, so the length is
    seconds_total + 6 s, rounded up to a whole latent frame and capped at the window."""
    samples = int((seconds + HEADROOM) * SAMPLE_RATE)
    samples = (samples + DOWNSAMPLING - 1) // DOWNSAMPLING * DOWNSAMPLING
    return min(samples, SAMPLE_SIZE) // DOWNSAMPLING


def valid_length(seconds: float) -> int:
    """Frames the official padding mask keeps (sample_diffusion: seconds_total + 6 s headroom, capped at the latent length)."""
    return min(math.ceil(int(seconds * SAMPLE_RATE) / DOWNSAMPLING) + int(HEADROOM * SAMPLE_RATE / DOWNSAMPLING), latent_length(seconds))


def output_samples(seconds: float) -> int:
    """mimetic_a0.generate() trims the decoded audio to this many samples."""
    return int(seconds * SAMPLE_RATE)


def bars_to_seconds(bars: float, bpm: float) -> float:
    """Length of `bars` bars of 4/4 at `bpm` (mimetic_a0.bars_to_seconds)."""
    return bars * 240.0 / bpm


def add_clauses(prompt: str, bpm: float, bars: int) -> str:
    """The command line's --bpm / --bars: add "N BPM" and "N bars" if the prompt does not have them, in the clause order of the
    training captions (..., tempo, key, length)."""
    if bpm and not re.search(r"\d\s*BPM", prompt):
        prompt += f", {bpm:g} BPM"
    if bars and not re.search(r"\d\s*bars?\b", prompt):
        prompt += f", {bars} bar{'s' if bars > 1 else ''}"
    return prompt
