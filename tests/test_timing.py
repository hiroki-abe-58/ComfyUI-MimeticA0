"""Lengths and prompt clauses against what the official tools did (latent frames and trimmed samples of seven official runs)."""

import pytest

from mimetic_comfy import timing

# seconds -> (latent frames, samples of the audio mimetic_a0.generate() returned), from the official runs in docs/VERIFICATION.md
OFFICIAL = {3.0: (225, 144000), 4.0: (250, 192000), 1.5: (188, 72000), 7.741935483870968: (344, 371612),
            10.105263157894736: (400, 485052), 10.666666666666666: (400, 512000), 16.0: (400, 768000)}


@pytest.mark.parametrize("seconds", sorted(OFFICIAL))
def test_lengths_are_the_official_ones(seconds):
    assert (timing.latent_length(seconds), timing.output_samples(seconds)) == OFFICIAL[seconds]


def test_the_official_padding_mask_never_masks_a_frame():
    """sample_diffusion masks frames past seconds_total + 6 s; with the latent sized the same way none is masked, so the pack has no mask."""
    lengths = [i / 1000 for i in range(1, 16001)] + [timing.bars_to_seconds(b, bpm) for bpm in range(40, 241) for b in timing.BARS]
    assert all(timing.valid_length(timing.clamp_seconds(s)) == timing.latent_length(timing.clamp_seconds(s)) for s in lengths)


def test_length_is_capped_at_the_window():
    assert timing.clamp_seconds(30.0) == timing.MAX_SECONDS == 16.0
    assert timing.latent_length(16.0) == timing.SAMPLE_SIZE // timing.DOWNSAMPLING == 400


def test_bars_to_seconds():
    assert timing.bars_to_seconds(4, 90) == pytest.approx(10.666666666666666, abs=0)
    assert timing.bars_to_seconds(1, 120) == 2.0


@pytest.mark.parametrize("prompt, bpm, bars, expected", [
    ("hip hop drum loop", 90.0, 4, "hip hop drum loop, 90 BPM, 4 bars"),
    ("hip hop drum loop", 92.5, 1, "hip hop drum loop, 92.5 BPM, 1 bar"),
    ("hip hop drum loop, 90 BPM", 90.0, 2, "hip hop drum loop, 90 BPM, 2 bars"),
    ("pads, 4 bars", 100.0, 4, "pads, 4 bars, 100 BPM"),
    ("single cello note, G3", 0.0, 0, "single cello note, G3"),
])
def test_clauses_like_the_command_line(prompt, bpm, bars, expected):
    assert timing.add_clauses(prompt, bpm, bars) == expected
