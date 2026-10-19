"""The sampler parts that are not the model: schedule, starting noise and guidance, against the official values and formulas."""

import pytest
import torch

from mimetic_comfy import sampling

# the timesteps of the official 50-step run (stable-audio-tools 3241adb on CUDA; the last step goes to 0)
OFFICIAL_50 = [1.0, 0.9976146, 0.9971907, 0.9966917, 0.9961044, 0.9954134, 0.9946003, 0.9936441, 0.9925199, 0.9911986, 0.9896463, 0.9878235,
               0.9856846, 0.9831764, 0.9802373, 0.976797, 0.9727745, 0.9680775, 0.9626012, 0.956228, 0.9488263, 0.9402513, 0.930345, 0.9189377,
               0.9058513, 0.8909032, 0.873912, 0.8547062, 0.8331335, 0.8090733, 0.7824498, 0.7532459, 0.7215172, 0.6874017, 0.6511277, 0.6130142,
               0.5734643, 0.5329521, 0.4920007, 0.4511563, 0.4109595, 0.3719174, 0.3344781, 0.2990135, 0.2658073, 0.2350522, 0.206853, 0.1812354,
               0.1581577, 0.1375251]


def test_schedule_is_the_official_one():
    s = sampling.schedule(50, "cpu")
    assert s.shape == (51,) and s.dtype == torch.float32
    assert s[0].item() == 1.0 and s[-1].item() == 0.0
    assert torch.allclose(s[:-1], torch.tensor(OFFICIAL_50), atol=2e-7, rtol=0)


@pytest.mark.parametrize("steps", [1, 8, 30, 50, 200])
def test_schedule_falls_from_one_to_zero(steps):
    s = sampling.schedule(steps, "cpu")
    assert s.shape == (steps + 1,) and s[0] == 1 and s[-1] == 0 and bool((s[1:] < s[:-1]).all())


def test_noise_is_manual_seed_then_randn():
    torch.manual_seed(1234)
    want = torch.randn([3, 64, 17])
    assert torch.equal(sampling.noise(1234, 3, 17, "cpu"), want)


def _pair(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn((2, 64, 40), generator=g), torch.randn((2, 64, 40), generator=g)


def test_projection_splits_v0_per_batch_item():
    v0, v1 = _pair()
    parallel, orthogonal = sampling.apg_project(v0, v1)
    assert torch.allclose(parallel + orthogonal, v0, atol=1e-6)
    dots = (orthogonal.double() * v1.double()).sum(dim=[-1, -2])
    assert dots.abs().max() < 1e-4
    alone = sampling.apg_project(v0[1:], v1[1:])[1]
    assert torch.equal(alone, orthogonal[1:])


def test_guidance_modes():
    cond, uncond = _pair(1)
    assert torch.allclose(sampling.guide(cond, uncond, 6.0, 0.0), cond + 5.0 * (cond - uncond))
    full = sampling.guide(cond, uncond, 6.0, 1.0)
    assert ((full - cond).double() * cond.double()).sum(dim=[-1, -2]).abs().max() < 1e-3  # only the orthogonal part is added
    half = sampling.guide(cond, uncond, 6.0, 0.5)
    assert torch.allclose(half, 0.5 * full + 0.5 * sampling.guide(cond, uncond, 6.0, 0.0), atol=1e-5)
    assert torch.equal(sampling.guide(cond, uncond, 1.0, 1.0), cond)
