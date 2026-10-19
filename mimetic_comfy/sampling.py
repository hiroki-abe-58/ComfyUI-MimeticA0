"""The parts of the official sampler that are not the model, as generate_diffusion_cond runs them for mimetic-a0
(stable-audio-tools 3241adb). Only torch is needed."""

from __future__ import annotations

import torch

LOGSNR_START, LOGSNR_END = -6.2, 2.0


def noise(seed: int, batch: int, length: int, device) -> torch.Tensor:
    """The starting noise: torch.manual_seed(seed), then one randn of (batch, 64, length) on the sampling device."""
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randn((batch, 64, length), generator=generator, device=device)


def schedule(steps: int, device) -> torch.Tensor:
    """build_schedule for rectified flow: linspace(1, 0) through the model's sampling shift. model_config.json sets none, so it is
    stable-audio-tools' default LogSNRShift(rate=0, anchor_logsnr=-6.2, logsnr_end=2.0): steps evenly spaced in log-SNR
    between -6.2 and 2, the same for every length, with the ends kept at exactly 1 and 0."""
    t = torch.linspace(1, 0, steps + 1, device=device)
    shifted = torch.sigmoid(-(LOGSNR_END - t * (LOGSNR_END - LOGSNR_START)))
    shifted = torch.where(t <= 0, torch.zeros_like(shifted), shifted)
    return torch.where(t >= 1, torch.ones_like(shifted), shifted)


def apg_project(v0: torch.Tensor, v1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """v0 split into the parts parallel and orthogonal to v1, per batch item over channels and time, in float64."""
    dtype = v0.dtype
    v0, v1 = v0.double(), v1.double()
    v1 = v1 / v1.norm(dim=[-1, -2], keepdim=True).clamp(min=1e-8)
    parallel = (v0 * v1).sum(dim=[-1, -2], keepdim=True) * v1
    return parallel.to(dtype), (v0 - parallel).to(dtype)


def guide(cond: torch.Tensor, uncond: torch.Tensor, cfg: float, apg_scale: float) -> torch.Tensor:
    """Classifier-free guidance on denoised predictions, as DiffusionTransformer.forward does it here (cfg_norm_threshold 0,
    scale_phi 0, a padding mask that keeps every frame). apg_scale 1 keeps only the part of the guidance that is orthogonal to
    the conditional prediction (adaptive projected guidance), 0 is plain CFG."""
    diff = cond - uncond
    if apg_scale == 0.0:
        cfg_diff = diff
    elif apg_scale == 1.0:
        cfg_diff = apg_project(diff, cond)[1]
    else:
        cfg_diff = apg_scale * apg_project(diff, cond)[1] + (1 - apg_scale) * diff
    return cond + (cfg - 1) * cfg_diff
