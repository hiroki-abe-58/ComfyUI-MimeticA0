"""mimetic-a0 on ComfyUI's own stable-audio classes: the DiT is comfy.ldm.audio.dit.AudioDiffusionTransformer, the autoencoder
ComfyUI's AudioOobleckVAE (the 48 kHz ACE-Step 1.5 layout), and the prompt goes through the T5-base text encoder of a CLIPLoader
of type stable_audio. Loaded as a Stable Audio Open checkpoint, three things would be wrong: mimetic-a0 is a rectified-flow
model (not v-prediction), its DiT is 1024 wide with 16 layers and 16 heads, and its seconds_start / seconds_total embedders
cover 0..32 s instead of 0..512 s. The numbers come from the release's model_config.json.
"""

from __future__ import annotations

import contextlib
import json
import os

import comfy.conds
import comfy.latent_formats
import comfy.ldm.audio.dit
import comfy.ldm.audio.embedders
import comfy.model_base
import comfy.model_detection
import comfy.model_management as mm
import comfy.model_patcher
import comfy.samplers
import comfy.sd
import comfy.supported_models
import comfy.utils
import torch

from . import sampling, timing

PREFIX = "model.model."
WEIGHTS = {"fp32": "model.safetensors", "fp16": "model.fp16.safetensors"}
DTYPES = {"fp32": torch.float32, "fp16": torch.float16}


class LatentFormat(comfy.latent_formats.StableAudio1):
    temporal_downscale_ratio = timing.DOWNSAMPLING


class MimeticA0(comfy.model_base.BaseModel):
    def __init__(self, model_config, seconds_start, seconds_total, device=None):
        super().__init__(model_config, comfy.model_base.ModelType.FLOW, device=device, unet_model=comfy.ldm.audio.dit.AudioDiffusionTransformer)
        self.seconds_start_embedder = seconds_start
        self.seconds_total_embedder = seconds_total

    def extra_conds(self, **kwargs):
        # stable-audio-tools' conditioner: the prompt, then one token each for seconds_start and seconds_total as cross attention,
        # the two numbers side by side as the global embedding. The unconditional branch zeroes all cross-attention tokens.
        device = kwargs["device"]
        start = self.seconds_start_embedder([kwargs["seconds_start"]])[0].to(device)
        total = self.seconds_total_embedder([kwargs["seconds_total"]])[0].to(device)
        out = {"global_embed": comfy.conds.CONDRegular(torch.cat([start, total], dim=-1).reshape((1, -1)))}
        cross_attn = kwargs["cross_attn"].to(device)
        cross_attn = torch.cat([cross_attn, start.repeat((cross_attn.shape[0], 1, 1)), total.repeat((cross_attn.shape[0], 1, 1))], dim=1)
        if kwargs.get("mimetic_null", False):
            cross_attn = torch.zeros_like(cross_attn)
        out["c_crossattn"] = comfy.conds.CONDRegular(cross_attn)
        return out


class Config(comfy.supported_models.StableAudio):
    sampling_settings = {"multiplier": 1.0, "shift": 1.0}
    latent_format = LatentFormat

    def __init__(self, unet_config, number_ranges):
        super().__init__(unet_config)
        self.number_ranges = number_ranges

    def get_model(self, state_dict, prefix="", device=None):
        embedders = []
        for name in ("seconds_start", "seconds_total"):
            low, high = self.number_ranges[name]
            embedder = comfy.ldm.audio.embedders.NumberConditioner(768, min_val=low, max_val=high)
            embedder.load_state_dict(comfy.utils.state_dict_prefix_replace(state_dict, {f"conditioner.conditioners.{name}.": ""}, filter_keys=True))
            embedders.append(embedder.to(self.manual_cast_dtype or self.unet_config["dtype"]))  # stable-audio-tools halves them with the model
        return MimeticA0(self, *embedders, device=device)


def read_config(folder: str) -> dict:
    with open(os.path.join(folder, "model_config.json"), encoding="utf-8") as f:
        config = json.load(f)
    model = config["model"]
    if (config.get("model_type") != "diffusion_cond" or model["diffusion"].get("diffusion_objective") != "rectified_flow" or config.get("sample_rate") != timing.SAMPLE_RATE
            or config.get("sample_size") != timing.SAMPLE_SIZE or model["pretransform"]["config"].get("downsampling_ratio") != timing.DOWNSAMPLING):
        raise ValueError(f"{folder} does not hold the mimetic-a0 release (model_config.json differs)")
    return config


def load(folder: str, weights: str):
    """-> (ModelPatcher, VAE) for the release in `folder`, with the weights of the fp32 or fp16 file."""
    config = read_config(folder)
    sd = comfy.utils.load_torch_file(os.path.join(folder, WEIGHTS[weights]))
    dit = config["model"]["diffusion"]["config"]
    unet_config = comfy.model_detection.detect_unet_config(sd, PREFIX)
    if unet_config is None or unet_config.get("embed_dim") != dit["embed_dim"] or unet_config.get("depth") != dit["depth"]:
        raise ValueError(f"{WEIGHTS[weights]} does not match model_config.json")
    unet_config["num_heads"] = dit["num_heads"]  # not stored in the weights of this DiT variant
    ranges = {c["id"]: (c["config"]["min_val"], c["config"]["max_val"]) for c in config["model"]["conditioning"]["configs"] if c["type"] == "number"}
    model_config = Config(unet_config, ranges)
    dtype = DTYPES[weights]
    load_device = mm.get_torch_device()
    model_config.set_inference_dtype(dtype, mm.unet_manual_cast(dtype, load_device, model_config.supported_inference_dtypes), device=load_device)
    model = model_config.get_model(sd, PREFIX, device=mm.unet_inital_load_device(comfy.utils.calculate_parameters(sd, PREFIX), dtype))
    patcher = comfy.model_patcher.ModelPatcher(model, load_device=load_device, offload_device=mm.unet_offload_device())
    model.load_model_weights(sd, PREFIX)
    # the autoencoder in the weights' dtype like the official model (fp32 on a device that computes fp16 weights in fp32, such as the CPU)
    vae = comfy.sd.VAE(sd=comfy.utils.state_dict_prefix_replace(sd, {"pretransform.model.": ""}, filter_keys=True), dtype=model_config.manual_cast_dtype or dtype)
    return patcher, vae


class Guider(comfy.samplers.CFGGuider):
    """CFGGuider with the official guidance: the negative is the zeroed prompt, combined by sampling.guide()."""

    def __init__(self, model_patcher, cfg: float, apg_scale: float):
        super().__init__(model_patcher)
        self.set_cfg(cfg)
        self.apg_scale = apg_scale

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        if self.cfg == 1.0:
            return comfy.samplers.calc_cond_batch(self.inner_model, [self.conds["positive"]], x, timestep, model_options)[0]
        cond, uncond = comfy.samplers.calc_cond_batch(self.inner_model, [self.conds["positive"], self.conds["negative"]], x, timestep, model_options)
        return sampling.guide(cond, uncond, self.cfg, self.apg_scale)


def conditioning(positive, seconds: float):
    """(positive, negative) for `seconds`; the negative has the same numbers and zeroed cross attention."""
    numbers = {"seconds_start": 0.0, "seconds_total": seconds}
    return [[t, {**d, **numbers}] for t, d in positive], [[t, {**d, **numbers, "mimetic_null": True}] for t, d in positive]


@contextlib.contextmanager
def official_precision():
    """The precision flags generate_diffusion_cond sets before it samples (and never resets). PyTorch's default runs cuDNN
    convolutions in TF32, which on its own changes a 50-step take noticeably; set while the pack samples or decodes, then restored."""
    flags = [(torch.backends.cuda.matmul, "allow_tf32"), (torch.backends.cudnn, "allow_tf32"),
             (torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction"), (torch.backends.cudnn, "benchmark")]
    saved = [getattr(owner, name) for owner, name in flags]
    for owner, name in flags:
        setattr(owner, name, False)
    try:
        yield
    finally:
        for (owner, name), value in zip(flags, saved):
            setattr(owner, name, value)
