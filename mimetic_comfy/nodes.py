"""ComfyUI nodes for mimetic-a0, a text-to-music model trained only on CC0 and synthetic audio (nothing is downloaded at run time)."""

from __future__ import annotations

from comfy_api.latest import ComfyExtension, io

from . import paths, timing

NOISE = ("official", "cpu")


class MimeticA0Loader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MimeticA0Loader",
            display_name="mimetic-a0 Loader",
            category="loaders/mimetic_a0",
            description=("Loads mimetic-a0 (text to music, 48 kHz stereo, up to 16 s) from models/mimetic_a0/<name>/, the release files of the "
                         "Hugging Face repository micknoisemachine/mimetic-a0. The prompt goes through a CLIPLoader with t5_base.safetensors, "
                         "type stable_audio."),
            inputs=[
                io.Combo.Input("model", options=paths.model_names(), tooltip="a folder in models/mimetic_a0/"),
                io.Combo.Input("weights", options=["fp32", "fp16"], default="fp32",
                               tooltip="fp32: model.safetensors (2 GB). fp16: model.fp16.safetensors (1 GB), faster on NVIDIA GPUs; a seed gives a slightly different take"),
            ],
            outputs=[io.Model.Output(display_name="model"), io.Vae.Output(display_name="vae")],
        )

    @classmethod
    def execute(cls, model, weights) -> io.NodeOutput:
        from . import model as mimetic

        if weights not in mimetic.WEIGHTS:
            raise ValueError(f"unknown weights {weights!r}")
        patcher, vae = mimetic.load(paths.model_dir(model), weights)
        return io.NodeOutput(patcher, vae)


class MimeticA0Prompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MimeticA0Prompt",
            display_name="mimetic-a0 Prompt (BPM and bars)",
            category="audio/mimetic_a0",
            description=("The command line's --bpm and --bars: adds \"N BPM\" and \"N bars\" to the prompt if it does not have them, and with "
                         "both set the length is exactly that many bars of 4/4. Otherwise the length is `seconds`."),
            inputs=[
                io.String.Input("prompt", multiline=True, default="boom bap drum loop, standard kit, swung, with ghost notes"),
                io.Float.Input("bpm", default=90.0, min=0.0, max=300.0, step=0.1, tooltip="0: no tempo clause"),
                io.Combo.Input("bars", options=["off", "1", "2", "4"], default="4", tooltip="the model knows 1, 2 or 4 bars of 4/4"),
                io.Float.Input("seconds", default=8.0, min=0.1, max=timing.MAX_SECONDS, step=0.01, tooltip="the length when bpm or bars is off"),
            ],
            outputs=[io.String.Output(display_name="prompt"), io.Float.Output(display_name="seconds")],
        )

    @classmethod
    def execute(cls, prompt, bpm, bars, seconds) -> io.NodeOutput:
        bars = 0 if bars == "off" else int(bars)
        if bars and bars not in timing.BARS:
            raise ValueError(f"bars must be one of {timing.BARS}")
        length = timing.bars_to_seconds(bars, bpm) if bpm and bars else seconds
        return io.NodeOutput(timing.add_clauses(prompt, bpm, bars), length)


class MimeticA0Sampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MimeticA0Sampler",
            display_name="mimetic-a0 Sampler",
            category="audio/mimetic_a0",
            description=("Samples like the official tools: Euler over the rectified-flow schedule spaced evenly in log-SNR, classifier-free "
                         "guidance against the zeroed prompt with adaptive projected guidance, the length plus 6 s of headroom, and the "
                         "starting noise drawn as the official code does, so a seed gives the same take."),
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("positive", tooltip="CLIPTextEncode with the T5-base text encoder (CLIPLoader type stable_audio)"),
                io.Float.Input("seconds", default=8.0, min=0.1, max=timing.MAX_SECONDS, step=0.01, tooltip="at most 16 s"),
                io.Int.Input("batch_size", default=1, min=1, max=16, tooltip="takes; one batch of noise, as -n in the command line"),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
                io.Int.Input("steps", default=50, min=1, max=500, tooltip="50 is what the model was tested with"),
                io.Float.Input("cfg", default=6.0, min=1.0, max=20.0, step=0.1, tooltip="6 is what the model was tested with; 1 turns guidance off"),
                io.Float.Input("apg_scale", default=1.0, min=0.0, max=1.0, step=0.01,
                               tooltip="1: adaptive projected guidance (the official default), 0: plain CFG, in between: a blend"),
                io.Combo.Input("noise", options=list(NOISE), default="official",
                               tooltip=("official: drawn on the sampling device like the official code (same seed, same take on the same kind "
                                        "of device). cpu: drawn on the CPU, the same on every machine")),
            ],
            outputs=[io.Latent.Output(display_name="latent")],
        )

    @classmethod
    def execute(cls, model, positive, seconds, batch_size, seed, steps, cfg, apg_scale, noise) -> io.NodeOutput:
        import comfy.model_management as mm
        import comfy.samplers
        import comfy.utils
        import latent_preview
        import torch

        from . import model as mimetic
        from . import sampling

        if noise not in NOISE:
            raise ValueError(f"unknown noise {noise!r}")
        if not isinstance(model.model, mimetic.MimeticA0):
            raise ValueError("connect the model output of a mimetic-a0 Loader")
        seconds = timing.clamp_seconds(seconds)
        length = timing.latent_length(seconds)
        device = model.load_device
        start = sampling.noise(seed, batch_size, length, device if noise == "official" else torch.device("cpu"))
        sigmas = sampling.schedule(steps, device)
        guider = mimetic.Guider(model, cfg, apg_scale)
        guider.set_conds(*mimetic.conditioning(positive, seconds))
        latent = torch.zeros((batch_size, 64, length))
        callback = latent_preview.prepare_callback(model, steps)
        with mimetic.official_precision():
            samples = guider.sample(start, latent, comfy.samplers.sampler_object("euler"), sigmas, callback=callback, disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=seed)
        return io.NodeOutput({"samples": samples.to(mm.intermediate_device()), "type": "audio", "mimetic_seconds": seconds})


class MimeticA0Decode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MimeticA0Decode",
            display_name="mimetic-a0 Decode",
            category="audio/mimetic_a0",
            description=("Decodes each take on its own and trims it to the sampler's length, as mimetic_a0.generate() does. Unlike VAE Decode "
                         "Audio, it does not rescale the volume."),
            inputs=[
                io.Latent.Input("samples"),
                io.Vae.Input("vae"),
                io.Boolean.Input("normalize", default=True,
                                 tooltip="on: each take peaks at -1 dBFS, like the command line's WAV files. off: the raw output, which can go past full scale"),
            ],
            outputs=[io.Audio.Output(display_name="audio")],
        )

    @classmethod
    def execute(cls, samples, vae, normalize) -> io.NodeOutput:
        import torch

        from . import model as mimetic

        latent = samples["samples"]
        with mimetic.official_precision():
            audio = torch.cat([vae.decode(latent[i:i + 1]).movedim(-1, 1).float() for i in range(latent.shape[0])])
        if "mimetic_seconds" in samples:
            audio = audio[..., : timing.output_samples(samples["mimetic_seconds"])]
        if normalize:  # mimetic_a0.peak_normalise
            audio = audio / audio.abs().amax(dim=[1, 2], keepdim=True).clamp(min=1e-6) * 0.891
        return io.NodeOutput({"waveform": audio.cpu(), "sample_rate": timing.SAMPLE_RATE})


class MimeticA0Extension(ComfyExtension):
    async def get_node_list(self):
        return [MimeticA0Loader, MimeticA0Prompt, MimeticA0Sampler, MimeticA0Decode]


async def comfy_entrypoint() -> MimeticA0Extension:
    paths.register()
    return MimeticA0Extension()
