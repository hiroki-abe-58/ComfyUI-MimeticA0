"""The nodes in ComfyUI with the real model on the CPU (few steps, short takes). Runs as ComfyUI's executor does: in inference mode."""

import asyncio

import pytest
import torch

from mimetic_comfy import timing


def out(result):
    return result.result if hasattr(result, "result") else result


@pytest.fixture(autouse=True)
def inference():
    with torch.inference_mode():
        yield


def test_extension_lists_the_nodes(comfy_dir):
    from mimetic_comfy import nodes

    ext = asyncio.run(nodes.comfy_entrypoint())
    ids = [n.define_schema().node_id for n in asyncio.run(ext.get_node_list())]
    assert ids == ["MimeticA0Loader", "MimeticA0Prompt", "MimeticA0Sampler", "MimeticA0Decode"]


def test_prompt_node(comfy_dir):
    from mimetic_comfy import nodes

    assert out(nodes.MimeticA0Prompt.execute("house drum loop", 124.0, "4", 8.0)) == ("house drum loop, 124 BPM, 4 bars", timing.bars_to_seconds(4, 124))
    assert out(nodes.MimeticA0Prompt.execute("single cello note, G3", 0.0, "off", 3.0)) == ("single cello note, G3", 3.0)
    assert out(nodes.MimeticA0Prompt.execute("pads", 90.0, "off", 5.0)) == ("pads, 90 BPM", 5.0)


def test_loader_builds_the_model_from_model_config(loaded):
    from mimetic_comfy import model

    patcher, vae = loaded
    m = patcher.model
    assert isinstance(m, model.MimeticA0)
    assert m.model_type.name == "FLOW" and m.model_sampling.timestep(torch.tensor(0.25)).item() == 0.25
    dit = m.diffusion_model
    assert len(dit.transformer.layers) == 16 and dit.transformer.layers[0].self_attn.num_heads == 16 and dit.transformer.dim == 1024
    assert (m.seconds_start_embedder.max_val, m.seconds_total_embedder.max_val) == (32, 32)
    assert vae.audio_sample_rate == 48000 and vae.downscale_ratio == timing.DOWNSAMPLING


def test_null_branch_zeroes_the_cross_attention(loaded, t5):
    m = loaded[0].model
    cond = t5.encode_from_tokens_scheduled(t5.tokenize("single cello note, G3"))[0][0]
    kw = {"device": torch.device("cpu"), "seconds_start": 0.0, "seconds_total": 3.0, "cross_attn": cond, "noise": None}
    pos, neg = m.extra_conds(**kw), m.extra_conds(**kw, mimetic_null=True)
    assert pos["c_crossattn"].cond.shape == (1, cond.shape[1] + 2, 768)
    assert neg["c_crossattn"].cond.abs().sum() == 0
    assert torch.equal(pos["global_embed"].cond, neg["global_embed"].cond) and pos["global_embed"].cond.shape == (1, 1536)


@pytest.fixture(scope="module")
def takes(loaded, t5):
    from mimetic_comfy import nodes

    patcher, vae = loaded
    cond = t5.encode_from_tokens_scheduled(t5.tokenize("grand piano chord, Cmaj7/E, rolled, first inversion, soft"))
    with torch.inference_mode():
        latent = out(nodes.MimeticA0Sampler.execute(patcher, cond, 0.5, 2, 7, 3, 6.0, 1.0, "official"))[0]
        audio = out(nodes.MimeticA0Decode.execute(latent, vae, True))[0]
        raw = out(nodes.MimeticA0Decode.execute(latent, vae, False))[0]
        again = out(nodes.MimeticA0Sampler.execute(patcher, cond, 0.5, 2, 7, 3, 6.0, 1.0, "cpu"))[0]
        other = out(nodes.MimeticA0Sampler.execute(patcher, cond, 0.5, 2, 8, 3, 6.0, 1.0, "official"))[0]
    return {"latent": latent, "audio": audio, "raw": raw, "again": again, "other": other}


def test_sampler_latent(takes):
    latent = takes["latent"]
    assert latent["samples"].shape == (2, 64, timing.latent_length(0.5)) and latent["mimetic_seconds"] == 0.5 and latent["type"] == "audio"
    assert torch.isfinite(latent["samples"]).all()


def test_noise_on_the_cpu_is_the_official_noise_there(takes):
    assert torch.equal(takes["latent"]["samples"], takes["again"]["samples"])  # on the CPU both draw the same numbers
    assert not torch.equal(takes["latent"]["samples"], takes["other"]["samples"])


def test_decode_trims_and_normalizes_each_take(takes):
    audio, raw = takes["audio"], takes["raw"]
    assert audio["sample_rate"] == 48000 and audio["waveform"].shape == (2, 2, timing.output_samples(0.5))
    assert torch.allclose(audio["waveform"].abs().amax(dim=[1, 2]), torch.full((2,), 0.891), atol=1e-6)
    peak = raw["waveform"].abs().amax(dim=[1, 2], keepdim=True)
    assert torch.allclose(audio["waveform"], raw["waveform"] / peak * 0.891, atol=1e-6)


def test_precision_flags_are_restored():
    from mimetic_comfy import model

    before = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark)
    torch.backends.cudnn.allow_tf32 = True
    with model.official_precision():
        assert not torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.benchmark
    assert torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = before[1]


def test_fp16_weights(comfy_dir, release_dir):
    from mimetic_comfy import model

    if not (release_dir / "model.fp16.safetensors").is_file():
        pytest.skip("model.fp16.safetensors not downloaded (tools/download_model.py --weights all)")
    patcher, vae = model.load(str(release_dir), "fp16")
    assert patcher.model.diffusion_model.transformer.layers[0].self_attn.to_qkv.weight.dtype == torch.float16


def test_sampler_refuses_another_model(comfy_dir):
    from mimetic_comfy import nodes

    class Other:
        model = object()

    with pytest.raises(ValueError, match="mimetic-a0 Loader"):
        nodes.MimeticA0Sampler.execute(Other(), [], 1.0, 1, 0, 1, 6.0, 1.0, "official")
