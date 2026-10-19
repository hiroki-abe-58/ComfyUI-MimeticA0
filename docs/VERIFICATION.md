# Verification

What was checked before the first release (v0.1.0), and how. Local runs: Windows 11, RTX 5090 (32 GB), ComfyUI v0.42.1 with
torch 2.14.1+cu130. The scripts that compare with the official code live outside this repository, because the official code pins
torch 2.7.1, numpy<2 and transformers<5, which a ComfyUI environment does not have.

## Versions

| | |
| --- | --- |
| Model | micknoisemachine/mimetic-a0 at 5f21e1807517cda2e6d72051cdb81dafc3931077 (weights checked against the release's SHA256SUMS) |
| Text encoder | google-t5/t5-base at a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1 (`model.safetensors`) |
| Official code | micknoise/mimetic-a0 9f91b67 with stable-audio-tools 3241adb, `uv sync` (Python 3.11, transformers 4.57.6, numpy 1.26.4), then torch 2.7.1+cu128 as its README says for RTX 50xx |
| Official runs | `mimetic_a0.generate()` (the audio), `generate_diffusion_cond(..., return_latents=True)` (the latents), the conditioner's tensors, the starting noise |

## The official reference

Seven cases, chosen to cover what changes the computation: lengths from 1.5 s to the 16 s cap (the latent length depends on it),
fp32 and fp16 weights, a batch of two takes, CFG 6 with full APG (the default), APG 0.5, plain CFG 4 with 30 steps, and no guidance
(CFG 1, 20 steps).

| case | prompt | seconds | seed | takes | steps | cfg | apg | weights |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| boombap | boom bap drum loop, standard kit, swung, with ghost notes, 90 BPM, 4 bars | 10.67 | 1 | 1 | 50 | 6 | 1 | fp32 |
| cello | single cello note, G3 | 3 | 1 | 1 | 50 | 6 | 1 | fp32 |
| piano_n2 | grand piano chord, Cmaj7/E, rolled, first inversion, soft | 4 | 7 | 2 | 50 | 6 | 1 | fp32 |
| house_fp16 | house drum loop, dance kit, open hi-hats, 124 BPM, 4 bars | 7.74 | 3 | 1 | 50 | 6 | 1 | fp16 |
| lead_16s_cfg | analog mono lead synth melody, over Am - F - C - G, laid-back, 110 BPM, A minor, 4 bars | 16 | 5 | 1 | 30 | 4 | 0 | fp32 |
| bass_apg_half | finger bass bassline, over Cm - Fm - G7, sparse, 95 BPM, C minor, 4 bars | 10.11 | 11 | 1 | 50 | 6 | 0.5 | fp32 |
| kick_nocfg | 808 kit snare hit, one-shot, hard | 1.5 | 2 | 1 | 20 | 1 | 1 | fp32 |

The official code is deterministic on CUDA: every case run twice gave the same audio, and decoding the saved latents gives exactly
the returned audio.

## What the official sampler does (and the pack copies)

Read from the official code and confirmed by the comparisons below:

- **Length**: the model sets `mask_padding_attention`, so the latent covers seconds_total + 6 s, rounded up to a frame of 1920
  samples and capped at 400 frames (16 s); the audio is trimmed to the requested length afterwards. The padding mask then keeps
  every frame for every length (checked for every millisecond from 0.001 to 16 s and for 1, 2 and 4 bars at 40–240 BPM), so the
  pack needs no mask.
- **Noise**: `torch.manual_seed(seed)`, then one `randn` of (takes, 64, frames) on the sampling device.
- **Schedule**: Euler steps on `linspace(1, 0)` passed through stable-audio-tools' default sampling shift,
  `LogSNRShift(rate=0, anchor_logsnr=-6.2, logsnr_end=2.0)`, because model_config.json sets none: steps are evenly spaced in log-SNR
  (t = sigmoid(8.2 t' - 2)), so most of them are near t = 1. A plain linear schedule (my first version) gave takes only 19 dB from
  the official ones.
- **Guidance**: cond and uncond in one batch, the uncond with all cross-attention tokens (prompt and the two number tokens) zeroed
  and the same global embedding; on denoised predictions, `cond + (cfg - 1) * d` where d is the part of `cond - uncond` orthogonal to
  `cond` (APG 1), `cond - uncond` (APG 0) or a blend, projected in float64. `rescale_cfg=True` is passed but not read
  (`scale_phi` stays 0), so there is no CFG rescale.
- **Conditioning**: T5-base (max 128 tokens, padding zeroed), then one token each for seconds_start (0) and seconds_total (the
  length), embedded over 0..32 s; the two numbers side by side are the global embedding.
- **Precision**: `generate_diffusion_cond` turns off TF32 (matmul and cuDNN) and reduced-precision fp16 reductions, and cuDNN
  benchmarking.

## Pack against official

`pack_compare.py` runs the pack's nodes in ComfyUI v0.42.1 (in process, in inference mode as ComfyUI's executor does), once with
ComfyUI's T5-base (the real workflow) and once with the official prompt embedding put in its place (the DiT, guidance, sampler and
decoder alone). SNR against the official tensors; the audio is the raw, trimmed output (Decode with normalize off).

| case | weights | T5 embedding | latents, official embedding | audio, official embedding | largest audio difference | audio, ComfyUI's T5 |
| --- | --- | --- | --- | --- | --- | --- |
| boombap | fp32 | 50.5 dB | 110.8 dB | 103.6 dB | 2.8e-5 | 32.0 dB |
| cello | fp32 | 58.7 dB | 115.0 dB | 116.0 dB | 3.6e-6 | 56.4 dB |
| piano_n2 (2 takes) | fp32 | 57.8 dB | 114.5 dB | 115.3 dB | 2.6e-6 | 56.2 dB |
| house_fp16 | fp16 | 57.6 dB | 41.6 dB | 36.4 dB | 4.3e-2 | 37.3 dB |
| lead_16s_cfg | fp32 | 57.8 dB | 114.3 dB | 107.7 dB | 2.9e-5 | 42.0 dB |
| bass_apg_half | fp32 | 48.8 dB | 109.9 dB | 108.5 dB | 6.0e-6 | 31.6 dB |
| kick_nocfg | fp32 | 57.5 dB | 135.0 dB | 95.3 dB | 7.3e-8 | 88.8 dB |

In every case the starting noise, the latent length, the trimmed length and the schedule are bit-identical to the official ones,
and so are the cross-attention tokens and the global embedding the DiT gets (one case 9.5e-7 apart).

- **fp32, same embedding**: 95–116 dB, the rounding of two PyTorch builds (2.7.1 against 2.14.1). One DiT forward on identical
  inputs agrees to 119–130 dB (largest difference 2.5e-6); step by step through 50 steps the latents stay 117–155 dB apart.
- **The text encoder**: ComfyUI runs T5-base with fp32 compute; the official code runs it in fp16 (autocast). The official recipe
  itself, run with transformers 5.20.1 and torch 2.14.1, is 55.5–56.3 dB from the official embedding, so the official fp16
  embedding depends on the library versions. Through 50 guided steps the 49–59 dB embedding difference becomes 32–89 dB in the
  audio (more with stronger guidance and longer takes). The official tokenizer call truncates at 128 tokens; ComfyUI's T5-base
  tokenizer pads to at least 128 and does not truncate, so the two agree for prompts up to 128 tokens (all seven cases) and differ
  beyond.
- **fp16**: the official code halves the whole model and keeps the sampler's state in fp16; ComfyUI keeps the state in fp32 and
  computes in fp16. 36 dB to the official fp16 take; the official fp16 and fp32 takes of the same seed are 19.1 dB apart.
- **TF32**: with PyTorch's default (cuDNN convolutions in TF32) one DiT forward was 68 dB from the official one and whole takes
  61–64 dB (boombap, cello, lead_16s_cfg, bass_apg_half; official embedding) instead of 110–115 dB. The nodes turn TF32 off while
  they sample and decode, as the official code does, and restore the previous flags afterwards.

## Real ComfyUI queue

`scripts/ci_queue_check.py --gpu` (ComfyUI v0.42.1, RTX 5090, the shipped workflows through `POST /prompt`):

- text_to_music (10.7 s, one take) and four_takes (four 3 s takes, fp16 weights) saved FLAC files equal to the same nodes run in
  process to within half a 16-bit step, every take peaking at -1 dBFS;
- the same job again gave the same file, seed 2 other audio, `noise = cpu` another take than `official` on CUDA;
- a model name outside the list, 20 s, "3" bars and CFG 0.5 were rejected by the server;
- the prompt preview showed "boom bap drum loop, standard kit, swung, with ghost notes, 90 BPM, 4 bars";
- 3.2 s for the first job (with loading), 1.6 s afterwards for a 10.7 s take; four 3 s takes in 2.6 s.

The workflows load in ComfyUI's frontend (1.55.18) without missing nodes, and `graphToPrompt()` gives back the same graph
(`scripts/export_gui_workflows.py`).

## Clean install

A fresh ComfyUI v0.42.1 checkout and virtual environment, `git clone` of this repository into `custom_nodes`, its (empty)
requirements, `tools/download_model.py --weights all` from Hugging Face (8 files, all SHA-256 verified), then the GPU queue check
against that installation (`--base-dir`): passed, the eight saved FLAC files byte-identical to the earlier run, the clone unchanged.
The Registry package (`comfy node validate` and `comfy node pack`, comfy-cli 1.22.0, on `git archive` of the release commit) has
only the package files (no tests, scripts or images) and none of the patterns the Registry's scan flags.

## Tests and CI

`tests/` (43 tests): lengths against the official runs and the padding-mask property, the prompt clauses, the schedule against the
official 50-step values, the noise recipe, the guidance formulas, and the nodes with the real model on the CPU (loading from
model_config.json, the zeroed uncond, takes, trimming, normalising, the cpu/official noise, the restored precision flags). They pass
with ComfyUI v0.38.0 and v0.42.1. CI runs them on Ubuntu and Windows with CPU torch, plus the real-queue check (shortened on the CPU),
with the model files downloaded by `tools/download_model.py` and checked against their SHA-256.
