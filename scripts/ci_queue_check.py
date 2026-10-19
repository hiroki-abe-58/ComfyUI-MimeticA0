"""Real-queue check for CI, for the GPU verification and for a clean-install test: start a real ComfyUI server with this node pack
installed, and run the shipped workflows through POST /prompt with the real model.

    python scripts/ci_queue_check.py --comfyui <ComfyUI checkout> --models-dir <ComfyUI-style models folder> [--gpu]
                                     [--workdir DIR] [--base-dir DIR]

--models-dir holds mimetic_a0/mimetic-a0/ and text_encoders/t5_base.safetensors (tools/download_model.py). On the CPU (CI) the
jobs are shortened (4 steps, one bar, two takes, fp32 weights); --gpu runs the workflows as shipped. Through the queue
- each saved FLAC equals the same nodes run in this process on the same device, after the server has stopped, to within one
  16-bit step (ComfyUI's FLAC is 16 bits): text_to_music (one take) and four_takes (every take);
- the same job again gives the same file, another seed other audio; noise "cpu" gives the cpu-drawn take;
- invalid inputs are rejected.
The comparison with the official stable-audio-tools code is in docs/VERIFICATION.md (scripts outside this repository: the
official code pins torch 2.7.1, numpy<2 and transformers<5, which a ComfyUI environment does not have).
The server is started and stopped by this script only. --base-dir reuses a ComfyUI base directory that already has the pack in
custom_nodes (clean-install test); otherwise a fresh one is made in the work directory with a copy of this checkout.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import queue_client  # noqa: E402

NODE_IDS = {"MimeticA0Loader", "MimeticA0Prompt", "MimeticA0Sampler", "MimeticA0Decode"}
STEP = 1 / 32768  # one 16-bit step
CPU_OVERRIDES = {"text_to_music": {"5.steps": 4, "3.bars": "1", "3.bpm": 120.0},
                 "four_takes": {"4.steps": 4, "4.batch_size": 2, "1.weights": "fp32", "4.seconds": 1.0}}


def wait_up(server: str, proc: subprocess.Popen, timeout: float = 300) -> None:
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            raise SystemExit(f"ComfyUI exited early with code {proc.returncode}")
        try:
            urllib.request.urlopen(f"{server}/system_stats", timeout=3).read()
            return
        except Exception:
            time.sleep(2)
    raise SystemExit("ComfyUI did not start in time")


def references(comfy: Path, models: Path, gpu: bool, jobs: dict) -> dict:
    """The audio of each job from the pack's nodes in this process, as ComfyUI's executor runs them (inference mode)."""
    sys.path.insert(0, str(comfy))
    sys.path.insert(0, str(ROOT))
    import comfy.options

    comfy.options.enable_args_parsing(False)
    import comfy.cli_args

    comfy.cli_args.args.cpu = not gpu
    import comfy.sd
    import torch

    from mimetic_comfy import model as mimetic
    from mimetic_comfy import nodes

    def out(result):
        return result.result if hasattr(result, "result") else result

    clip = comfy.sd.load_clip(ckpt_paths=[str(models / "text_encoders" / "t5_base.safetensors")], clip_type=comfy.sd.CLIPType.STABLE_AUDIO)
    loaded, audio = {}, {}
    with torch.inference_mode():
        for tag, (name, wf) in jobs.items():
            sampler = next(n["inputs"] for n in wf.values() if n["class_type"] == "MimeticA0Sampler")
            loader = next(n["inputs"] for n in wf.values() if n["class_type"] == "MimeticA0Loader")
            prompt_node = next((n["inputs"] for n in wf.values() if n["class_type"] == "MimeticA0Prompt"), None)
            text = next(n["inputs"]["text"] for n in wf.values() if n["class_type"] == "CLIPTextEncode")
            seconds = sampler["seconds"]
            if prompt_node is not None:
                text, seconds = out(nodes.MimeticA0Prompt.execute(prompt_node["prompt"], prompt_node["bpm"], prompt_node["bars"], prompt_node["seconds"]))
            if loader["weights"] not in loaded:
                loaded[loader["weights"]] = mimetic.load(str(models / "mimetic_a0" / loader["model"]), loader["weights"])
            patcher, vae = loaded[loader["weights"]]
            cond = clip.encode_from_tokens_scheduled(clip.tokenize(text))
            latent = out(nodes.MimeticA0Sampler.execute(patcher, cond, seconds, sampler["batch_size"], sampler["seed"], sampler["steps"], sampler["cfg"],
                                                        sampler["apg_scale"], sampler["noise"]))[0]
            decoded = out(nodes.MimeticA0Decode.execute(latent, vae, True))[0]
            audio[tag] = decoded["waveform"].numpy()
    return audio


def main(argv=None) -> int:
    import soundfile as sf

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--comfyui", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True, help="a models folder holding mimetic_a0/ and text_encoders/")
    parser.add_argument("--gpu", action="store_true", help="run ComfyUI and the reference on CUDA with the shipped settings (default: --cpu, shortened)")
    parser.add_argument("--port", type=int, default=8199)
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--base-dir", type=Path, default=None, help="a ComfyUI base directory with the pack installed (clean-install test)")
    args = parser.parse_args(argv)

    comfy = args.comfyui.resolve()
    models = args.models_dir.resolve()
    work = (args.workdir or Path(tempfile.mkdtemp(prefix="mimetic-ci-"))).resolve()
    work.mkdir(parents=True, exist_ok=True)
    base = (args.base_dir or work / "base").resolve()
    pack = base / "custom_nodes" / "ComfyUI-MimeticA0"
    if args.base_dir is None:
        if pack.exists():
            shutil.rmtree(pack)
        shutil.copytree(ROOT, pack, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", ".ruff_cache", "outputs"))
    elif not pack.is_dir():
        raise SystemExit(f"--base-dir: {pack} is not installed")
    output = base / "output"
    yaml = work / "extra_model_paths.yaml"
    yaml.write_text(f"mimetic_ci:\n  base_path: {models.as_posix()}\n  mimetic_a0: mimetic_a0\n  text_encoders: text_encoders\n", encoding="utf-8")
    shipped = {n: json.loads((ROOT / "workflows" / "api" / f"{n}.json").read_text(encoding="utf-8")) for n in ("text_to_music", "four_takes")}

    def job(name: str, tag: str, **overrides) -> dict:
        wf = json.loads(json.dumps(shipped[name]))
        for key, value in {**({} if args.gpu else CPU_OVERRIDES[name]), **overrides}.items():
            node, _, field = key.partition(".")
            wf[node]["inputs"][field] = value
        for node in wf.values():
            if "filename_prefix" in node["inputs"]:
                node["inputs"]["filename_prefix"] = f"ci/{tag}"
        return wf

    jobs = {"loop_a": ("text_to_music", job("text_to_music", "loop_a")),
            "takes": ("four_takes", job("four_takes", "takes")),
            "loop_seed2": ("text_to_music", job("text_to_music", "loop_seed2", **{"5.seed": 2})),
            "loop_cpu_noise": ("text_to_music", job("text_to_music", "loop_cpu_noise", **{"5.noise": "cpu"})),
            "loop_b": ("text_to_music", job("text_to_music", "loop_b"))}
    invalid = {"bad_model": job("text_to_music", "bad_model", **{"1.model": "../mimetic-a0"}),
               "too_long": job("text_to_music", "too_long", **{"3.bars": "off", "3.seconds": 20.0}),
               "three_bars": job("text_to_music", "three_bars", **{"3.bars": "3"}),
               "low_cfg": job("text_to_music", "low_cfg", **{"5.cfg": 0.5})}
    for wf in invalid.values():
        del wf["8"]  # with the prompt preview as a second output, ComfyUI would run that and report success

    log = (work / "comfyui.log").open("w", encoding="utf-8")
    server = f"http://127.0.0.1:{args.port}"
    cmd = [sys.executable, "main.py", "--listen", "127.0.0.1", "--port", str(args.port), "--disable-auto-launch", "--disable-api-nodes",
           "--base-directory", str(base), "--extra-model-paths-config", str(yaml)]
    proc = subprocess.Popen(cmd + ([] if args.gpu else ["--cpu"]), cwd=comfy, stdout=log, stderr=subprocess.STDOUT)
    failures, prompts = [], {}
    try:
        wait_up(server, proc)
        info = json.loads(urllib.request.urlopen(f"{server}/object_info", timeout=30).read())
        print("registered:", sorted(NODE_IDS & set(info)))
        if NODE_IDS - set(info):
            raise SystemExit(f"nodes missing from /object_info: {sorted(NODE_IDS - set(info))}")
        for tag, wf in [*((t, w) for t, (_, w) in jobs.items()), *invalid.items()]:
            expect = tag not in invalid
            try:
                result = queue_client.run(server, wf, None, 1800)
            except SystemExit as rejected:  # POST /prompt refused the workflow (validation error)
                result = {"status": "rejected", "wall_seconds": 0, "error": str(rejected)[:300]}
            error = result.get("error") or {}
            print(json.dumps({"run": tag, "status": result["status"], "seconds": result["wall_seconds"], "files": result.get("files"),
                              "error": error.get("exception_message", "")[:200] if isinstance(error, dict) else error}, ensure_ascii=False))
            if (result["status"] == "success") != expect:
                failures.append(f"{tag}: expected {'success' if expect else 'an error'}, got {result['status']} {error}")
            prompts[tag] = result.get("text", {})
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()

    def saved(tag: str):
        files = sorted((output / "ci").glob(f"{tag}_*.flac"))
        data = [sf.read(str(f), dtype="float32", always_2d=True) for f in files]
        if any(rate != 48000 for _, rate in data):
            failures.append(f"{tag}: not 48 kHz")
        return [d.T for d, _ in data]  # (channels, samples) per take

    expected_prompt = "boom bap drum loop, standard kit, swung, with ghost notes, 90 BPM, 4 bars" if args.gpu else \
        "boom bap drum loop, standard kit, swung, with ghost notes, 120 BPM, 1 bar"
    if [v[0] for v in prompts.get("loop_a", {}).values()] != [expected_prompt]:
        failures.append(f"prompt sent {prompts.get('loop_a')} != {expected_prompt!r}")
    ref = references(comfy, models, args.gpu, jobs)
    got = {tag: saved(tag) for tag in jobs}
    for tag, takes in got.items():
        want = ref[tag]
        if len(takes) != want.shape[0]:
            failures.append(f"{tag}: {len(takes)} files for {want.shape[0]} takes")
            continue
        for i, take in enumerate(takes):
            diff = float(np.abs(take - np.clip(want[i], -1, 1)).max()) if take.shape == want[i].shape else None
            print(json.dumps({"compare": f"{tag}[{i}]", "samples": int(take.shape[-1]), "peak": round(float(np.abs(take).max()), 4),
                              "max_diff_in_16bit_steps": None if diff is None else round(diff / STEP, 3)}))
            if diff is None or diff > STEP:
                failures.append(f"{tag}[{i}]: saved FLAC differs from the in-process nodes ({diff}, {take.shape} vs {want[i].shape})")
    if got["loop_a"] and got["loop_b"] and not np.array_equal(got["loop_a"][0], got["loop_b"][0]):
        failures.append("the same job gave different audio")
    if got["loop_a"] and got["loop_seed2"] and np.array_equal(got["loop_a"][0], got["loop_seed2"][0]):
        failures.append("seeds 1 and 2 gave the same audio")
    if got["loop_a"] and got["loop_cpu_noise"] and np.array_equal(got["loop_a"][0], got["loop_cpu_noise"][0]) == args.gpu:
        failures.append("noise official and cpu should differ on CUDA and agree on the CPU")
    if failures:
        print("FAILED:\n  " + "\n  ".join(failures))
        print((work / "comfyui.log").read_text(encoding="utf-8", errors="replace")[-4000:])
        return 1
    print("queue check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
