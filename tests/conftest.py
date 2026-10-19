"""Test setup. References come from environment variables, never from this repository:

MIMETIC_MODELS_DIR    a ComfyUI-style models folder holding mimetic_a0/mimetic-a0/ and text_encoders/t5_base.safetensors
                      (tools/download_model.py fetches them; 2.9 GB with the fp32 weights)
MIMETIC_COMFYUI_DIR   ComfyUI checkout (v0.38.0 or later)
MIMETIC_REQUIRE_REFERENCES=1  fail instead of skip when one of them is missing (CI sets this)
test_timing.py and test_sampling.py need neither. The node tests run the real model on the CPU with few steps. The comparison with
the official stable-audio-tools code (which pins torch 2.7.1, numpy<2 and transformers<5) is in docs/VERIFICATION.md.
"""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
REQUIRE = os.environ.get("MIMETIC_REQUIRE_REFERENCES") == "1"


def _path(var):
    p = os.environ.get(var)
    return p if p and Path(p).exists() else None


COMFY = _path("MIMETIC_COMFYUI_DIR")
MODELS = _path("MIMETIC_MODELS_DIR")
if COMFY:
    sys.path.insert(1, COMFY)
    import comfy.options

    comfy.options.enable_args_parsing(False)
    import comfy.cli_args

    comfy.cli_args.args.cpu = True


def _need(value, what):
    if value:
        return value
    if REQUIRE:
        pytest.fail(f"{what} is required (MIMETIC_REQUIRE_REFERENCES=1)")
    pytest.skip(f"{what} not available")


@pytest.fixture(scope="session")
def comfy_dir():
    return _need(COMFY, "ComfyUI checkout (MIMETIC_COMFYUI_DIR)")


@pytest.fixture(scope="session")
def models_dir():
    """The models folder (holding mimetic_a0/ and text_encoders/)."""
    return Path(_need(MODELS, "model files (MIMETIC_MODELS_DIR)"))


@pytest.fixture(scope="session")
def release_dir(models_dir):
    return models_dir / "mimetic_a0" / "mimetic-a0"


@pytest.fixture(scope="session")
def loaded(comfy_dir, release_dir):
    """(ModelPatcher, VAE) of the fp32 weights, loaded once."""
    from mimetic_comfy import model

    return model.load(str(release_dir), "fp32")


@pytest.fixture(scope="session")
def t5(comfy_dir, models_dir):
    import comfy.sd

    return comfy.sd.load_clip(ckpt_paths=[str(models_dir / "text_encoders" / "t5_base.safetensors")], clip_type=comfy.sd.CLIPType.STABLE_AUDIO)
