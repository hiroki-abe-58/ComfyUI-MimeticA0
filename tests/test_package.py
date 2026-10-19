"""The installed package must not contain code that reads environment variables, opens network connections or starts processes (the
categories of the Comfy Registry's automated scan; scripts/ and tests/ are not part of a Registry package), and the nodes must
never download: only tools/download_model.py, run by the user, talks to Hugging Face. The pack has no Python dependency of its own."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_FILES = [p for p in [ROOT / "__init__.py", *(ROOT / "mimetic_comfy").rglob("*.py"), *(ROOT / "tools").rglob("*.py")]
                 if "__pycache__" not in p.parts]
FORBIDDEN = re.compile(r"os\.environ|getenv|putenv|urllib|requests\.|http\.client|socket\.|subprocess|os\.system|importlib|(?<![.\w])exec\(|(?<![.\w])eval\(|__import__")
NODE_TYPES = {"MimeticA0Loader", "MimeticA0Prompt", "MimeticA0Sampler", "MimeticA0Decode", "CLIPLoader", "CLIPTextEncode", "SaveAudioAdvanced", "PreviewAny"}


def test_package_has_no_environment_network_or_process_code():
    hits = [f"{p.relative_to(ROOT).as_posix()}:{n}: {line.strip()}" for p in PACKAGE_FILES
            for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1) if FORBIDDEN.search(line)]
    assert not hits, "\n".join(hits)


def test_only_the_download_tool_knows_about_the_hub():
    text = "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "mimetic_comfy").rglob("*.py") if "__pycache__" not in p.parts)
    assert not re.search(r"^\s*(from|import)\s+(huggingface_hub|transformers)|hf_hub_download|snapshot_download|from_pretrained", text, re.M)
    assert "hf_hub_download" in (ROOT / "tools" / "download_model.py").read_text(encoding="utf-8")


def test_comfyignore_keeps_tests_and_maintainer_tools_out_of_the_registry_package():
    lines = {line.strip() for line in (ROOT / ".comfyignore").read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")}
    assert {"tests/", "scripts/", ".github/"} <= lines


def test_no_python_dependencies():
    assert re.search(r"^dependencies = \[\]$", (ROOT / "pyproject.toml").read_text(encoding="utf-8"), re.M)
    reqs = [line for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
    assert reqs == []


def test_workflows_use_known_nodes_and_the_downloaded_files():
    for f in sorted((ROOT / "workflows" / "api").glob("*.json")):
        wf = json.loads(f.read_text(encoding="utf-8"))
        assert {n["class_type"] for n in wf.values()} <= NODE_TYPES, f.name
        for n in wf.values():
            if n["class_type"] == "MimeticA0Loader":
                assert n["inputs"]["model"] == "mimetic-a0"
            if n["class_type"] == "CLIPLoader":
                assert n["inputs"] == {"clip_name": "t5_base.safetensors", "type": "stable_audio"}
    assert sorted(p.name for p in (ROOT / "workflows" / "gui").glob("*.json")) == sorted(p.name for p in (ROOT / "workflows" / "api").glob("*.json"))
