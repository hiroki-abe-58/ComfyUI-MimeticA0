"""Download mimetic-a0 and its T5-base text encoder into ComfyUI's models folder, and verify them.

    python tools/download_model.py --models-dir <ComfyUI>/models [--weights fp32|fp16|all] [--verify-only]

Sources (pinned revisions):
  https://huggingface.co/micknoisemachine/mimetic-a0   MIT; model.safetensors 2.0 GB (fp32), model.fp16.safetensors 1.0 GB
  https://huggingface.co/google-t5/t5-base             Apache-2.0; model.safetensors 0.9 GB, saved as t5_base.safetensors
Result:
  models/mimetic_a0/mimetic-a0/model_config.json, model(.fp16).safetensors, LICENSE, NOTICE, README.md, PROMPTING.md
  models/text_encoders/t5_base.safetensors   (the file ComfyUI's Stable Audio workflows use; skipped when it is already there)
Every file is checked against its SHA-256 in tools/model_files.sha256. Needs `huggingface_hub` (installed with ComfyUI's
requirements). Nothing is downloaded unless you run this script.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

SOURCES = {"mimetic_a0": ("micknoisemachine/mimetic-a0", "5f21e1807517cda2e6d72051cdb81dafc3931077"),
           "text_encoders": ("google-t5/t5-base", "a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1")}
WEIGHTS = {"fp32": "model.safetensors", "fp16": "model.fp16.safetensors"}
T5_FILE = "model.safetensors"
MANIFEST = Path(__file__).resolve().with_name("model_files.sha256")


def expected_files() -> dict[str, str]:
    """{path below models/: sha256} from tools/model_files.sha256."""
    out = {}
    for line in MANIFEST.read_text(encoding="ascii").splitlines():
        if line and not line.startswith("#"):
            digest, _, name = line.partition("  ")
            out[name] = digest
    return out


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(models: Path, name: str) -> None:
    from huggingface_hub import hf_hub_download

    top, rest = name.split("/", 1)
    repo, revision = SOURCES[top]
    target = models / name
    if top == "text_encoders":  # saved under another name: download next to it, then move
        with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
            shutil.move(hf_hub_download(repo_id=repo, filename=T5_FILE, revision=revision, local_dir=tmp), target)
    else:
        hf_hub_download(repo_id=repo, filename=rest.split("/", 1)[1], revision=revision, local_dir=str(target.parent))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models-dir", type=Path, required=True, help="ComfyUI's models folder")
    parser.add_argument("--weights", choices=["all", *WEIGHTS], default="fp32", help="mimetic-a0 weight files to fetch (default: fp32)")
    parser.add_argument("--verify-only", action="store_true", help="check existing files, download nothing")
    args = parser.parse_args(argv)

    skip = {f"mimetic_a0/mimetic-a0/{f}" for w, f in WEIGHTS.items() if args.weights not in ("all", w)}
    ok = True
    for name, digest in expected_files().items():
        if name in skip:
            continue
        target = args.models_dir / name
        if name.startswith("text_encoders/") and target.is_file() and sha256(target) != digest:
            print(f"kept     {name} (not the google-t5/t5-base file; any T5-base works, but the comparison in docs/VERIFICATION.md used that one)")
            continue
        if not args.verify_only and not (target.is_file() and sha256(target) == digest):
            target.parent.mkdir(parents=True, exist_ok=True)
            fetch(args.models_dir, name)
        good = target.is_file() and sha256(target) == digest
        ok &= good
        print(f"{'ok      ' if good else ('MISMATCH' if target.is_file() else 'MISSING ')} {name}", flush=True)
    print("all files verified" if ok else "verification FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
