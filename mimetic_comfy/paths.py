"""Model location: `<ComfyUI models>/mimetic_a0/<name>/` holding the release files of the Hugging Face repository
micknoisemachine/mimetic-a0 (model_config.json and model.safetensors and/or model.fp16.safetensors), plus any path added for
`mimetic_a0` in extra_model_paths.yaml. Nothing is downloaded at run time.
"""

from __future__ import annotations

import os

FOLDER = "mimetic_a0"
WEIGHT_FILES = ("model.safetensors", "model.fp16.safetensors")


def register() -> None:
    import folder_paths

    folder_paths.add_model_folder_path(FOLDER, os.path.join(folder_paths.models_dir, FOLDER), is_default=False)


def _is_release(folder: str) -> bool:
    return os.path.isfile(os.path.join(folder, "model_config.json")) and any(os.path.isfile(os.path.join(folder, f)) for f in WEIGHT_FILES)


def model_names() -> list[str]:
    """Sub-folders of the registered mimetic_a0 folders that hold a release."""
    import folder_paths

    names = set()
    for base in folder_paths.get_folder_paths(FOLDER):
        if os.path.isdir(base):
            names.update(n for n in os.listdir(base) if os.path.isdir(os.path.join(base, n)) and _is_release(os.path.join(base, n)))
    return sorted(names)


def model_dir(name: str) -> str:
    """The folder of model `name`. Only names listed by model_names() are accepted, so a value cannot leave the models folders."""
    import folder_paths

    if name not in model_names():
        raise FileNotFoundError(f"no mimetic-a0 release named {name!r} in models/{FOLDER}/ (see the README for the download command)")
    for base in folder_paths.get_folder_paths(FOLDER):
        folder = os.path.join(base, name)
        if os.path.isdir(folder) and _is_release(folder):
            return folder
    raise FileNotFoundError(name)
