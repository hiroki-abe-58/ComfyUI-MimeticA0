"""mimetic-a0 for ComfyUI (unofficial integration of https://huggingface.co/micknoisemachine/mimetic-a0)."""

import logging

try:
    from .mimetic_comfy.nodes import comfy_entrypoint  # noqa: F401
except ImportError as e:  # ComfyUI without the V3 node API or without the stable-audio classes
    logging.error("[mimetic-a0] not loaded: %s", e)
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
