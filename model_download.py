#!/usr/bin/env python3
"""
Shared model download helpers for SimpleTool servers.
"""

import inspect
import os
from typing import Dict

HF_REPO_ID = "Cialtion/SimpleTool"
HF_MODEL_SUBDIR = "RT-Qwen3-4B-AWQ"
MODEL_ROOT = os.environ.get("MODEL_ROOT", "./models")
DEFAULT_MODEL_PATH = os.path.join(MODEL_ROOT, HF_MODEL_SUBDIR)


def _is_falsey_env(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"0", "false", "no", "off"}


def _looks_like_model_dir(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "config.json"))


def ensure_default_model(local_model_path: str = DEFAULT_MODEL_PATH) -> str:
    """
    Ensure RT-Qwen3-4B-AWQ exists locally. Auto-download by default.
    """
    model_path = os.path.abspath(local_model_path)
    if _looks_like_model_dir(model_path):
        return model_path

    if _is_falsey_env("SIMPLETOOL_AUTO_DOWNLOAD", default="1"):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Set SIMPLETOOL_AUTO_DOWNLOAD=1 or place model manually."
        )

    model_root = os.path.abspath(os.path.dirname(model_path))
    os.makedirs(model_root, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Auto-download requires huggingface_hub. Install it with: uv pip install huggingface_hub"
        ) from exc

    print(f"[Model] Downloading {HF_REPO_ID}/{HF_MODEL_SUBDIR} to {model_root} ...")
    kwargs: Dict[str, object] = {
        "repo_id": HF_REPO_ID,
        "allow_patterns": [f"{HF_MODEL_SUBDIR}/*"],
        "local_dir": model_root,
        "resume_download": True,
    }

    # Some huggingface_hub versions support this flag; keep compatibility across versions.
    if "local_dir_use_symlinks" in inspect.signature(snapshot_download).parameters:
        kwargs["local_dir_use_symlinks"] = False

    snapshot_download(**kwargs)

    if not _looks_like_model_dir(model_path):
        raise RuntimeError(
            f"Download finished but model files were not found at {model_path}. "
            f"Please verify repo={HF_REPO_ID}, subdir={HF_MODEL_SUBDIR}."
        )

    print(f"[Model] Ready: {model_path}")
    return model_path
