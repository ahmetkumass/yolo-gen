"""Preflight validation for dataset.yaml and vlm_dataset config.

Replaces cryptic Ultralytics FileNotFoundError / KeyError tracebacks with
a single clear message pointing at the exact thing the user needs to fix.

Also resolves a relative ``path:`` in dataset.yaml to an absolute path so
both Ultralytics and VLMDatasetGenerator see the same dataset root, even
when the user writes ``path: .`` or omits it entirely.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import yaml


class PreflightError(SystemExit):
    """Raised with a user-facing error; exits with code 2."""

    def __init__(self, msg: str):
        super().__init__(f"[preflight] {msg}")


def preflight_dataset(
    data_yaml: Path,
    vlm_dataset_config: Optional[Dict] = None,
) -> Path:
    """Validate dataset.yaml and (optionally) the vlm_dataset block.

    Returns a path to a dataset.yaml with an absolute ``path:`` key.
    Writes a sidecar ``<name>.resolved.yaml`` next to the original when
    resolution was needed, so the user's yaml stays portable.
    """
    data_yaml = Path(data_yaml)
    if not data_yaml.exists():
        raise PreflightError(f"dataset.yaml not found: {data_yaml}")

    try:
        cfg = yaml.safe_load(data_yaml.read_text()) or {}
    except yaml.YAMLError as e:
        raise PreflightError(f"{data_yaml} is not valid YAML: {e}")
    if not isinstance(cfg, dict):
        raise PreflightError(f"{data_yaml} must be a YAML mapping at the top level")

    raw_path = cfg.get("path")
    if raw_path is None:
        root = data_yaml.parent.resolve()
    else:
        p = Path(str(raw_path))
        root = p if p.is_absolute() else (data_yaml.parent / p).resolve()

    if not root.exists():
        raise PreflightError(
            f"dataset.yaml 'path:' resolves to a nonexistent directory: {root}\n"
            f"         Edit '{data_yaml}' to point 'path:' at your dataset root."
        )

    train_rel = cfg.get("train", "images/train")
    val_rel = cfg.get("val", "images/val")
    missing = [p for p in (root / train_rel, root / val_rel) if not p.exists()]
    if missing:
        listing = "\n".join(f"         - {p}" for p in missing)
        raise PreflightError(
            f"dataset.yaml references image dirs that do not exist:\n{listing}\n"
            f"         Expected layout:\n"
            f"           {root}/images/train/\n"
            f"           {root}/images/val/\n"
            f"           {root}/labels/train/\n"
            f"           {root}/labels/val/"
        )

    for img_split, split_name in ((root / train_rel, "train"), (root / val_rel, "val")):
        label_split = root / "labels" / split_name
        if not label_split.exists():
            raise PreflightError(
                f"labels dir missing for {split_name} split: {label_split}"
            )

    names = cfg.get("names")
    if names is None:
        raise PreflightError(f"dataset.yaml is missing required key 'names'")
    if isinstance(names, dict):
        name_list = [names[k] for k in sorted(names)]
    elif isinstance(names, list):
        name_list = list(names)
    else:
        raise PreflightError(f"dataset.yaml 'names' must be a list or dict, got {type(names).__name__}")
    if not name_list:
        raise PreflightError("dataset.yaml 'names' is empty — add at least one class")

    if vlm_dataset_config:
        qa = vlm_dataset_config.get("qa_format", "descriptive")
        if qa == "binary_multiclass":
            class_prompts = vlm_dataset_config.get("class_prompts") or {}
            missing = [n for n in name_list if n not in class_prompts]
            if missing:
                raise PreflightError(
                    f"vlm_dataset.qa_format='binary_multiclass' requires a "
                    f"class_prompts entry for every class in dataset.yaml 'names'.\n"
                    f"         Missing: {missing}\n"
                    f"         Present: {list(class_prompts)}"
                )

    if str(root) != str(raw_path or ""):
        cfg["path"] = str(root)
        sidecar = data_yaml.with_name(data_yaml.stem + ".resolved.yaml")
        sidecar.write_text(yaml.safe_dump(cfg, sort_keys=False))
        return sidecar
    return data_yaml
