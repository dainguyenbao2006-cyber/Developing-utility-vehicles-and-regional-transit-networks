"""YAML configuration loading, attribute access wrapper, and schema validation."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import yaml

from utils.constants import DET_CLASSES, SEG_CHANNELS

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_DIR = Path("configs")


class ConfigNode(Mapping[str, Any]):
    """Read-friendly config container supporting both attribute and dict-style access."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data: dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(value, dict):
                self._data[key] = ConfigNode(value)
            elif isinstance(value, list):
                self._data[key] = [
                    ConfigNode(item) if isinstance(item, dict) else item for item in value
                ]
            else:
                self._data[key] = value

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._data[name]
        except KeyError as exc:
            raise AttributeError(f"Config has no attribute '{name}'") from exc

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"ConfigNode({self._data!r})"

    def to_dict(self) -> dict[str, Any]:
        """Recursively convert ConfigNode back into a plain Python dictionary."""
        result: dict[str, Any] = {}
        for key, value in self._data.items():
            if isinstance(value, ConfigNode):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [
                    item.to_dict() if isinstance(item, ConfigNode) else item for item in value
                ]
            else:
                result[key] = value
        return result


def load_yaml(path: str | Path) -> ConfigNode:
    """Load a YAML file into a ConfigNode."""
    yaml_path = Path(path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if raw is None:
        raise ValueError(f"Config file is empty: {yaml_path}")
    if not isinstance(raw, dict):
        raise ValueError(f"Top-level YAML must be a mapping in: {yaml_path}")

    return ConfigNode(raw)


def _apply_env_overrides(paths_dict: dict[str, Any]) -> dict[str, Any]:
    """Override project paths/device from environment variables when provided."""
    updated = {
        "project": dict(paths_dict.get("project", {})),
        "paths": dict(paths_dict.get("paths", {})),
    }

    env_path_map = {
        "BDD_ROOT": "bdd_root",
        "CITYSCAPES_ROOT": "cityscapes_root",
        "OUTPUT_DIR": "output_dir",
    }
    for env_key, cfg_key in env_path_map.items():
        env_val = os.environ.get(env_key)
        if env_val:
            updated["paths"][cfg_key] = env_val
            logger.debug("Applied env override %s -> paths.%s", env_key, cfg_key)

    device_env = os.environ.get("DEVICE")
    if device_env:
        updated["project"]["device"] = device_env

    return updated


def load_project_configs(
    paths_cfg: str | Path = DEFAULT_CONFIG_DIR / "paths.yaml",
    detector_cfg: str | Path = DEFAULT_CONFIG_DIR / "detector_yolo11.yaml",
    segmenter_cfg: str | Path = DEFAULT_CONFIG_DIR / "segmenter_deeplab.yaml",
    adas_cfg: str | Path = DEFAULT_CONFIG_DIR / "adas_rules.yaml",
) -> ConfigNode:
    """Load and merge all project YAML configs into a single validated ConfigNode."""
    base = _apply_env_overrides(load_yaml(paths_cfg).to_dict())
    det = load_yaml(detector_cfg).to_dict()
    seg = load_yaml(segmenter_cfg).to_dict()
    adas_raw = load_yaml(adas_cfg).to_dict()

    merged = {
        "project": base["project"],
        "paths": base["paths"],
        "detector": det,
        "segmenter": seg,
        "adas": adas_raw.get("adas", adas_raw),
    }
    cfg = ConfigNode(merged)
    validate_configs(cfg)
    return cfg


def validate_configs(cfg: ConfigNode) -> None:
    """Validate critical configuration keys and invariant dimensions."""
    for section in ("project", "paths", "detector", "segmenter", "adas"):
        if section not in cfg:
            raise ValueError(f"Missing required config section: '{section}'")

    if "seed" not in cfg.project:
        raise ValueError("Missing required key: 'project.seed'")

    for path_key in (
        "bdd_root",
        "cityscapes_root",
        "detector_weights",
        "segmenter_weights",
        "output_dir",
    ):
        if path_key not in cfg.paths:
            raise ValueError(f"Missing required key: 'paths.{path_key}'")

    seg_model = cfg.segmenter.model
    if seg_model.num_classes != len(SEG_CHANNELS):
        raise ValueError(
            f"segmenter.model.num_classes ({seg_model.num_classes}) "
            f"must match SEG_CHANNELS ({len(SEG_CHANNELS)})"
        )
    if len(seg_model.thresholds) != len(SEG_CHANNELS):
        raise ValueError(
            f"segmenter.model.thresholds length ({len(seg_model.thresholds)}) "
            f"must match SEG_CHANNELS ({len(SEG_CHANNELS)})"
        )
    if len(seg_model.input_size_hw) != 2:
        raise ValueError("segmenter.model.input_size_hw must have 2 elements [H, W]")

    for adas_section in ("ldw", "fcw", "pcw", "tsr", "alert_manager"):
        if adas_section not in cfg.adas:
            raise ValueError(f"Missing required key: 'adas.{adas_section}'")

    for target in cfg.adas.fcw.target_classes:
        if target not in DET_CLASSES:
            raise ValueError(f"Unknown FCW target class: '{target}'")

    for target in cfg.adas.pcw.target_classes:
        if target not in DET_CLASSES:
            raise ValueError(f"Unknown PCW target class: '{target}'")
