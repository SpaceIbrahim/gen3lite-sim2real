"""Config loading helpers.

All tunable numbers live in YAML files under ``configs/``.  This module loads
them into plain nested dicts and supports command-line style overrides such as
``task.reward.weights.reach=2.0``.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Iterable

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_ENV_CONFIGS = {
    "robot": "configs/robot.yaml",
    "drawer": "configs/drawer.yaml",
    "scene": "configs/scene.yaml",
    "task": "configs/task.yaml",
}


def resolve_path(p: str | os.PathLike) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_yaml(path: str | os.PathLike) -> dict:
    with open(resolve_path(path), "r") as f:
        return yaml.safe_load(f) or {}


def _set_dotted(cfg: dict, dotted: str, value: Any) -> None:
    """Set ``a.b.0.c``-style keys; numeric parts index into lists."""
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node[int(k)] if isinstance(node, list) else node.setdefault(k, {})
    if isinstance(node, list):
        node[int(keys[-1])] = value
    else:
        node[keys[-1]] = value


def apply_overrides(cfg: dict, overrides: Iterable[str] | None) -> dict:
    """Apply ``a.b.c=value`` overrides (value parsed as YAML)."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must look like key.path=value")
        key, val = item.split("=", 1)
        _set_dotted(cfg, key.strip(), yaml.safe_load(val))
    return cfg


def load_env_config(paths: dict | None = None, overrides: Iterable[str] | None = None) -> dict:
    """Load the four environment config files into one dict:
    ``{"robot": ..., "drawer": ..., "scene": ..., "task": ...}``.

    ``overrides`` use the same top-level names, e.g. ``drawer.top_drawer.slide.damping=12``.
    """
    paths = {**DEFAULT_ENV_CONFIGS, **(paths or {})}
    cfg = {name: load_yaml(p) for name, p in paths.items()}
    return apply_overrides(cfg, overrides)


def save_yaml(cfg: dict, path: str | os.PathLike) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
