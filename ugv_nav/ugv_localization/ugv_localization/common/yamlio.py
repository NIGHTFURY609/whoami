"""Strict YAML loading: file must exist, top level must be a mapping, keys must match exactly."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_yaml_mapping(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"YAML not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{p}: top level must be a mapping")
    return data


def require_exact_keys(data: dict, expected: set[str], *, where: str) -> None:
    missing = expected - set(data)
    unknown = set(data) - expected
    if missing:
        raise KeyError(f"{where}: missing keys {sorted(missing)}")
    if unknown:
        raise KeyError(f"{where}: unknown keys {sorted(unknown)}")
