"""Load config/pose_validity.yaml strictly (exact keys, float thresholds)."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

from ugv_localization.common.checks import require_bool, require_positive_float
from ugv_localization.common.yamlio import load_yaml_mapping, require_exact_keys
from ugv_localization.validity.monitor import ValidityProfile

# Thresholds where 0.0 is a meaningful value (disabled / no hysteresis / no future tolerance).
_ZERO_OK = {"max_dead_reckon_s", "recover_hold_s", "max_future_s"}
_BOOLS = {"dead_reckon_in_mapping"}


def load_validity_profile(path: str | Path) -> ValidityProfile:
    data = load_yaml_mapping(path)
    names = {f.name for f in fields(ValidityProfile)}
    require_exact_keys(data, names, where=str(path))
    kwargs: dict[str, object] = {}
    for name in names:
        if name in _BOOLS:
            kwargs[name] = require_bool(data[name], name=name)
        else:
            kwargs[name] = require_positive_float(data[name], name=name, allow_zero=name in _ZERO_OK)
    return ValidityProfile(**kwargs)  # type: ignore[arg-type]
