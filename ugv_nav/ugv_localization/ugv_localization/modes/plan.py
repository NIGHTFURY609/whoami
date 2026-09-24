"""mapping vs localize (architecture §10) → RTAB-Map params + CLI args.

mapping : build / extend the visual graph; database saved on shutdown.
          fresh=True deletes the old database at start ("-d").
          fresh=False on an existing db continues it as a new session (multi-session map).
localize: load database, graph read-only (Mem/IncrementalMemory=false), all nodes in WM
          so relocalization can match anywhere. Missing/empty db fails fast — never silently
          start an empty map while claiming to localize.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from ugv_localization.common.checks import require_bool


class Mode(str, Enum):
    MAPPING = "mapping"
    LOCALIZE = "localize"


class ModeError(ValueError):
    """Mode / database combination is not runnable."""


_MODE_PARAMS: dict[Mode, dict[str, str]] = {
    Mode.MAPPING: {"Mem/IncrementalMemory": "true", "Mem/InitWMWithAllNodes": "false"},
    Mode.LOCALIZE: {"Mem/IncrementalMemory": "false", "Mem/InitWMWithAllNodes": "true"},
}


@dataclass(frozen=True, slots=True)
class ModePlan:
    mode: Mode
    database_path: str
    rtabmap_args: tuple[str, ...]
    _params: tuple[tuple[str, str], ...]

    def rtabmap_params(self) -> dict[str, str]:
        return dict(self._params)


def parse_mode(value: str) -> Mode:
    try:
        return Mode(value)
    except ValueError:
        raise ModeError(f"mode must be 'mapping' or 'localize', got {value!r}") from None


def plan_mode(mode: Mode, database_path: str, *, fresh: bool) -> ModePlan:
    if type(mode) is not Mode:
        raise TypeError("mode must be a Mode")
    fresh = require_bool(fresh, name="fresh")
    path = Path(os.path.expanduser(database_path))
    if not path.is_absolute():
        raise ModeError(f"database_path must be absolute, got {database_path!r}")
    if path.suffix != ".db":
        raise ModeError(f"database_path must end in .db, got {path.name!r}")

    if mode is Mode.LOCALIZE:
        if fresh:
            raise ModeError("fresh database is meaningless in localize mode (nothing to localize against)")
        if not path.exists():
            raise ModeError(f"localize: database {path} does not exist — run mode:=mapping first")
        if not path.is_file():
            raise ModeError(f"localize: {path} is not a file")
        if path.stat().st_size == 0:
            raise ModeError(f"localize: database {path} is empty")
        args: tuple[str, ...] = ()
    else:
        if not path.parent.is_dir():
            raise ModeError(f"mapping: parent directory {path.parent} does not exist")
        args = ("-d",) if fresh else ()

    return ModePlan(
        mode=mode,
        database_path=str(path),
        rtabmap_args=args,
        _params=tuple(sorted(_MODE_PARAMS[mode].items())),
    )
