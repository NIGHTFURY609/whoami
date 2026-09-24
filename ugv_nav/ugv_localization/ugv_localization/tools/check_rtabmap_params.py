"""Verify every RTAB-Map library parameter we set actually exists in the installed RTAB-Map.

RTAB-Map silently ignores unknown parameter names, and names drift between versions
(e.g. Grid/FromDepth → Grid/Sensor). A typo in the mono config would quietly run defaults.

Usage (on a machine with ROS + rtabmap_ros):
    ros2 run rtabmap_slam rtabmap --params > /tmp/rtabmap_params.txt
    ros2 run ugv_localization check_rtabmap_params --dump /tmp/rtabmap_params.txt
Exit 0 = all known; 1 = unknown names printed.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

from ugv_localization.modes import Mode

_PARAM_LINE = re.compile(r"^\s*(?:Param:\s*)?([A-Za-z0-9_]+/[A-Za-z0-9_]+)\s*=")


def parse_params_dump(text: str) -> set[str]:
    names = {m.group(1) for line in text.splitlines() if (m := _PARAM_LINE.match(line))}
    if not names:
        raise ValueError("no parameters recognized in dump — is this `rtabmap --params` output?")
    return names


def config_library_keys(config_path: str | Path) -> set[str]:
    data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    keys: set[str] = set()
    for node_block in data.values():
        params = node_block.get("ros__parameters", {}) if isinstance(node_block, dict) else {}
        keys |= {k for k in params if "/" in k}
    return keys


def _mode_keys() -> set[str]:
    # Local import keeps the module importable without touching the filesystem.
    from ugv_localization.modes.plan import _MODE_PARAMS

    return {k for mode in Mode for k in _MODE_PARAMS[mode]}


def unknown_keys(ours: set[str], known: set[str]) -> list[str]:
    return sorted(ours - known)


def _default_config() -> Path:
    try:
        from ament_index_python.packages import get_package_share_directory

        return Path(get_package_share_directory("ugv_localization")) / "config" / "rtabmap_mono.yaml"
    except Exception:  # noqa: BLE001 — no ROS: fall back to the source tree
        return Path(__file__).resolve().parents[2] / "config" / "rtabmap_mono.yaml"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", required=True, help="file with `rtabmap --params` output, or - for stdin")
    ap.add_argument("--config", default=None, help="rtabmap params YAML (default: installed rtabmap_mono.yaml)")
    args = ap.parse_args(argv)

    text = sys.stdin.read() if args.dump == "-" else Path(args.dump).read_text(encoding="utf-8")
    known = parse_params_dump(text)
    cfg = Path(args.config) if args.config else _default_config()
    ours = config_library_keys(cfg) | _mode_keys()
    bad = unknown_keys(ours, known)
    if bad:
        print(f"UNKNOWN RTAB-Map parameters ({len(bad)}) in {cfg} / mode params:")
        for name in bad:
            print(f"  {name}")
        return 1
    print(f"OK: {len(ours)} parameters all known to this RTAB-Map ({len(known)} available)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
