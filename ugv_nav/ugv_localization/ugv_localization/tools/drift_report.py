"""Offline drift report from two CSVs (t_ns,x,y,z): estimate vs ground truth → JSON.

    ros2 run ugv_localization drift_report --est est.csv --gt gt.csv --out report.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from ugv_localization.drift import drift_report


def read_csv(path: str | Path) -> tuple[list[int], np.ndarray]:
    stamps: list[int] = []
    xyz: list[tuple[float, float, float]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != ["t_ns", "x", "y", "z"]:
            raise ValueError(f"{path}: header must be t_ns,x,y,z")
        for row in reader:
            stamps.append(int(row["t_ns"]))
            xyz.append((float(row["x"]), float(row["y"]), float(row["z"])))
    return stamps, np.asarray(xyz, dtype=np.float64).reshape(-1, 3)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--est", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-dt-ms", type=float, default=20.0)
    ap.add_argument("--segments", type=float, nargs="+", default=[5.0, 10.0, 20.0])
    ap.add_argument("--with-scale", action="store_true", help="similarity alignment (mono-only runs)")
    args = ap.parse_args(argv)

    est_t, est = read_csv(args.est)
    gt_t, gt = read_csv(args.gt)
    rep = drift_report(
        est_t,
        est,
        gt_t,
        gt,
        max_dt_ns=int(args.max_dt_ms * 1e6),
        segments_m=args.segments,
        with_scale=args.with_scale,
    )
    out = {
        "matched": rep.matched,
        "gt_path_length_m": rep.gt_path_length_m,
        "ate_rmse_m": rep.ate_rmse_m,
        "ate_max_m": rep.ate_max_m,
        "rpe_mean_pct": {str(k): (None if math.isnan(v) else v) for k, v in rep.rpe_mean_pct.items()},
        "endpoint_error_m": rep.endpoint_error_m,
        "endpoint_drift_pct": rep.endpoint_drift_pct,
    }
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
