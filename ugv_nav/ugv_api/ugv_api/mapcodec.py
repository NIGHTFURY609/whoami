"""Binary encoders for the 3D map layers the gateway sends to the web viewer.

Pure numpy + struct, no ROS imports. The byte layouts are the "Binary format v1" contract with the viewer's
TypeScript decoder, the only decoder (ui/src/map/codec.ts): the golden files in test/fixtures/map/ pin this
encoder's bytes here and that decoder's reading of them there. Everything is
little-endian:

    prelude, 24 bytes, every layer:  char[4] magic | u16 format = 1 | u16 header_bytes | u32 epoch |
                                     u32 seq | f64 stamp_s
    UGVC cloud       header 64  u32 count, u32 source_count, f32 spacing_m, u32 flags (bit0 = rgb),
                                f32[3] bbox_min, f32[3] bbox_max; body f32 xyz[3n] then u8 rgb[3n]
    UGVT trajectory  header 32  u32 count, f32 length_m; body f32 x,y,z,qx,qy,qz,qw per pose
    UGVG cost grid   header 48  u32 width, u32 height, f32 resolution_m, f32 origin_x, f32 origin_y,
                                f32 origin_yaw; body i8 cell[w*h]

2-D layers are row-major with row = y index and column = x index; the origin is the world position of the
corner of cell (0, 0). `epoch` (random per gateway process) and `seq` let the viewer tell a restarted
gateway from a stale frame, so both are always written, modulo 2**32.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence

import numpy as np

FORMAT = 1

MAGIC_CLOUD = b"UGVC"
MAGIC_TRAJECTORY = b"UGVT"
MAGIC_GRID = b"UGVG"

PRELUDE_BYTES = 24
CLOUD_HEADER_BYTES = 64
TRAJECTORY_HEADER_BYTES = 32
GRID_HEADER_BYTES = 48

FLAG_RGB = 1  # UGVC flags bit0


# sensor_msgs/msg/PointField datatype codes
POINTFIELD_UINT32 = 6
POINTFIELD_FLOAT32 = 7

_PRELUDE = struct.Struct("<4sHHIId")
_CLOUD = struct.Struct("<IIfI3f3f")
_TRAJECTORY = struct.Struct("<If")
_GRID = struct.Struct("<IIffff")

assert _PRELUDE.size == PRELUDE_BYTES
assert PRELUDE_BYTES + _CLOUD.size == CLOUD_HEADER_BYTES
assert PRELUDE_BYTES + _TRAJECTORY.size == TRAJECTORY_HEADER_BYTES
assert PRELUDE_BYTES + _GRID.size == GRID_HEADER_BYTES

__all__ = [
    "cloud_view",
    "encode_cloud",
    "encode_trajectory",
    "encode_grid",
]


# ------------------------------------------------------------------------------------------ shared


def _prelude(magic: bytes, header_bytes: int, epoch: int, seq: int, stamp_s: float) -> bytes:
    return _PRELUDE.pack(magic, FORMAT, header_bytes, int(epoch) & 0xFFFFFFFF, int(seq) & 0xFFFFFFFF, float(stamp_s))


def _positive_finite(value: float, name: str) -> float:
    v = float(value)
    if not (math.isfinite(v) and v > 0.0):
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return v


def _origin(origin_xy: Sequence[float]) -> tuple[float, float]:
    ox, oy = float(origin_xy[0]), float(origin_xy[1])
    if not (math.isfinite(ox) and math.isfinite(oy)):
        raise ValueError("origin_xy must be finite")
    return ox, oy


# ------------------------------------------------------------------------------------------- cloud


def cloud_view(
    *,
    fields: Sequence[tuple[str, int, int, int]],
    point_step: int,
    n_points: int,
    is_bigendian: bool,
    data: bytes,
) -> tuple[np.ndarray, np.ndarray | None]:
    """(xyz (N, 3) float32, rgb (N, 3) uint8 or None) over a PointCloud2 payload, without copying `data`.

    `fields` are (name, offset, datatype, count) tuples with sensor_msgs/PointField datatype codes. The
    returned arrays are views of `data` whenever the layout allows (x, y, z adjacent in that order; rgb
    always); otherwise xyz is gathered into a new array. rgb comes from a packed `rgb` (preferred) or
    `rgba` field: a 32-bit word holding 0x??RRGGBB (float32 or uint32), the alpha byte is ignored.
    Raises ValueError for a big-endian cloud, a missing or non-FLOAT32 x/y/z, or a buffer that is too short.
    """
    if is_bigendian:
        raise ValueError("big-endian point clouds are not supported")
    point_step, n_points = int(point_step), int(n_points)
    if point_step <= 0 or n_points < 0:
        raise ValueError("point_step must be > 0 and n_points >= 0")
    by_name = {str(name): (int(offset), int(datatype), int(count)) for name, offset, datatype, count in fields}
    offsets = []
    for axis in "xyz":
        if axis not in by_name:
            raise ValueError(f"point cloud has no '{axis}' field")
        offset, datatype, _ = by_name[axis]
        if datatype != POINTFIELD_FLOAT32:
            raise ValueError(f"'{axis}' field must be FLOAT32 (7), got datatype {datatype}")
        offsets.append(offset)
    rgb_offset = None
    for name in ("rgb", "rgba"):
        if name in by_name and by_name[name][1] in (POINTFIELD_UINT32, POINTFIELD_FLOAT32):
            rgb_offset = by_name[name][0]
            break
    for offset in (*offsets, *(() if rgb_offset is None else (rgb_offset,))):
        if offset < 0 or offset + 4 > point_step:
            raise ValueError(f"field at offset {offset} does not fit in point_step {point_step}")

    base = np.frombuffer(data, dtype=np.uint8)
    if base.size < point_step * n_points:
        raise ValueError(f"data holds {base.size} bytes, fewer than point_step * n_points = {point_step * n_points}")

    ox, oy, oz = offsets
    if n_points == 0:
        xyz = np.zeros((0, 3), dtype=np.float32)
    elif oy == ox + 4 and oz == ox + 8:
        xyz = np.ndarray((n_points, 3), dtype="<f4", buffer=base, offset=ox, strides=(point_step, 4))
    else:
        xyz = np.empty((n_points, 3), dtype=np.float32)
        for col, offset in enumerate(offsets):
            xyz[:, col] = np.ndarray((n_points,), dtype="<f4", buffer=base, offset=offset, strides=(point_step,))

    if rgb_offset is None:
        return xyz, None
    if n_points == 0:
        return xyz, np.zeros((0, 3), dtype=np.uint8)
    # little-endian word 0xAARRGGBB is stored B, G, R, A: columns 2, 1, 0 are R, G, B
    bgra = np.ndarray((n_points, 4), dtype=np.uint8, buffer=base, offset=rgb_offset, strides=(point_step, 1))
    return xyz, bgra[:, 2::-1]


_HASH_MUL = (np.uint64(0x9E3779B185EBCA87), np.uint64(0xC2B2AE3D27D4EB4F), np.uint64(0x165667B19E3779F9))
_SPLITMIX = (np.uint64(0x9E3779B97F4A7C15), np.uint64(0xBF58476D1CE4E5B9), np.uint64(0x94D049BB133111EB))


def _voxel_hash(xyz: np.ndarray, spacing_m: float) -> np.ndarray:
    """uint64 hash of each point's voxel index at `spacing_m`: equal for points in the same voxel."""
    idx = np.floor(xyz.astype(np.float64) / spacing_m)
    idx = np.clip(idx, -(2.0**62), 2.0**62).astype(np.int64).view(np.uint64)
    with np.errstate(over="ignore"):
        h = idx[:, 0] * _HASH_MUL[0] + idx[:, 1] * _HASH_MUL[1] + idx[:, 2] * _HASH_MUL[2]
        h = h + _SPLITMIX[0]
        h = (h ^ (h >> np.uint64(30))) * _SPLITMIX[1]
        h = (h ^ (h >> np.uint64(27))) * _SPLITMIX[2]
        h = h ^ (h >> np.uint64(31))
    return h


def _select_by_voxel_hash(xyz: np.ndarray, rgb: np.ndarray | None, spacing_m: float, budget: int) -> np.ndarray:
    """Boolean mask of the `budget` points kept: those in the voxels with the lowest hash values. When the
    last voxel only partly fits, its points are ranked by their own values, so the chosen set depends only
    on the point set, never on the order the points arrived in."""
    h = _voxel_hash(xyz, spacing_m)
    last = np.partition(h, budget - 1)[budget - 1]  # hash of the last voxel that is kept, maybe in part
    keep = h < last
    edge = np.flatnonzero(h == last)
    room = budget - int(keep.sum())
    if room < len(edge):
        keys = [xyz[edge, 2], xyz[edge, 1], xyz[edge, 0]]
        if rgb is not None:
            keys = [rgb[edge, 2], rgb[edge, 1], rgb[edge, 0], *keys]
        edge = edge[np.lexsort(keys)[:room]]
    keep[edge] = True
    return keep


def _thin_by_voxel(xyz: np.ndarray, spacing_m: float, budget: int) -> np.ndarray:
    """Indices of an even thinning: the first point (in input order) of every voxel of `spacing_m`, then, if
    that is still over `budget`, every k-th of those. Unlike the lowest-hash cut it leaves no holes: every
    occupied voxel of the scan is equally likely to stay, so a dense scan thins out evenly."""
    _, first = np.unique(_voxel_hash(xyz, spacing_m), return_index=True)
    first.sort()  # input order: the scan's row order, which the stride below spreads evenly
    if len(first) > budget:
        first = first[np.linspace(0, len(first) - 1, budget).astype(np.int64)]
    return first


def encode_cloud(
    xyz: np.ndarray,
    rgb: np.ndarray | None,
    *,
    epoch: int,
    seq: int,
    stamp_s: float,
    budget: int,
    spacing_m: float,
    thin: bool = False,
) -> bytes:
    """UGVC. Drops non-finite points; if more than `budget` finite points remain, keeps `budget` of them by
    voxel hash (voxels of `spacing_m`: the lowest-hash voxels first), so the same point set always yields
    the same selection whatever its order. Kept points stay in input order. `source_count` is the finite
    count before that cut; bbox is over the points written.

    `thin=True` (the live scan) replaces that cut: one point per voxel of `spacing_m` always, then an even
    stride down to `budget` (see `_thin_by_voxel`)."""
    pts = np.asarray(xyz, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"xyz must have shape (N, 3), got {pts.shape}")
    col = None
    if rgb is not None:
        col = np.asarray(rgb)
        if col.dtype != np.uint8 or col.shape != pts.shape:
            raise ValueError(f"rgb must be uint8 with the shape of xyz {pts.shape}, got {col.dtype} {col.shape}")
    if int(budget) < 0:
        raise ValueError("budget must be >= 0")
    spacing = _positive_finite(spacing_m, "spacing_m")

    finite = np.isfinite(pts).all(axis=1)
    if not finite.all():
        pts = pts[finite]
        col = None if col is None else col[finite]
    source_count = len(pts)
    if thin and source_count:
        idx = _thin_by_voxel(pts, spacing, int(budget)) if budget else np.zeros(0, np.int64)
        pts = pts[idx]
        col = None if col is None else col[idx]
    elif source_count > budget:
        if budget == 0:
            keep = np.zeros(source_count, dtype=bool)
        else:
            keep = _select_by_voxel_hash(pts, col, spacing, int(budget))
        pts = pts[keep]
        col = None if col is None else col[keep]
    count = len(pts)
    bbox_min = pts.min(axis=0) if count else np.zeros(3, np.float32)
    bbox_max = pts.max(axis=0) if count else np.zeros(3, np.float32)

    out = [
        _prelude(MAGIC_CLOUD, CLOUD_HEADER_BYTES, epoch, seq, stamp_s),
        _CLOUD.pack(count, source_count, spacing, FLAG_RGB if col is not None else 0, *bbox_min, *bbox_max),
        np.ascontiguousarray(pts, dtype="<f4").tobytes(),
    ]
    if col is not None:
        out.append(np.ascontiguousarray(col, dtype=np.uint8).tobytes())
    return b"".join(out)


# --------------------------------------------------------------------------------------- trajectory


def encode_trajectory(poses: np.ndarray, *, epoch: int, seq: int, stamp_s: float) -> bytes:
    """UGVT from an (N, 7) float32 array of x, y, z, qx, qy, qz, qw. Poses with a non-finite value are
    dropped. `length_m` is the 3-D length of the polyline through the poses written."""
    p = np.asarray(poses, dtype=np.float32)
    if p.size == 0:
        p = p.reshape(0, 7)
    if p.ndim != 2 or p.shape[1] != 7:
        raise ValueError(f"poses must have shape (N, 7), got {p.shape}")
    p = p[np.isfinite(p).all(axis=1)]
    steps = np.diff(p[:, :3].astype(np.float64), axis=0)
    length = float(np.sqrt((steps * steps).sum(axis=1)).sum())
    return b"".join(
        (
            _prelude(MAGIC_TRAJECTORY, TRAJECTORY_HEADER_BYTES, epoch, seq, stamp_s),
            _TRAJECTORY.pack(len(p), length),
            np.ascontiguousarray(p, dtype="<f4").tobytes(),
        )
    )


# --------------------------------------------------------------------------------------- cost grid


def encode_grid(
    cells: np.ndarray,
    *,
    epoch: int,
    seq: int,
    stamp_s: float,
    resolution: float,
    origin_xy: Sequence[float],
    origin_yaw: float,
) -> bytes:
    """UGVG from an integer (H, W) array in OccupancyGrid convention (-1 unknown, 0..100 cost); values
    outside -1..100 are clamped into it."""
    a = np.asarray(cells)
    if a.ndim != 2 or a.dtype.kind not in "iu":
        raise ValueError(f"cells must be a 2-D integer array, got {a.dtype} with shape {a.shape}")
    a = np.minimum(a, 100) if a.dtype.kind == "u" else np.clip(a, -1, 100)
    a = a.astype(np.int8)
    res = _positive_finite(resolution, "resolution")
    ox, oy = _origin(origin_xy)
    rows, cols = a.shape
    return b"".join(
        (
            _prelude(MAGIC_GRID, GRID_HEADER_BYTES, epoch, seq, stamp_s),
            _GRID.pack(cols, rows, res, ox, oy, float(origin_yaw)),
            np.ascontiguousarray(a).tobytes(),
        )
    )
