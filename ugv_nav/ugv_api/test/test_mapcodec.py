"""Binary map codec ("Binary format v1"): layouts, lengths, selection rules and golden fixtures.

The golden files in test/fixtures/map/ are the byte-level contract with the web viewer's TypeScript
decoder. `build_golden()` below holds the exact inputs each file is encoded from, so a decoder author
can read what a file must decode to without running Python. Rewrite the files (only when the format
changes on purpose) with:  python3 test/test_mapcodec.py --write

All three fixtures use epoch=7, seq=3, stamp_s=1234.5. Arrays are row-major, row = y index, column = x
index. Floats are float32 on the wire, and every value below is exactly representable unless noted.

  cloud.bin       UGVC, 184 bytes (64 + 15 * 8)   count 8, source_count 8, spacing_m 0.25, flags 1 (rgb),
                  bbox_min (-1.5, -0.75, 0), bbox_max (1, 2.25, 3). Points and colours, in order:
                    0 ( 0.00,  0.00, 0.0) (255,   0,   0)    4 ( 0.0, 0.0, 0.5) (  0, 255, 255)
                    1 ( 1.00,  0.00, 0.0) (  0, 255,   0)    5 ( 1.0, 0.0, 0.5) (255,   0, 255)
                    2 ( 0.00,  1.00, 0.0) (  0,   0, 255)    6 (-1.5, 2.25, 3.0) ( 16,  32,  48)
                    3 ( 1.00,  1.00, 0.0) (255, 255,   0)    7 ( 0.25, -0.75, 1.5) (250, 128,   7)
  trajectory.bin  UGVT, 116 bytes (32 + 28 * 3)   count 3, length_m 17 (3-D polyline: 5 + 12).
                  poses (x y z qx qy qz qw):  (0 0 0  0 0 0 1)  (3 4 0  0 0 1 0)  (3 4 12  .5 .5 .5 .5)
  grid.bin        UGVG, 60 bytes (48 + 12)        width 4, height 3, resolution_m 0.25, origin (-0.5, 1),
                  origin_yaw 0.5. cells (int8):   -1 0 10 20 | 30 40 50 60 | 70 80 90 100
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

import mapread
from ugv_api import mapcodec as mc

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "map"
EPOCH, SEQ, STAMP_S = 7, 3, 1234.5
KW = {"epoch": EPOCH, "seq": SEQ, "stamp_s": STAMP_S}
NAN = float("nan")

# sensor_msgs/PointField datatype codes
UINT32, FLOAT32, FLOAT64 = 6, 7, 8


# ---------------------------------------------------------------------------------- golden inputs

CLOUD_XYZ = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 0.5],
        [1.0, 0.0, 0.5],
        [-1.5, 2.25, 3.0],
        [0.25, -0.75, 1.5],
    ],
    dtype=np.float32,
)
CLOUD_RGB = np.array(
    [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255], [16, 32, 48], [250, 128, 7]],
    dtype=np.uint8,
)
TRAJ_POSES = np.array(
    [[0, 0, 0, 0, 0, 0, 1], [3, 4, 0, 0, 0, 1, 0], [3, 4, 12, 0.5, 0.5, 0.5, 0.5]], dtype=np.float32
)
GRID_CELLS = np.array([[-1, 0, 10, 20], [30, 40, 50, 60], [70, 80, 90, 100]], dtype=np.int8)
GOLDEN_NAMES = ("cloud.bin", "trajectory.bin", "grid.bin")


def build_golden() -> dict[str, bytes]:
    return {
        "cloud.bin": mc.encode_cloud(CLOUD_XYZ, CLOUD_RGB, budget=100, spacing_m=0.25, **KW),
        "trajectory.bin": mc.encode_trajectory(TRAJ_POSES, **KW),
        "grid.bin": mc.encode_grid(GRID_CELLS, resolution=0.25, origin_xy=(-0.5, 1.0), origin_yaw=0.5, **KW),
    }


# --------------------------------------------------------------------------------------- helpers


def f32(v: float) -> float:
    return float(np.float32(v))


def rows(a: np.ndarray) -> set[tuple]:
    return {tuple(r) for r in a.tolist()}


def point_cloud(xyz, rgb=None, *, packed="rgb", pad=4):
    """(kwargs for cloud_view) for an x,y,z[,rgb] float32 cloud with `pad` bytes between z and rgb."""
    n = len(xyz)
    names = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if pad:
        names.append(("pad", f"V{pad}"))
    if rgb is not None:
        names.append((packed, "<f4"))
    dt = np.dtype(names)
    arr = np.zeros(n, dtype=dt)
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    fields = [("x", 0, FLOAT32, 1), ("y", 4, FLOAT32, 1), ("z", 8, FLOAT32, 1)]
    if rgb is not None:
        r, g, b = (rgb[:, i].astype(np.uint32) for i in range(3))
        arr[packed] = ((r << 16) | (g << 8) | b).astype("<u4").view("<f4")
        fields.append((packed, dt.fields[packed][1], FLOAT32, 1))
    return {"fields": fields, "point_step": dt.itemsize, "n_points": n, "is_bigendian": False, "data": arr.tobytes()}


def random_cloud(n=2000, seed=1):
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(-20.0, 20.0, size=(n, 3)).astype(np.float32)
    rgb = rng.integers(0, 256, size=(n, 3), dtype=np.uint8)
    return xyz, rgb


# ------------------------------------------------------------------------------ prelude / epoch


def test_prelude_layout():
    b = mc.encode_trajectory(TRAJ_POSES, **KW)
    assert b[:4] == b"UGVT"
    fmt, header_bytes, epoch, seq, stamp = struct.unpack_from("<HHIId", b, 4)
    assert (fmt, header_bytes, epoch, seq, stamp) == (1, 32, 7, 3, 1234.5)


def test_epoch_alone_changes_bytes_8_to_11():
    a = mc.encode_cloud(CLOUD_XYZ, None, epoch=1, seq=5, stamp_s=2.0, budget=100, spacing_m=0.25)
    b = mc.encode_cloud(CLOUD_XYZ, None, epoch=2, seq=5, stamp_s=2.0, budget=100, spacing_m=0.25)
    assert a[8:12] != b[8:12]
    assert a[:8] == b[:8] and a[12:] == b[12:]
    assert struct.unpack_from("<I", a, 8)[0] == 1 and struct.unpack_from("<I", b, 8)[0] == 2
    assert struct.unpack_from("<I", a, 12)[0] == struct.unpack_from("<I", b, 12)[0] == 5


def test_epoch_and_seq_wrap_to_u32():
    b = mc.encode_trajectory(TRAJ_POSES, epoch=2**32 + 9, seq=-1 & 0xFFFFFFFF, stamp_s=0.0)
    d = mapread.trajectory(b)
    assert (d["epoch"], d["seq"]) == (9, 0xFFFFFFFF)


# --------------------------------------------------------------------------------------- cloud


def test_cloud_roundtrip_with_rgb_has_spec_length():
    b = mc.encode_cloud(CLOUD_XYZ, CLOUD_RGB, budget=100, spacing_m=0.25, **KW)
    assert len(b) == 64 + 15 * 8
    d = mapread.cloud(b)
    assert (d["epoch"], d["seq"], d["stamp_s"]) == (7, 3, 1234.5)
    assert (d["count"], d["source_count"], d["spacing_m"], d["flags"]) == (8, 8, 0.25, 1)
    assert d["has_rgb"] is True
    np.testing.assert_array_equal(d["xyz"], CLOUD_XYZ)
    np.testing.assert_array_equal(d["rgb"], CLOUD_RGB)
    np.testing.assert_array_equal(d["bbox_min"], np.array([-1.5, -0.75, 0.0], np.float32))
    np.testing.assert_array_equal(d["bbox_max"], np.array([1.0, 2.25, 3.0], np.float32))


def test_cloud_without_rgb_clears_flag_bit0_and_is_shorter():
    b = mc.encode_cloud(CLOUD_XYZ, None, budget=100, spacing_m=0.25, **KW)
    assert len(b) == 64 + 12 * 8
    d = mapread.cloud(b)
    assert d["flags"] & 1 == 0 and d["has_rgb"] is False and d["rgb"] is None
    np.testing.assert_array_equal(d["xyz"], CLOUD_XYZ)


def test_cloud_header_offsets():
    b = mc.encode_cloud(CLOUD_XYZ, CLOUD_RGB, budget=100, spacing_m=0.25, **KW)
    assert struct.unpack_from("<IIfI", b, 24) == (8, 8, 0.25, 1)
    assert struct.unpack_from("<3f", b, 40) == (-1.5, -0.75, 0.0)
    assert struct.unpack_from("<3f", b, 52) == (1.0, 2.25, 3.0)
    assert struct.unpack_from("<3f", b, 64) == (0.0, 0.0, 0.0)  # first point
    assert struct.unpack_from("<3f", b, 64 + 4 * 12) == (0.0, 0.0, 0.5)  # fifth point
    assert b[64 + 96 : 64 + 99] == bytes([255, 0, 0])  # rgb block starts after 12 * count xyz bytes


def test_cloud_drops_non_finite_points_and_reports_finite_source_count():
    xyz = np.array([[1, 2, 3], [NAN, 0, 0], [0, np.inf, 0], [0, 0, -np.inf], [4, 5, 6]], dtype=np.float32)
    rgb = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]], dtype=np.uint8)
    d = mapread.cloud(mc.encode_cloud(xyz, rgb, budget=100, spacing_m=0.1, **KW))
    assert (d["count"], d["source_count"]) == (2, 2)
    np.testing.assert_array_equal(d["xyz"], np.array([[1, 2, 3], [4, 5, 6]], np.float32))
    np.testing.assert_array_equal(d["rgb"], np.array([[1, 1, 1], [5, 5, 5]], np.uint8))


def test_cloud_budget_caps_count_and_source_count_is_pre_selection():
    xyz, rgb = random_cloud(2000)
    b = mc.encode_cloud(xyz, rgb, budget=300, spacing_m=1.0, **KW)
    d = mapread.cloud(b)
    assert d["count"] <= 300 and len(b) == 64 + 15 * d["count"]
    assert d["source_count"] == 2000
    assert d["spacing_m"] == 1.0


def test_cloud_selection_is_a_subset_that_keeps_each_points_colour():
    xyz, rgb = random_cloud(2000)
    d = mapread.cloud(mc.encode_cloud(xyz, rgb, budget=300, spacing_m=1.0, **KW))
    assert 0 < d["count"] <= 300
    colour_of = {tuple(p): tuple(c) for p, c in zip(xyz.tolist(), rgb.tolist())}
    for p, c in zip(d["xyz"].tolist(), d["rgb"].tolist()):
        assert colour_of[tuple(p)] == tuple(c)


def test_cloud_selection_is_deterministic_and_independent_of_point_order():
    xyz, rgb = random_cloud(2000)
    kw = {"budget": 300, "spacing_m": 1.0, **KW}
    first = mc.encode_cloud(xyz, rgb, **kw)
    assert mc.encode_cloud(xyz, rgb, **kw) == first
    for seed in (5, 6):
        perm = np.random.default_rng(seed).permutation(len(xyz))
        shuffled = mapread.cloud(mc.encode_cloud(xyz[perm], rgb[perm], **kw))
        original = mapread.cloud(first)
        assert rows(shuffled["xyz"]) == rows(original["xyz"])
        assert shuffled["count"] == original["count"]


def test_cloud_selection_keeps_whole_voxels_by_hash_not_by_input_position():
    # All points of one voxel share a hash, so a voxel is either kept whole or dropped whole while the budget
    # is not exhausted mid-voxel. 400 voxels of 2 points; budget 100 = 50 whole voxels.
    g = np.arange(400, dtype=np.float32)
    voxel_xyz = np.stack([g, np.zeros_like(g), np.zeros_like(g)], axis=1) + 0.25
    xyz = np.concatenate([voxel_xyz, voxel_xyz + np.float32(0.5)])
    d = mapread.cloud(mc.encode_cloud(xyz, None, budget=100, spacing_m=1.0, **KW))
    assert d["count"] == 100
    cells = np.floor(d["xyz"][:, 0]).astype(int)
    assert np.unique(cells).size == 50
    # reversing the input must not change which voxels were kept
    d2 = mapread.cloud(mc.encode_cloud(xyz[::-1], None, budget=100, spacing_m=1.0, **KW))
    assert set(cells.tolist()) == set(np.floor(d2["xyz"][:, 0]).astype(int).tolist())


def test_cloud_selection_cutting_through_one_voxel_ignores_input_order():
    rng = np.random.default_rng(3)
    xyz = rng.uniform(1.0, 9.0, size=(500, 3)).astype(np.float32)  # one 100 m voxel holds every point
    rgb = rng.integers(0, 256, size=(500, 3), dtype=np.uint8)
    kw = {"budget": 120, "spacing_m": 100.0, **KW}
    want = mapread.cloud(mc.encode_cloud(xyz, rgb, **kw))
    assert want["count"] == 120
    for seed in (1, 2, 3):
        perm = np.random.default_rng(seed).permutation(500)
        got = mapread.cloud(mc.encode_cloud(xyz[perm], rgb[perm], **kw))
        assert rows(got["xyz"]) == rows(want["xyz"])


def test_thin_keeps_one_point_per_occupied_voxel_and_drops_non_finite_rows():
    # a 1 m slab sampled at 2 cm, plus NaN rows: every 10 cm voxel of it keeps exactly one point (no holes,
    # unlike the lowest-hash cut, which keeps whole voxels and drops the others)
    g = np.arange(0.0, 1.0, 0.02, dtype=np.float32)
    xx, yy = np.meshgrid(g, g)
    xyz = np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, 0.5, np.float32)], axis=1)
    xyz = np.concatenate([xyz, np.full((3, 3), np.nan, np.float32)])
    d = mapread.cloud(mc.encode_cloud(xyz, None, budget=10_000, spacing_m=0.1, thin=True, **KW))
    assert d["source_count"] == 2500 and d["count"] == 100
    cells = np.floor(d["xyz"][:, :2] / 0.1 + 1e-4).astype(int)
    assert len({tuple(c) for c in cells.tolist()}) == 100
    cut = mapread.cloud(mc.encode_cloud(xyz, None, budget=25, spacing_m=0.1, thin=True, **KW))
    assert cut["count"] == 25 and np.isfinite(cut["xyz"]).all()


def test_cloud_budget_larger_than_cloud_keeps_everything_in_input_order():
    xyz, rgb = random_cloud(50)
    d = mapread.cloud(mc.encode_cloud(xyz, rgb, budget=50, spacing_m=1.0, **KW))
    np.testing.assert_array_equal(d["xyz"], xyz)
    np.testing.assert_array_equal(d["rgb"], rgb)


def test_cloud_bbox_is_over_selected_points():
    xyz, _ = random_cloud(2000)
    d = mapread.cloud(mc.encode_cloud(xyz, None, budget=100, spacing_m=2.0, **KW))
    np.testing.assert_array_equal(d["bbox_min"], d["xyz"].min(axis=0))
    np.testing.assert_array_equal(d["bbox_max"], d["xyz"].max(axis=0))


@pytest.mark.parametrize("rgb_given", [True, False])
def test_cloud_empty_has_zero_bbox_and_header_only_length(rgb_given):
    xyz = np.zeros((0, 3), np.float32)
    rgb = np.zeros((0, 3), np.uint8) if rgb_given else None
    b = mc.encode_cloud(xyz, rgb, budget=10, spacing_m=0.1, **KW)
    assert len(b) == 64
    d = mapread.cloud(b)
    assert d["count"] == d["source_count"] == 0
    assert not d["bbox_min"].any() and not d["bbox_max"].any()
    assert d["has_rgb"] is rgb_given


def test_cloud_all_non_finite_is_empty():
    xyz = np.full((4, 3), NAN, np.float32)
    d = mapread.cloud(mc.encode_cloud(xyz, None, budget=10, spacing_m=0.1, **KW))
    assert d["count"] == d["source_count"] == 0


def test_cloud_budget_zero_selects_nothing():
    d = mapread.cloud(mc.encode_cloud(CLOUD_XYZ, CLOUD_RGB, budget=0, spacing_m=0.25, **KW))
    assert d["count"] == 0 and d["source_count"] == 8


def test_cloud_rejects_bad_arguments():
    with pytest.raises(ValueError):
        mc.encode_cloud(CLOUD_XYZ[:, :2], None, budget=10, spacing_m=0.1, **KW)
    with pytest.raises(ValueError):
        mc.encode_cloud(CLOUD_XYZ, CLOUD_RGB[:3], budget=10, spacing_m=0.1, **KW)
    with pytest.raises(ValueError):
        mc.encode_cloud(CLOUD_XYZ, CLOUD_RGB.astype(np.int32), budget=10, spacing_m=0.1, **KW)
    with pytest.raises(ValueError):
        mc.encode_cloud(CLOUD_XYZ, None, budget=-1, spacing_m=0.1, **KW)
    for bad in (0.0, -1.0, NAN, np.inf):
        with pytest.raises(ValueError):
            mc.encode_cloud(CLOUD_XYZ, None, budget=10, spacing_m=bad, **KW)


# ------------------------------------------------------------------------------------ cloud_view


def test_cloud_view_reads_xyz_without_copying_the_buffer():
    kw = point_cloud(CLOUD_XYZ)
    xyz, rgb = mc.cloud_view(**kw)
    assert rgb is None
    assert xyz.shape == (8, 3) and xyz.dtype == np.float32
    np.testing.assert_array_equal(xyz, CLOUD_XYZ)
    assert np.shares_memory(xyz, np.frombuffer(kw["data"], np.uint8))


def test_cloud_view_extracts_packed_rgb_float32():
    xyz, rgb = mc.cloud_view(**point_cloud(CLOUD_XYZ, CLOUD_RGB))
    np.testing.assert_array_equal(xyz, CLOUD_XYZ)
    assert rgb.shape == (8, 3) and rgb.dtype == np.uint8
    np.testing.assert_array_equal(rgb, CLOUD_RGB)


def test_cloud_view_extracts_rgba_field_and_ignores_alpha():
    kw = point_cloud(CLOUD_XYZ, CLOUD_RGB, packed="rgba")
    data = bytearray(kw["data"])
    for i in range(8):  # alpha lives in the top byte of the packed word
        data[i * kw["point_step"] + 16 + 3] = 0x7F
    kw["data"] = bytes(data)
    _, rgb = mc.cloud_view(**kw)
    np.testing.assert_array_equal(rgb, CLOUD_RGB)


def test_cloud_view_accepts_uint32_rgb_field_and_numpy_data():
    kw = point_cloud(CLOUD_XYZ, CLOUD_RGB)
    kw["fields"][3] = ("rgb", kw["fields"][3][1], UINT32, 1)
    kw["data"] = np.frombuffer(kw["data"], np.uint8)
    xyz, rgb = mc.cloud_view(**kw)
    np.testing.assert_array_equal(rgb, CLOUD_RGB)
    np.testing.assert_array_equal(xyz, CLOUD_XYZ)


def test_cloud_view_handles_xyz_that_are_not_adjacent():
    n = 5
    dt = np.dtype([("x", "<f4"), ("a", "<f4"), ("y", "<f4"), ("b", "<f4"), ("z", "<f4")])
    arr = np.zeros(n, dtype=dt)
    arr["x"], arr["y"], arr["z"] = CLOUD_XYZ[:n, 0], CLOUD_XYZ[:n, 1], CLOUD_XYZ[:n, 2]
    fields = [("x", 0, FLOAT32, 1), ("y", 8, FLOAT32, 1), ("z", 16, FLOAT32, 1)]
    xyz, rgb = mc.cloud_view(fields=fields, point_step=20, n_points=n, is_bigendian=False, data=arr.tobytes())
    np.testing.assert_array_equal(xyz, CLOUD_XYZ[:n])
    assert rgb is None


def test_cloud_view_empty_cloud():
    kw = point_cloud(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8))
    xyz, rgb = mc.cloud_view(**kw)
    assert xyz.shape == (0, 3) and rgb.shape == (0, 3)


def test_cloud_view_rejects_big_endian():
    kw = point_cloud(CLOUD_XYZ)
    kw["is_bigendian"] = True
    with pytest.raises(ValueError):
        mc.cloud_view(**kw)


def test_cloud_view_rejects_non_float32_xyz():
    kw = point_cloud(CLOUD_XYZ)
    kw["fields"][0] = ("x", 0, FLOAT64, 1)
    with pytest.raises(ValueError):
        mc.cloud_view(**kw)
    kw["fields"][0] = ("x", 0, UINT32, 1)
    with pytest.raises(ValueError):
        mc.cloud_view(**kw)


@pytest.mark.parametrize("missing", ["x", "y", "z"])
def test_cloud_view_rejects_missing_axis(missing):
    kw = point_cloud(CLOUD_XYZ)
    kw["fields"] = [f for f in kw["fields"] if f[0] != missing]
    with pytest.raises(ValueError):
        mc.cloud_view(**kw)


def test_cloud_view_rejects_short_buffer_and_field_past_point_step():
    kw = point_cloud(CLOUD_XYZ)
    with pytest.raises(ValueError):
        mc.cloud_view(**{**kw, "data": kw["data"][:-1]})
    with pytest.raises(ValueError):
        mc.cloud_view(**{**kw, "point_step": 10})


def test_cloud_view_feeds_encode_cloud():
    xyz, rgb = mc.cloud_view(**point_cloud(CLOUD_XYZ, CLOUD_RGB))
    assert mc.encode_cloud(xyz, rgb, budget=100, spacing_m=0.25, **KW) == build_golden()["cloud.bin"]


# ------------------------------------------------------------------------------------ trajectory


def test_trajectory_roundtrip_has_spec_length_and_3d_length():
    b = mc.encode_trajectory(TRAJ_POSES, **KW)
    assert len(b) == 32 + 28 * 3
    assert struct.unpack_from("<If", b, 24) == (3, 17.0)
    d = mapread.trajectory(b)
    assert (d["epoch"], d["seq"], d["stamp_s"], d["count"], d["length_m"]) == (7, 3, 1234.5, 3, 17.0)
    np.testing.assert_array_equal(d["poses"], TRAJ_POSES)


@pytest.mark.parametrize("poses", [np.zeros((0, 7), np.float32), []])
def test_trajectory_empty(poses):
    b = mc.encode_trajectory(poses, **KW)
    assert len(b) == 32
    d = mapread.trajectory(b)
    assert d["count"] == 0 and d["length_m"] == 0.0 and d["poses"].shape == (0, 7)


def test_trajectory_single_pose_has_zero_length():
    d = mapread.trajectory(mc.encode_trajectory(TRAJ_POSES[:1], **KW))
    assert d["count"] == 1 and d["length_m"] == 0.0


def test_trajectory_drops_poses_with_non_finite_values():
    poses = TRAJ_POSES.copy()
    poses[1, 3] = NAN
    d = mapread.trajectory(mc.encode_trajectory(poses, **KW))
    assert d["count"] == 2
    np.testing.assert_array_equal(d["poses"], TRAJ_POSES[[0, 2]])
    assert d["length_m"] == pytest.approx(float(np.linalg.norm(TRAJ_POSES[2, :3])), rel=1e-6)


def test_trajectory_rejects_wrong_width():
    with pytest.raises(ValueError):
        mc.encode_trajectory(np.zeros((3, 6), np.float32), **KW)


# ---------------------------------------------------------------------------------------- grid


def test_grid_roundtrip_has_spec_length():
    b = mc.encode_grid(GRID_CELLS, resolution=0.25, origin_xy=(-0.5, 1.0), origin_yaw=0.5, **KW)
    assert len(b) == 48 + 12
    assert struct.unpack_from("<IIffff", b, 24) == (4, 3, 0.25, -0.5, 1.0, 0.5)
    assert struct.unpack_from("<4b", b, 48) == (-1, 0, 10, 20)  # row 0 first
    d = mapread.grid(b)
    assert (d["epoch"], d["seq"], d["stamp_s"]) == (7, 3, 1234.5)
    assert (d["width"], d["height"], d["resolution_m"]) == (4, 3, 0.25)
    assert (d["origin_x"], d["origin_y"], d["origin_yaw"]) == (-0.5, 1.0, 0.5)
    assert d["cells"].dtype == np.int8
    np.testing.assert_array_equal(d["cells"], GRID_CELLS)


def test_grid_clamps_out_of_range_values_and_accepts_wider_dtypes():
    cells = np.array([[-5, -1, 0, 100, 101, 127]], dtype=np.int16)
    d = mapread.grid(mc.encode_grid(cells, resolution=0.1, origin_xy=(0, 0), origin_yaw=0.0, **KW))
    assert d["cells"][0].tolist() == [-1, -1, 0, 100, 100, 100]
    d = mapread.grid(mc.encode_grid(np.array([[0, 100, 200]], np.uint8), resolution=0.1, origin_xy=(0, 0), origin_yaw=0.0, **KW))
    assert d["cells"][0].tolist() == [0, 100, 100]


def test_grid_empty_and_bad_input():
    b = mc.encode_grid(np.zeros((0, 0), np.int8), resolution=0.1, origin_xy=(0, 0), origin_yaw=0.0, **KW)
    assert len(b) == 48 and mapread.grid(b)["cells"].size == 0
    with pytest.raises(ValueError):
        mc.encode_grid(np.zeros(5, np.int8), resolution=0.1, origin_xy=(0, 0), origin_yaw=0.0, **KW)
    with pytest.raises(ValueError):
        mc.encode_grid(np.zeros((2, 2), np.float32), resolution=0.1, origin_xy=(0, 0), origin_yaw=0.0, **KW)


# -------------------------------------------------------------------------------- golden fixtures


@pytest.mark.parametrize("name", GOLDEN_NAMES)
def test_golden_fixture_is_byte_identical_to_a_fresh_encode(name):
    path = FIXTURES / name
    assert path.is_file(), f"{path} is missing; run: python3 test/test_mapcodec.py --write"
    assert path.read_bytes() == build_golden()[name]


def test_fixtures_are_marked_binary_for_git():
    assert "*.bin binary" in (FIXTURES / ".gitattributes").read_text().splitlines()


if __name__ == "__main__":
    if "--write" not in sys.argv[1:]:
        raise SystemExit("usage: python3 test/test_mapcodec.py --write   (rewrites test/fixtures/map/*.bin)")
    FIXTURES.mkdir(parents=True, exist_ok=True)
    for fname, payload in build_golden().items():
        (FIXTURES / fname).write_bytes(payload)
        print(f"wrote {FIXTURES / fname} ({len(payload)} bytes)")
