import copy
import math

import pytest

from costmap_core.footprint import (
    FootprintError,
    pad_footprint,
    signed_area,
    validate_footprint,
    validate_padding,
)

# TEST-ONLY SYNTHETIC FOOTPRINTS. These are NOT the real UGV footprint
# (which Dev 5 has not provided yet) and must not be used as robot config.
SYNTHETIC_HALF_LENGTH = 0.5  # test-only value, metres
SYNTHETIC_HALF_WIDTH = 0.3  # test-only value, metres
SYNTHETIC_RECTANGLE_CCW = [
    (SYNTHETIC_HALF_LENGTH, SYNTHETIC_HALF_WIDTH),
    (-SYNTHETIC_HALF_LENGTH, SYNTHETIC_HALF_WIDTH),
    (-SYNTHETIC_HALF_LENGTH, -SYNTHETIC_HALF_WIDTH),
    (SYNTHETIC_HALF_LENGTH, -SYNTHETIC_HALF_WIDTH),
]
SYNTHETIC_RECTANGLE_CW = list(reversed(SYNTHETIC_RECTANGLE_CCW))
SYNTHETIC_TRIANGLE = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]  # test-only
SYNTHETIC_PENTAGON = [  # test-only regular pentagon, circumradius 1.0
    (math.cos(math.radians(90 + 72 * k)), math.sin(math.radians(90 + 72 * k))) for k in range(5)
]

APPROX = pytest.approx


def _distance_to_line(p, a, b):
    (px, py), (ax, ay), (bx, by) = p, a, b
    return abs((bx - ax) * (ay - py) - (ax - px) * (by - ay)) / math.hypot(bx - ax, by - ay)


def _assert_points_close(actual, expected):
    assert len(actual) == len(expected)
    for (ax, ay), (ex, ey) in zip(actual, expected):
        assert ax == APPROX(ex)
        assert ay == APPROX(ey)


# --- validation: valid footprints ---------------------------------------


@pytest.mark.parametrize(
    "footprint",
    [SYNTHETIC_RECTANGLE_CCW, SYNTHETIC_RECTANGLE_CW, SYNTHETIC_TRIANGLE, SYNTHETIC_PENTAGON],
)
def test_valid_footprint_is_accepted_and_returned_as_float_tuples(footprint):
    result = validate_footprint(footprint)
    _assert_points_close(result, footprint)
    assert all(isinstance(p, tuple) and all(isinstance(c, float) for c in p) for p in result)


def test_valid_footprint_accepts_lists_and_ints():
    result = validate_footprint([[0, 0], [2, 0], [2, 1], [0, 1]])
    assert result == [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)]


def test_valid_footprint_allows_collinear_vertex():
    # Midpoint on the bottom edge of a synthetic 2x1 rectangle.
    validate_footprint([(0, 0), (1, 0), (2, 0), (2, 1), (0, 1)])


# --- validation: invalid footprints -------------------------------------


@pytest.mark.parametrize(
    "footprint",
    [
        [],
        [(0, 0), (1, 0)],  # too few vertices
        [(0, 0), (1, 0), (2, 0)],  # zero area
        [(0, 0), (1, 0), (1, 0), (0, 1)],  # repeated consecutive vertex
        [(0, 0), (1, 0), (0, 1), (0, 0)],  # explicitly closed ring
        [(0, 0), (1, 1), (1, 0), (0, 1)],  # bow-tie self-intersection
        [(0, 0), (2, 0), (1, 0.5), (2, 2), (0, 2)],  # concave
        [(0, 0), (1, 0), (math.nan, 1)],  # non-finite
        [(0, 0), (1, 0), (math.inf, 1)],  # non-finite
        [(0, 0), (1, 0), (0, 1, 2)],  # vertex not a pair
        [(0, 0), (1, 0), ("0", 1)],  # non-numeric coordinate
        [(0, 0), (1, 0), (True, 1)],  # bool coordinate
        "abc",
        None,
    ],
)
def test_invalid_footprint_is_rejected(footprint):
    with pytest.raises(FootprintError):
        validate_footprint(footprint)


def test_self_winding_star_is_rejected():
    # Pentagram: every turn has the same direction, but edges cross.
    star = [SYNTHETIC_PENTAGON[(2 * k) % 5] for k in range(5)]
    with pytest.raises(FootprintError, match="self-intersecting"):
        validate_footprint(star)


def test_invalid_footprint_is_rejected_by_pad_footprint():
    with pytest.raises(FootprintError):
        pad_footprint([(0, 0), (1, 0)], 0.1)


# --- padding validation -------------------------------------------------


@pytest.mark.parametrize("padding", [0, 0.0, 0.1, 2])
def test_valid_padding_is_accepted(padding):
    assert validate_padding(padding) == float(padding)


@pytest.mark.parametrize("padding", [-0.01, -1, math.nan, math.inf, -math.inf, None, "0.1", True])
def test_invalid_padding_is_rejected(padding):
    with pytest.raises(FootprintError):
        validate_padding(padding)


def test_negative_padding_rejected_by_pad_footprint():
    with pytest.raises(FootprintError, match=">= 0"):
        pad_footprint(SYNTHETIC_RECTANGLE_CCW, -0.1)


# --- padding behaviour --------------------------------------------------


def test_zero_padding_returns_unchanged_copy():
    result = pad_footprint(SYNTHETIC_RECTANGLE_CCW, 0.0)
    assert result == SYNTHETIC_RECTANGLE_CCW
    assert result is not SYNTHETIC_RECTANGLE_CCW


@pytest.mark.parametrize("rectangle", [SYNTHETIC_RECTANGLE_CCW, SYNTHETIC_RECTANGLE_CW])
def test_rectangle_padding_grows_each_half_dimension_by_padding(rectangle):
    padding = 0.1  # test-only value, metres
    result = pad_footprint(rectangle, padding)
    hl = SYNTHETIC_HALF_LENGTH + padding
    hw = SYNTHETIC_HALF_WIDTH + padding
    expected = [(math.copysign(hl, x), math.copysign(hw, y)) for x, y in rectangle]
    _assert_points_close(result, expected)


def test_off_centre_rectangle_is_padded_outward_not_away_from_origin():
    # Synthetic rectangle entirely in +x: the rear edge (x=1) must move to
    # x=0.9, i.e. outward from the polygon, not away from the origin.
    result = pad_footprint([(1, 0), (3, 0), (3, 1), (1, 1)], 0.1)
    _assert_points_close(result, [(0.9, -0.1), (3.1, -0.1), (3.1, 1.1), (0.9, 1.1)])


@pytest.mark.parametrize(
    "footprint",
    [SYNTHETIC_RECTANGLE_CCW, SYNTHETIC_RECTANGLE_CW, SYNTHETIC_TRIANGLE, SYNTHETIC_PENTAGON],
)
@pytest.mark.parametrize("padding", [0.05, 0.25, 1.0])
def test_every_edge_moves_outward_by_exactly_padding(footprint, padding):
    result = pad_footprint(footprint, padding)
    n = len(footprint)
    for i in range(n):
        a, b = footprint[i], footprint[(i + 1) % n]
        pa, pb = result[i], result[(i + 1) % n]
        # Padded edge endpoints lie on a line exactly `padding` from the original edge line.
        assert _distance_to_line(pa, a, b) == APPROX(padding)
        assert _distance_to_line(pb, a, b) == APPROX(padding)
    # ...and on the outside: every original vertex is inside the padded polygon.
    for vertex in footprint:
        assert _point_strictly_inside_convex(vertex, result)


def _point_strictly_inside_convex(p, polygon):
    n = len(polygon)
    crosses = []
    for i in range(n):
        (ax, ay), (bx, by) = polygon[i], polygon[(i + 1) % n]
        crosses.append((bx - ax) * (p[1] - ay) - (by - ay) * (p[0] - ax))
    return all(c > 0 for c in crosses) or all(c < 0 for c in crosses)


@pytest.mark.parametrize(
    "footprint",
    [SYNTHETIC_RECTANGLE_CCW, SYNTHETIC_RECTANGLE_CW, SYNTHETIC_TRIANGLE, SYNTHETIC_PENTAGON],
)
@pytest.mark.parametrize("padding", [0.05, 0.25, 1.0])
def test_padded_footprint_remains_geometrically_valid(footprint, padding):
    result = pad_footprint(footprint, padding)
    assert validate_footprint(result) == result
    # Same vertex count, same winding direction, strictly larger area.
    assert len(result) == len(footprint)
    assert math.copysign(1, signed_area(result)) == math.copysign(1, signed_area(footprint))
    assert abs(signed_area(result)) > abs(signed_area(footprint))


def test_collinear_vertex_moves_straight_out():
    result = pad_footprint([(0, 0), (1, 0), (2, 0), (2, 1), (0, 1)], 0.1)
    _assert_points_close(
        result, [(-0.1, -0.1), (1.0, -0.1), (2.1, -0.1), (2.1, 1.1), (-0.1, 1.1)]
    )


def test_input_footprint_is_not_mutated():
    footprint = [[0.5, 0.3], [-0.5, 0.3], [-0.5, -0.3], [0.5, -0.3]]  # test-only, mutable lists
    snapshot = copy.deepcopy(footprint)
    pad_footprint(footprint, 0.2)
    validate_footprint(footprint)
    assert footprint == snapshot

    rect_snapshot = list(SYNTHETIC_RECTANGLE_CCW)
    pad_footprint(SYNTHETIC_RECTANGLE_CCW, 0.2)
    assert SYNTHETIC_RECTANGLE_CCW == rect_snapshot
