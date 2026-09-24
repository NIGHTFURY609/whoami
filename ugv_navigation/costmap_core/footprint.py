"""ROS-independent robot footprint validation and padding.

A footprint is a 2D polygon given as an ordered sequence of (x, y)
vertices in metres, in the robot's own frame (the ring is implicitly
closed: the last vertex connects back to the first, and the first vertex
must NOT be repeated at the end).

This is the Dev 3 core implementation only: no Nav2 footprint messages,
no TF, no robot description parsing, and no robot-specific dimensions.
The real UGV footprint is owned by Dev 5 and is not known yet, so this
module has no default footprint -- callers must supply one explicitly
(e.g. loaded from configuration once Dev 5 provides it).

Padding semantics (defined here, NOT copied from Nav2): every edge of the
polygon is moved outward, perpendicular to itself, by exactly `padding`
metres, and adjacent offset edges are joined at their intersection
(a mitered offset). For a rectangle this grows each half-width/half-length
by `padding`. This differs from Nav2's `padFootprint`, which shifts each
vertex by `padding` along the sign of its x and y coordinates; the two
agree for origin-centred axis-aligned rectangles but not in general.

Only convex footprints are supported. A mitered offset of a convex
polygon is always a valid convex polygon; for concave polygons it can
self-intersect, so concave input is rejected loudly rather than producing
an ill-defined padded footprint.
"""

from __future__ import annotations

import math
from typing import Sequence

Point = tuple[float, float]


class FootprintError(ValueError):
    """Raised when a footprint polygon or padding value is invalid.

    Deliberately not silently coerced -- a malformed footprint must fail
    loudly rather than silently produce a collision shape with undefined
    safety meaning.
    """


def _require_finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FootprintError(f"{name} must be a real number, got {type(value).__name__}")
    value = float(value)
    if not math.isfinite(value):
        raise FootprintError(f"{name} must be finite, got {value!r}")
    return value


def _cross(o: Point, a: Point, b: Point) -> float:
    """z-component of (a - o) x (b - o)."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _segments_intersect(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    """True if closed segments p1-p2 and q1-q2 share any point."""
    d1 = _cross(q1, q2, p1)
    d2 = _cross(q1, q2, p2)
    d3 = _cross(p1, p2, q1)
    d4 = _cross(p1, p2, q2)
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)) and 0 not in (d1, d2, d3, d4):
        return True

    def on_segment(a: Point, b: Point, c: Point) -> bool:
        return (
            min(a[0], b[0]) <= c[0] <= max(a[0], b[0])
            and min(a[1], b[1]) <= c[1] <= max(a[1], b[1])
        )

    return (
        (d1 == 0 and on_segment(q1, q2, p1))
        or (d2 == 0 and on_segment(q1, q2, p2))
        or (d3 == 0 and on_segment(p1, p2, q1))
        or (d4 == 0 and on_segment(p1, p2, q2))
    )


def signed_area(footprint: Sequence[Point]) -> float:
    """Shoelace signed area: > 0 for counter-clockwise, < 0 for clockwise."""
    n = len(footprint)
    total = 0.0
    for i in range(n):
        x0, y0 = footprint[i]
        x1, y1 = footprint[(i + 1) % n]
        total += x0 * y1 - x1 * y0
    return 0.5 * total


def validate_footprint(footprint: Sequence[Sequence[float]]) -> list[Point]:
    """Validate a footprint polygon and return a normalised copy.

    Requirements:
    - at least 3 vertices, each a pair of finite real numbers;
    - no two consecutive vertices equal (including last -> first, so the
      ring must not be explicitly closed);
    - non-zero area;
    - simple (no self-intersections) and convex. Collinear vertices are
      allowed. Either winding direction is accepted.

    Returns:
        A new list of (float, float) tuples, same order as the input.
        `footprint` itself is never mutated.

    Raises:
        FootprintError: if any requirement above is violated.
    """
    if isinstance(footprint, (str, bytes)) or not isinstance(footprint, Sequence):
        raise FootprintError(
            f"footprint must be a sequence of (x, y) points, got {type(footprint).__name__}"
        )

    points: list[Point] = []
    for i, vertex in enumerate(footprint):
        if isinstance(vertex, (str, bytes)) or not isinstance(vertex, Sequence) or len(vertex) != 2:
            raise FootprintError(f"footprint vertex {i} must be an (x, y) pair, got {vertex!r}")
        x = _require_finite_number(vertex[0], name=f"footprint vertex {i} x")
        y = _require_finite_number(vertex[1], name=f"footprint vertex {i} y")
        points.append((x, y))

    n = len(points)
    if n < 3:
        raise FootprintError(f"footprint must have at least 3 vertices, got {n}.")

    for i in range(n):
        if points[i] == points[(i + 1) % n]:
            raise FootprintError(
                f"footprint vertices {i} and {(i + 1) % n} are identical {points[i]!r}; "
                "consecutive vertices must differ and the ring must not be explicitly closed."
            )

    if signed_area(points) == 0.0:
        raise FootprintError("footprint has zero area (all vertices collinear).")

    # Convexity: every turn must go the same way (collinear turns allowed).
    turn_sign = 0
    for i in range(n):
        turn = _cross(points[i], points[(i + 1) % n], points[(i + 2) % n])
        if turn == 0.0:
            continue
        sign = 1 if turn > 0 else -1
        if turn_sign == 0:
            turn_sign = sign
        elif sign != turn_sign:
            raise FootprintError(
                f"footprint is not convex (turn direction changes at vertex {(i + 1) % n}); "
                "only convex footprints are supported."
            )

    # Consistent turning alone admits self-winding shapes (e.g. a pentagram),
    # so also require that no two non-adjacent edges touch.
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue  # edges n-1 and 0 are adjacent
            if _segments_intersect(points[i], points[(i + 1) % n], points[j], points[(j + 1) % n]):
                raise FootprintError(
                    f"footprint is self-intersecting (edges {i} and {j} intersect)."
                )

    return points


def validate_padding(padding: object) -> float:
    """Validate a padding distance in metres. Must be finite and >= 0."""
    padding = _require_finite_number(padding, name="padding")
    if padding < 0.0:
        raise FootprintError(f"padding must be >= 0, got {padding!r}.")
    return padding


def pad_footprint(footprint: Sequence[Sequence[float]], padding: float) -> list[Point]:
    """Expand a convex footprint outward by `padding` metres.

    Every edge is moved outward, perpendicular to itself, by `padding`,
    and each vertex is moved to the intersection of its two offset edges
    (mitered corners). See the module docstring for how this relates to
    Nav2's `padFootprint`.

    Args:
        footprint: convex polygon as a sequence of (x, y) vertices in
            metres; see `validate_footprint` for all requirements.
        padding: outward offset distance in metres. Must be finite and
            >= 0. A padding of 0 returns an unchanged copy.

    Returns:
        A new list of (float, float) tuples with the same vertex count and
        winding direction as the input. `footprint` is never mutated.

    Raises:
        FootprintError: if `footprint` or `padding` is invalid.
    """
    points = validate_footprint(footprint)
    padding = validate_padding(padding)
    if padding == 0.0:
        return points

    n = len(points)
    # Outward unit normal of edge i (points[i] -> points[i+1]): the right-hand
    # normal for a counter-clockwise ring, the left-hand one for clockwise.
    ccw = signed_area(points) > 0.0
    normals: list[Point] = []
    for i in range(n):
        (x0, y0), (x1, y1) = points[i], points[(i + 1) % n]
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if ccw:
            normals.append((dy / length, -dx / length))
        else:
            normals.append((-dy / length, dx / length))

    padded: list[Point] = []
    for i in range(n):
        # Vertex i joins edge i-1 (incoming) and edge i (outgoing). The miter
        # point v + padding * (n1 + n2) / (1 + n1.n2) lies at distance
        # `padding` from both offset edge lines. For a convex polygon the
        # interior angle is < 180 deg, so n1.n2 > -1 and this is well defined;
        # collinear vertices (n1 == n2) reduce to v + padding * n1.
        n1 = normals[i - 1]
        n2 = normals[i]
        scale = padding / (1.0 + n1[0] * n2[0] + n1[1] * n2[1])
        x, y = points[i]
        padded.append((x + scale * (n1[0] + n2[0]), y + scale * (n1[1] + n2[1])))

    return padded
