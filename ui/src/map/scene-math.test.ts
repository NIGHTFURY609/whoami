import { describe, expect, it } from 'vitest'
import {
  GROUND_MARGIN_M, GROUND_MAX_DIVISIONS, GROUND_MIN_SIZE_M, HEAT_BAND_ABOVE_M, HEAT_BAND_BELOW_M, VIEW_MAX_DIST_M,
  VIEW_MIN_DIST_M, cloudBounds, gridQuad, groundGridFor, heatHeightBand, homeView, pointBounds, unionBounds, yawOf,
  type Bounds,
} from './scene-math'

const box = (minX: number, minY: number, minZ: number, maxX: number, maxY: number, maxZ: number): Bounds =>
  ({ minX, minY, minZ, maxX, maxY, maxZ })

const yawQuat = (yaw: number) => ({ qx: 0, qy: 0, qz: Math.sin(yaw / 2), qw: Math.cos(yaw / 2) })

describe('yawOf', () => {
  it.each([0, 0.5, Math.PI / 2, -2, 3])('reads the heading of a pure z rotation (%s rad)', (yaw) => {
    const q = yawQuat(yaw)
    expect(yawOf(q.qx, q.qy, q.qz, q.qw)).toBeCloseTo(yaw, 9)
  })

  it('ignores roll and pitch around the heading', () => {
    // 30 degree roll about x, then 1 rad of yaw: the heading is still 1 rad
    const r = Math.PI / 6
    const [cr, sr, cy, sy] = [Math.cos(r / 2), Math.sin(r / 2), Math.cos(0.5), Math.sin(0.5)]
    // q = qz(yaw) * qx(roll)
    expect(yawOf(cy * sr, sy * sr, sy * cr, cy * cr)).toBeCloseTo(1, 9)
  })
})

describe('bounds', () => {
  it('reads a cloud header box, and refuses an empty or broken one', () => {
    expect(cloudBounds({ count: 3, bboxMin: [-1, -2, -0.5], bboxMax: [4, 5, 2] })).toEqual(box(-1, -2, -0.5, 4, 5, 2))
    expect(cloudBounds({ count: 0, bboxMin: [0, 0, 0], bboxMax: [0, 0, 0] })).toBeNull()
    expect(cloudBounds({ count: 3, bboxMin: [Number.NaN, 0, 0], bboxMax: [1, 1, 1] })).toBeNull()
    expect(cloudBounds({ count: 3, bboxMin: [2, 0, 0], bboxMax: [1, 1, 1] })).toBeNull()
  })

  it('makes a point box and unions boxes, skipping the absent ones', () => {
    expect(pointBounds(1, 2, 3)).toEqual(box(1, 2, 3, 1, 2, 3))
    expect(pointBounds(Number.NaN, 2, 3)).toBeNull()
    expect(unionBounds([null, box(0, 0, 0, 1, 1, 1), null, box(-2, 0.5, -1, 0.5, 3, 0)])).toEqual(box(-2, 0, -1, 1, 3, 1))
    expect(unionBounds([null, null])).toBeNull()
    expect(unionBounds([])).toBeNull()
  })
})

describe('homeView', () => {
  const dist = (v: { position: { x: number; y: number; z: number }; target: { x: number; y: number; z: number } }) =>
    Math.hypot(v.position.x - v.target.x, v.position.y - v.target.y, v.position.z - v.target.z)

  it.each([0, Math.PI / 2, -2.5])('looks over the robot from behind and above, along its heading (yaw %s)', (yaw) => {
    const pose = { x: 3, y: -1, z: 0.2, yaw }
    const v = homeView(pose, box(-5, -5, -1, 5, 5, 2))
    const fx = Math.cos(yaw)
    const fy = Math.sin(yaw)
    // the camera is behind the robot and above it
    expect((v.position.x - pose.x) * fx + (v.position.y - pose.y) * fy).toBeLessThan(0)
    expect(v.position.z).toBeGreaterThan(pose.z + 1)
    // the target lies ahead of the robot on its heading, at its height (the robot sits low in the picture)
    const ahead = (v.target.x - pose.x) * fx + (v.target.y - pose.y) * fy
    expect(ahead).toBeGreaterThan(0)
    expect(Math.abs((v.target.x - pose.x) * -fy + (v.target.y - pose.y) * fx)).toBeLessThan(1e-9)
    expect(v.target.z).toBeCloseTo(pose.z, 12)
    // the view direction has no sideways component: the camera looks straight along the heading
    const dx = v.target.x - v.position.x
    const dy = v.target.y - v.position.y
    expect(Math.abs(dx * -fy + dy * fx)).toBeLessThan(1e-9)
    expect(dx * fx + dy * fy).toBeGreaterThan(0)
  })

  it('without a pose looks at the centre of the data along +x, on the ground when the data spans it', () => {
    const v = homeView(null, box(10, 20, -1, 30, 24, 3))
    expect(v.target).toEqual({ x: 20, y: 22, z: 0 })
    expect(v.position.x).toBeLessThan(v.target.x)
    expect(v.position.y).toBeCloseTo(22, 12)
    expect(v.position.z).toBeGreaterThan(0)
    // data entirely above the ground: the target is the nearest height inside it
    expect(homeView(null, box(0, 0, 2, 1, 1, 5)).target.z).toBe(2)
  })

  it('backs off with the size of the data, within limits', () => {
    const small = homeView(null, box(0, 0, 0, 1, 1, 1))
    const mid = homeView(null, box(0, 0, 0, 40, 10, 1))
    const huge = homeView(null, box(0, 0, 0, 1e5, 1, 1))
    expect(dist(small)).toBeCloseTo(VIEW_MIN_DIST_M, 9)
    expect(dist(mid)).toBeGreaterThan(VIEW_MIN_DIST_M)
    expect(dist(mid)).toBeLessThan(VIEW_MAX_DIST_M)
    expect(dist(huge)).toBeCloseTo(VIEW_MAX_DIST_M, 9)
  })

  it('has a sane default with nothing known, and ignores a non-finite pose or box', () => {
    const v = homeView(null, null)
    expect(v.target).toEqual({ x: 0, y: 0, z: 0 })
    expect(dist(v)).toBeCloseTo(VIEW_MIN_DIST_M, 9)
    expect(homeView({ x: Number.NaN, y: 0, z: 0, yaw: 0 }, null)).toEqual(v)
    expect(homeView(null, box(0, 0, 0, Number.POSITIVE_INFINITY, 1, 1))).toEqual(v)
    for (const n of Object.values(homeView({ x: 1, y: 1, z: 0, yaw: 1 }, null))) {
      expect(Number.isFinite(n.x) && Number.isFinite(n.y) && Number.isFinite(n.z)).toBe(true)
    }
  })
})

describe('gridQuad', () => {
  const grid = (o: Partial<{ width: number; height: number; resolution: number; originX: number; originY: number; originYaw: number }>) =>
    ({ width: 40, height: 20, resolution: 0.05, originX: -1, originY: 2, originYaw: 0, ...o })

  // The quad's (u, v) corner in the world, from the placement: centre + R(yaw) * ((u - 0.5) sizeX, (v - 0.5) sizeY).
  const corner = (q: NonNullable<ReturnType<typeof gridQuad>>, u: number, v: number) => {
    const lx = (u - 0.5) * q.sizeX
    const ly = (v - 0.5) * q.sizeY
    return [q.cx + Math.cos(q.yaw) * lx - Math.sin(q.yaw) * ly, q.cy + Math.sin(q.yaw) * lx + Math.cos(q.yaw) * ly]
  }

  it('sizes the quad from cells and resolution, and centres it from the origin corner', () => {
    const q = gridQuad(grid({}))!
    expect(q.sizeX).toBeCloseTo(2, 12)
    expect(q.sizeY).toBeCloseTo(1, 12)
    expect(q.cx).toBeCloseTo(0, 12)
    expect(q.cy).toBeCloseTo(2.5, 12)
    expect(q.yaw).toBe(0)
  })

  it.each([0, Math.PI / 2, 0.7, -2])('puts texture row 0 / column 0 (uv 0,0) on the origin and turns the grid about it (yaw %s)', (yaw) => {
    const q = gridQuad(grid({ originYaw: yaw }))!
    const [x0, y0] = corner(q, 0, 0)
    expect(x0).toBeCloseTo(-1, 9)
    expect(y0).toBeCloseTo(2, 9)
    // +u runs along the grid's x axis (the origin yaw), +v along its y axis
    const [xu, yu] = corner(q, 1, 0)
    expect(xu - x0).toBeCloseTo(2 * Math.cos(yaw), 9)
    expect(yu - y0).toBeCloseTo(2 * Math.sin(yaw), 9)
    const [xv, yv] = corner(q, 0, 1)
    expect(xv - x0).toBeCloseTo(-1 * Math.sin(yaw), 9)
    expect(yv - y0).toBeCloseTo(1 * Math.cos(yaw), 9)
  })

  it('bounds the turned quad by its four corners', () => {
    const q = gridQuad(grid({ originX: 0, originY: 0, originYaw: Math.PI / 2 }))!
    // 2 m along +y, 1 m along -x
    expect(q.bounds.minX).toBeCloseTo(-1, 9)
    expect(q.bounds.maxX).toBeCloseTo(0, 9)
    expect(q.bounds.minY).toBeCloseTo(0, 9)
    expect(q.bounds.maxY).toBeCloseTo(2, 9)
  })

  it('refuses an empty grid or one it cannot place', () => {
    expect(gridQuad(grid({ width: 0 }))).toBeNull()
    expect(gridQuad(grid({ height: 0 }))).toBeNull()
    expect(gridQuad(grid({ resolution: 0 }))).toBeNull()
    expect(gridQuad(grid({ resolution: Number.NaN }))).toBeNull()
    expect(gridQuad(grid({ originX: Number.POSITIVE_INFINITY }))).toBeNull()
    expect(gridQuad(grid({ originYaw: Number.NaN }))).toBeNull()
  })
})

describe('groundGridFor', () => {
  // the data plus the margin on every side, after the centre snapped to the cell
  const covers = (g: ReturnType<typeof groundGridFor>, b: Bounds) =>
    g.cx - g.size / 2 <= b.minX - GROUND_MARGIN_M && g.cx + g.size / 2 >= b.maxX + GROUND_MARGIN_M &&
    g.cy - g.size / 2 <= b.minY - GROUND_MARGIN_M && g.cy + g.size / 2 >= b.maxY + GROUND_MARGIN_M

  it('defaults to a small grid around the origin', () => {
    expect(groundGridFor(null)).toEqual({ cx: 0, cy: 0, size: GROUND_MIN_SIZE_M, cell: 1 })
    expect(groundGridFor(box(0, 0, 0, Number.NaN, 1, 1))).toEqual({ cx: 0, cy: 0, size: GROUND_MIN_SIZE_M, cell: 1 })
  })

  it.each([
    box(-1, -1, 0, 1, 1, 0), box(3.3, -7.2, 0, 9.9, 4.1, 0), box(-60, -10, 0, 25, 30, 0), box(100, 200, 0, 480, 230, 0),
    box(-3000, -3000, 0, 2500, 100, 0), box(-0.4, -0.4, 0, -0.1, -0.1, 0), box(0.5, 0, 0, 36.5, 1, 0),
  ])('covers the data and its margin with whole cells, centred on a line, within the division cap (%o)', (b) => {
    const g = groundGridFor(b)
    expect(covers(g, b)).toBe(true)
    expect(g.size).toBeGreaterThanOrEqual(GROUND_MIN_SIZE_M)
    const divisions = g.size / g.cell
    expect(Number.isInteger(divisions)).toBe(true)
    expect(divisions % 2).toBe(0) // the centre is on a line, so every line sits on a multiple of the cell
    expect(divisions).toBeLessThanOrEqual(GROUND_MAX_DIVISIONS)
    expect(Number.isInteger(g.cx / g.cell) && Number.isInteger(g.cy / g.cell)).toBe(true)
    expect(Object.is(g.cx, -0) || Object.is(g.cy, -0)).toBe(false)
  })

  it('keeps 1 m cells for a room-sized map and grows the cell for a large one', () => {
    expect(groundGridFor(box(0, 0, 0, 30, 30, 0)).cell).toBe(1)
    expect(groundGridFor(box(0, 0, 0, 400, 50, 0)).cell).toBeGreaterThan(1)
  })

  it('does not change for data that moves inside the same extent', () => {
    expect(groundGridFor(box(1, 1, 0, 3, 3, 0))).toEqual(groundGridFor(box(1.2, 1.1, 0, 3.1, 2.9, 0)))
  })
})

describe('heatHeightBand', () => {
  it('is a fixed band around the robot base, with the ground a fifth of the way up', () => {
    expect(heatHeightBand(0)).toEqual([-HEAT_BAND_BELOW_M, HEAT_BAND_ABOVE_M])
    expect(heatHeightBand(1.5)).toEqual([1.5 - HEAT_BAND_BELOW_M, 1.5 + HEAT_BAND_ABOVE_M])
    expect(HEAT_BAND_BELOW_M / (HEAT_BAND_BELOW_M + HEAT_BAND_ABOVE_M)).toBeCloseTo(0.2, 12)
  })

  it('uses the map ground when the robot height is not known', () => {
    expect(heatHeightBand(null)).toEqual(heatHeightBand(0))
    expect(heatHeightBand(Number.NaN)).toEqual(heatHeightBand(0))
  })
})
