// Pure helpers for the 3D map scene (scene.ts): where the camera goes, where the cost-grid quad lies, how big the
// ground grid is, the live terrain's height band and the car's size. No three.js and no DOM, so they are tested in node. World frame =
// the map frame: metres, x forward/east, y left/north, z up. (The view's toggles live in mapToggles.ts.)

export interface Vec3 { x: number; y: number; z: number }
export interface Bounds { minX: number; minY: number; minZ: number; maxX: number; maxY: number; maxZ: number }

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))

// ---- bounds ---------------------------------------------------------------------------------------
const isBounds = (b: Bounds) =>
  [b.minX, b.minY, b.minZ, b.maxX, b.maxY, b.maxZ].every(Number.isFinite) && b.minX <= b.maxX && b.minY <= b.maxY && b.minZ <= b.maxZ

// The box a cloud's header carries (the gateway computes it over the points it wrote), or null for an empty cloud
// or a box that is not finite and ordered. Using it avoids a pass over up to 500 000 points.
export function cloudBounds(f: { count: number; bboxMin: readonly number[]; bboxMax: readonly number[] }): Bounds | null {
  if (f.count <= 0) return null
  const b = { minX: f.bboxMin[0], minY: f.bboxMin[1], minZ: f.bboxMin[2], maxX: f.bboxMax[0], maxY: f.bboxMax[1], maxZ: f.bboxMax[2] }
  return isBounds(b) ? b : null
}

export function pointBounds(x: number, y: number, z: number): Bounds | null {
  const b = { minX: x, minY: y, minZ: z, maxX: x, maxY: y, maxZ: z }
  return isBounds(b) ? b : null
}

export function unionBounds(list: readonly (Bounds | null)[]): Bounds | null {
  let out: Bounds | null = null
  for (const b of list) {
    if (!b) continue
    out = out
      ? {
        minX: Math.min(out.minX, b.minX), minY: Math.min(out.minY, b.minY), minZ: Math.min(out.minZ, b.minZ),
        maxX: Math.max(out.maxX, b.maxX), maxY: Math.max(out.maxY, b.maxY), maxZ: Math.max(out.maxZ, b.maxZ),
      }
      : { ...b }
  }
  return out
}

// ---- pose -----------------------------------------------------------------------------------------
// Heading (rotation about +z) of a unit quaternion, ZYX convention: roll and pitch do not change it.
export function yawOf(qx: number, qy: number, qz: number, qw: number): number {
  return Math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
}

export interface PlanarPose { x: number; y: number; z: number; yaw: number }

// ---- camera ---------------------------------------------------------------------------------------
export const VIEW_PITCH_RAD = (55 * Math.PI) / 180 // camera elevation above the horizon, looking at the target
export const VIEW_SPAN_FACTOR = 0.8 // view distance per metre of the data's larger side
export const VIEW_MIN_DIST_M = 6
export const VIEW_MAX_DIST_M = 150
export const VIEW_AHEAD = 0.3 // the target sits this fraction of the view distance ahead of the robot

export interface View { position: Vec3; target: Vec3 }

// The home view, as in the owner's reference picture: from behind and above the robot, looking along its heading at
// a point a little ahead of it, so the robot sits low in the picture with what it sees fanning out above it. Without a
// pose: the centre of the data (on the ground when the data spans z = 0), looking along +x. The distance grows with
// the data's extent, within limits.
export function homeView(pose: PlanarPose | null, bounds: Bounds | null): View {
  const p = pose && [pose.x, pose.y, pose.z, pose.yaw].every(Number.isFinite) ? pose : null
  const b = bounds && isBounds(bounds) ? bounds : null
  const span = b ? Math.max(b.maxX - b.minX, b.maxY - b.minY) : 0
  const dist = clamp(span * VIEW_SPAN_FACTOR, VIEW_MIN_DIST_M, VIEW_MAX_DIST_M)
  const yaw = p ? p.yaw : 0
  const fx = Math.cos(yaw)
  const fy = Math.sin(yaw)
  let target: Vec3
  if (p) target = { x: p.x + fx * dist * VIEW_AHEAD, y: p.y + fy * dist * VIEW_AHEAD, z: p.z }
  else if (b) target = { x: (b.minX + b.maxX) / 2, y: (b.minY + b.maxY) / 2, z: clamp(0, b.minZ, b.maxZ) }
  else target = { x: 0, y: 0, z: 0 }
  const back = dist * Math.cos(VIEW_PITCH_RAD)
  const up = dist * Math.sin(VIEW_PITCH_RAD)
  return { target, position: { x: target.x - fx * back, y: target.y - fy * back, z: target.z + up } }
}

// ---- cost grid ------------------------------------------------------------------------------------
export interface GridPlacement {
  cx: number // centre of the quad in the map frame
  cy: number
  sizeX: number // metres along the grid's own x axis (columns)
  sizeY: number // metres along its y axis (rows)
  yaw: number // the quad's rotation about +z
  bounds: Bounds // of the four corners, at z = 0
}

// Where the cost grid's quad goes. The grid's origin is the outer corner of cell (row 0, column 0) and the grid turns
// about it by originYaw; the texture is not rotated (buildGridTexture ignores the yaw), so the quad is. A unit plane
// whose u follows +x and v follows +y, scaled to (sizeX, sizeY), turned by yaw and moved to (cx, cy), puts uv (0, 0) -
// texture row 0, column 0 - on the origin. null for an empty grid or one whose placement is not finite.
export function gridQuad(f: {
  width: number; height: number; resolution: number; originX: number; originY: number; originYaw: number
}): GridPlacement | null {
  const sizeX = f.width * f.resolution
  const sizeY = f.height * f.resolution
  if (!(sizeX > 0 && sizeY > 0) || ![sizeX, sizeY, f.originX, f.originY, f.originYaw].every(Number.isFinite)) return null
  const c = Math.cos(f.originYaw)
  const s = Math.sin(f.originYaw)
  const at = (lx: number, ly: number) => [f.originX + c * lx - s * ly, f.originY + s * lx + c * ly]
  const corners = [at(0, 0), at(sizeX, 0), at(0, sizeY), at(sizeX, sizeY)]
  const xs = corners.map((p) => p[0])
  const ys = corners.map((p) => p[1])
  const [cx, cy] = at(sizeX / 2, sizeY / 2)
  return {
    cx, cy, sizeX, sizeY, yaw: f.originYaw,
    bounds: { minX: Math.min(...xs), minY: Math.min(...ys), minZ: 0, maxX: Math.max(...xs), maxY: Math.max(...ys), maxZ: 0 },
  }
}

// ---- ground grid ----------------------------------------------------------------------------------
export const GROUND_MIN_SIZE_M = 20
export const GROUND_MARGIN_M = 2 // kept clear around the data on every side
export const GROUND_MAX_DIVISIONS = 120 // lines per side, whatever the map's size
const GROUND_CELLS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000] as const

export interface GroundGrid { cx: number; cy: number; size: number; cell: number } // divisions = size / cell

// A square ground grid covering the data plus a margin, with the smallest cell from GROUND_CELLS that keeps it within
// GROUND_MAX_DIVISIONS, an even number of cells and its centre on a multiple of the cell, so every line sits on a
// multiple of the cell and the lines stay put while the grid grows with the map. Data too large even for the biggest
// cell gets the biggest grid allowed, centred on it.
export function groundGridFor(bounds: Bounds | null): GroundGrid {
  if (!bounds || !isBounds(bounds)) return { cx: 0, cy: 0, size: GROUND_MIN_SIZE_M, cell: 1 }
  const span = Math.max(bounds.maxX - bounds.minX, bounds.maxY - bounds.minY) + 2 * GROUND_MARGIN_M
  const cell = GROUND_CELLS.find((c) => span + c <= GROUND_MAX_DIVISIONS * c) ?? GROUND_CELLS[GROUND_CELLS.length - 1]
  const cells = Math.min(GROUND_MAX_DIVISIONS, 2 * Math.ceil((span + cell) / (2 * cell))) // + cell: the centre snaps by up to half a cell
  const size = Math.max(GROUND_MIN_SIZE_M, cells * cell)
  const snap = (m: number) => Math.round(m / cell) * cell + 0 // + 0 turns -0 into 0
  return { cx: snap((bounds.minX + bounds.maxX) / 2), cy: snap((bounds.minY + bounds.maxY) / 2), size, cell }
}

// ---- live heat colours ----------------------------------------------------------------------------
// The live terrain and scan are coloured (blue low, through cyan, green and yellow, to red high) over a fixed band
// around the robot's base rather than their own min..max, so a colour means the same height from one scan to the next
// (outliers do not repaint it). The band is sized for a small RC car: the ground sits a fifth of the way up (blue),
// a 0.5 m obstacle is in the middle (green to yellow) and anything 1.2 m above the base or more is red.
export const HEAT_BAND_BELOW_M = 0.3
export const HEAT_BAND_ABOVE_M = 1.2

export function heatHeightBand(groundZ: number | null): [number, number] {
  const z0 = groundZ !== null && Number.isFinite(groundZ) ? groundZ : 0
  return [z0 - HEAT_BAND_BELOW_M, z0 + HEAT_BAND_ABOVE_M]
}

// ---- the car --------------------------------------------------------------------------------------
// A 1:10 RC car, drawn at the robot pose (base_link at its centre) and as the travelled path's width. A display size
// only: Nav2 plans with its own footprint (the D25 placeholder until Dev 5's footprint files exist).
export const CAR_LENGTH_M = 0.43
export const CAR_WIDTH_M = 0.2
