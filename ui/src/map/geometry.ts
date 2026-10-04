// Pure geometry and colour builders for the 3D map view. They turn the decoded layer frames (codec.ts) into typed
// arrays a three.js scene can upload as they are, and they import nothing from three.js, so they run (and are
// tested) without WebGL. Every builder is one or two passes over typed arrays and allocates only its result:
// they run on the main thread. (Point heights are coloured in the scene's vertex shader, not here.)
//
// Conventions the scene must follow:
//   World frame   map coordinates in metres, x east-ish, y north-ish, z up.
//   Textures      row 0 of a texture is row 0 of the grid / depth image, row-major, RGBA. For the cost grid that is
//                 the row at originY, so a texture with flipY = false laid on a plane whose v axis follows +y needs
//                 no flip. Alpha is straight (not premultiplied).
//   Colours       the cost grid's halo uses writeRamp: dark purple (low) through teal to yellow (high). Bytes are
//                 sRGB.

import type { GridFrame } from './codec'

// ---- the one colour ramp --------------------------------------------------------------------------
// A perceptually ordered ramp (the viridis samples at eighths), lightness rising monotonically from low to high.
const RAMP_STOPS: readonly (readonly [number, number, number])[] = [
  [68, 1, 84], [71, 44, 122], [59, 82, 139], [44, 113, 142], [33, 144, 141],
  [39, 173, 129], [92, 200, 99], [170, 220, 50], [253, 231, 37],
]

const RAMP_STEPS = 256
const RAMP_LUT = (() => {
  const lut = new Uint8Array(RAMP_STEPS * 3)
  const last = RAMP_STOPS.length - 1
  for (let i = 0; i < RAMP_STEPS; i++) {
    const pos = (i / (RAMP_STEPS - 1)) * last
    const lo = Math.min(Math.floor(pos), last - 1)
    const k = pos - lo
    for (let ch = 0; ch < 3; ch++) lut[3 * i + ch] = Math.round(RAMP_STOPS[lo][ch] + (RAMP_STOPS[lo + 1][ch] - RAMP_STOPS[lo][ch]) * k)
  }
  return lut
})()

type Bytes = Uint8Array | Uint8ClampedArray

// Writes the ramp colour for t in [0, 1] as three bytes at out[offset..offset + 2]. t is clamped; NaN reads as 0.
export function writeRamp(t: number, out: Bytes, offset: number): void {
  const i = t > 0 ? (t < 1 ? Math.round(t * (RAMP_STEPS - 1)) : RAMP_STEPS - 1) : 0
  out[offset] = RAMP_LUT[3 * i]
  out[offset + 1] = RAMP_LUT[3 * i + 1]
  out[offset + 2] = RAMP_LUT[3 * i + 2]
}

// ---- the travelled path ---------------------------------------------------------------------------
// A flat ribbon `widthM` wide along a path (x y z per point, map frame), `liftM` above it, as two vertices per point
// (left, right of the path's direction there) and two triangles per segment. Points closer than MIN_STEP_M to the
// one kept before them are skipped (a robot standing still adds no zero-length segment, whose direction is
// undefined); a non-finite point is skipped too. null when fewer than two points are left.
const MIN_STEP_M = 0.005

export interface Ribbon {
  positions: Float32Array // 3 per vertex: vertex 2i is left of point i, 2i + 1 right of it
  index: Uint32Array // 6 per segment
  lengthM: number // along the kept points, in the plane
}

export function ribbonStrip(path: Float32Array, widthM: number, liftM = 0): Ribbon | null {
  const n = Math.floor(path.length / 3)
  const kept: number[] = []
  let lastX = Number.NaN
  let lastY = Number.NaN
  for (let i = 0; i < n; i++) {
    const x = path[3 * i]
    const y = path[3 * i + 1]
    if (![x, y, path[3 * i + 2]].every(Number.isFinite)) continue
    if (kept.length > 0 && Math.hypot(x - lastX, y - lastY) < MIN_STEP_M) continue
    kept.push(i)
    lastX = x
    lastY = y
  }
  const m = kept.length
  if (m < 2) return null
  const half = widthM / 2
  const positions = new Float32Array(6 * m)
  let lengthM = 0
  for (let k = 0; k < m; k++) {
    const i = kept[k]
    const prev = kept[Math.max(0, k - 1)]
    const next = kept[Math.min(m - 1, k + 1)]
    // direction at the point: from its neighbour before to its neighbour after (the segment itself at the ends)
    let dx = path[3 * next] - path[3 * prev]
    let dy = path[3 * next + 1] - path[3 * prev + 1]
    let len = Math.hypot(dx, dy)
    if (len < 1e-9) { // a path that doubles straight back: use the segment into the point
      dx = path[3 * i] - path[3 * prev]
      dy = path[3 * i + 1] - path[3 * prev + 1]
      len = Math.hypot(dx, dy) || 1
    }
    const nx = (-dy / len) * half // left normal
    const ny = (dx / len) * half
    const x = path[3 * i]
    const y = path[3 * i + 1]
    const z = path[3 * i + 2] + liftM
    positions.set([x + nx, y + ny, z, x - nx, y - ny, z], 6 * k)
    if (k > 0) lengthM += Math.hypot(x - path[3 * kept[k - 1]], y - path[3 * kept[k - 1] + 1])
  }
  const index = new Uint32Array(6 * (m - 1))
  for (let k = 0; k < m - 1; k++) {
    const a = 2 * k
    index.set([a, a + 1, a + 2, a + 1, a + 3, a + 2], 6 * k)
  }
  return { positions, index, lengthM }
}

// ---- textures -------------------------------------------------------------------------------------
const PATH_COLOR = [60, 200, 120] as const // cost 0: free, drawn as the painted pathway
const PATH_ALPHA = 110
const INSCRIBED_COLOR = [255, 140, 0] as const // cost 99: the footprint would touch an obstacle
const LETHAL_COLOR = [230, 30, 40] as const // cost 100 and above
const HALO_ALPHA_MIN = 40 // the faintest halo cell (cost 1)
const HALO_ALPHA_SPAN = 120 // added up to cost 98

// RGBA texture of a cost grid, width * height * 4, row 0 = grid row 0. Unknown (negative) cells are fully
// transparent; free (0) cells are the painted pathway, a translucent green; costs 1..98, the inflation halo, are
// translucent and stronger with the cost; 99 (inscribed) and 100 (lethal) are opaque strong colours. A value above
// 100 is not valid and is drawn as lethal rather than dropped.
//
// Free is what Nav2 says, not what the camera saw: with track_unknown_space off (mindmap D22) a cell the camera has
// never observed is free too, so it is painted as path as well (mindmap D27).
export function buildGridTexture(f: GridFrame): Uint8ClampedArray {
  const n = f.width * f.height
  const { cells } = f
  const out = new Uint8ClampedArray(4 * n)
  for (let i = 0; i < n; i++) {
    const v = cells[i]
    if (v < 0) continue
    const o = 4 * i
    if (v === 0) {
      out[o] = PATH_COLOR[0]
      out[o + 1] = PATH_COLOR[1]
      out[o + 2] = PATH_COLOR[2]
      out[o + 3] = PATH_ALPHA
    } else if (v >= 100) {
      out[o] = LETHAL_COLOR[0]
      out[o + 1] = LETHAL_COLOR[1]
      out[o + 2] = LETHAL_COLOR[2]
      out[o + 3] = 255
    } else if (v === 99) {
      out[o] = INSCRIBED_COLOR[0]
      out[o + 1] = INSCRIBED_COLOR[1]
      out[o + 2] = INSCRIBED_COLOR[2]
      out[o + 3] = 255
    } else {
      writeRamp(v / 100, out, o)
      out[o + 3] = HALO_ALPHA_MIN + (HALO_ALPHA_SPAN * v) / 98
    }
  }
  return out
}

// RGBA grey image of a depth image in metres (width * height values, row 0 first), width * height * 4. Brightness
// falls linearly from white at 0 m to black at maxRangeM and beyond (near = bright). Holes (NaN, infinite, <= 0) are
// fully transparent; if maxRangeM is not usable every valid pixel is mid grey.
export function depthToRgba(depthM: Float32Array, width: number, height: number, maxRangeM: number): Uint8ClampedArray {
  const n = Math.min(width * height, depthM.length)
  const out = new Uint8ClampedArray(4 * width * height)
  const usable = Number.isFinite(maxRangeM) && maxRangeM > 0
  for (let i = 0; i < n; i++) {
    const m = depthM[i]
    if (!(m > 0) || !Number.isFinite(m)) continue
    const grey = usable ? 255 * (1 - Math.min(1, m / maxRangeM)) : 128
    const o = 4 * i
    out[o] = grey
    out[o + 1] = grey
    out[o + 2] = grey
    out[o + 3] = 255
  }
  return out
}
