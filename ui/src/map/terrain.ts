// The live terrain (map view, live mode `terrain`): every live depth scan merged into a rolling 2.5D height grid
// around the robot, so ground the camera has already seen stays on screen after it has left the view. Pure and
// three.js-free, so it is tested in node; scene.ts uploads `z` as it is.
//
// Layout. A square window of `size` x `size` cells of `cellM` metres, row-major (row = y, column = x), whose cell
// (0, 0) has its outer corner at (originX, originY) in the map frame. One float per cell: the height, or EMPTY_Z
// for a cell no scan has reached. The cell's x and y are not stored: the scene's vertex shader derives them from the
// vertex index and the origin, so a window shift moves heights, not coordinates, and the GPU buffer holds one float
// per cell (600 x 600 cells = 1.4 MB).
//
// Merging. Within one scan a cell takes the highest point that falls in it (an obstacle's top wins over the ground
// at its foot); a later scan that reaches the cell replaces it (a moved obstacle is not kept forever).
//
// Window. It is centred on the robot the first time it is given a position; when the robot comes within
// `marginM` of an edge, it is re-centred on it by a whole number of cells, keeping every cell that is still inside.
// Points outside the window are dropped.

export const TERRAIN_CELL_M = 0.05
export const TERRAIN_SIZE = 600 // cells per side: 30 m x 30 m at 5 cm
export const TERRAIN_MARGIN_M = 5
export const EMPTY_Z = -1e6 // below anything real; the shader hides cells under EMPTY_Z / 2

// Which part of `z` changed since the last upload: cells [start, end). null = nothing.
export interface DirtyRange { start: number; end: number }

export class Terrain {
  readonly cellM: number
  readonly size: number
  readonly marginM: number
  readonly z: Float32Array
  originX = 0
  originY = 0
  filled = 0 // cells that hold a height
  private placed = false
  private scan = 0
  private scanOf: Uint32Array // the scan that last wrote each cell, for the max-within-a-scan rule

  constructor(cellM = TERRAIN_CELL_M, size = TERRAIN_SIZE, marginM = TERRAIN_MARGIN_M) {
    if (!(cellM > 0) || !Number.isInteger(size) || size < 1 || !(marginM >= 0) || 2 * marginM >= size * cellM) {
      throw new RangeError('terrain: cellM > 0, an integer size >= 1 and a margin below half the window are required')
    }
    this.cellM = cellM
    this.size = size
    this.marginM = marginM
    this.z = new Float32Array(size * size).fill(EMPTY_Z)
    this.scanOf = new Uint32Array(size * size)
  }

  // Keeps the robot at least marginM inside the window. Returns true when the window moved (every cell changed
  // place, so the whole buffer and the origin need uploading). A non-finite position changes nothing.
  follow(x: number, y: number): boolean {
    if (!Number.isFinite(x) || !Number.isFinite(y)) return false
    const span = this.size * this.cellM
    if (this.placed) {
      const inside = x - this.originX >= this.marginM && this.originX + span - x >= this.marginM
        && y - this.originY >= this.marginM && this.originY + span - y >= this.marginM
      if (inside) return false
    }
    const half = Math.floor(this.size / 2)
    const ox = (Math.floor(x / this.cellM) - half) * this.cellM
    const oy = (Math.floor(y / this.cellM) - half) * this.cellM
    if (this.placed) this.shift(Math.round((ox - this.originX) / this.cellM), Math.round((oy - this.originY) / this.cellM))
    this.originX = ox
    this.originY = oy
    this.placed = true
    return true
  }

  // Merges one scan (x y z per point, map frame). Before the window is placed (no robot position yet) it is placed
  // on the first finite point. Returns the range of cells written, or null when nothing landed in the window.
  ingest(xyz: Float32Array, count = Math.floor(xyz.length / 3)): DirtyRange | null {
    const n = Math.min(count, Math.floor(xyz.length / 3))
    if (!this.placed) {
      for (let i = 0; i < n && !this.placed; i++) this.follow(xyz[3 * i], xyz[3 * i + 1])
      if (!this.placed) return null
    }
    this.scan = this.scan === 0xffffffff ? 1 : this.scan + 1
    const { size, cellM, originX, originY, z, scanOf, scan } = this
    let start = Infinity
    let end = -1
    for (let i = 0; i < n; i++) {
      const pz = xyz[3 * i + 2]
      if (!Number.isFinite(pz)) continue
      const col = Math.floor((xyz[3 * i] - originX) / cellM)
      const row = Math.floor((xyz[3 * i + 1] - originY) / cellM)
      if (!(col >= 0 && col < size && row >= 0 && row < size)) continue // also drops a NaN x or y
      const c = row * size + col
      if (scanOf[c] === scan) {
        if (pz > z[c]) z[c] = pz
        continue
      }
      if (z[c] === EMPTY_Z) this.filled++
      scanOf[c] = scan
      z[c] = pz
      if (c < start) start = c
      if (c > end) end = c
    }
    return end < 0 ? null : { start, end: end + 1 }
  }

  // Forgets everything (the window is placed again on the next position or scan).
  clear(): void {
    this.z.fill(EMPTY_Z)
    this.scanOf.fill(0)
    this.filled = 0
    this.placed = false
  }

  // Moves the content by (dc, dr) cells: the cell that was at (col, row) is at (col - dc, row - dr) afterwards.
  private shift(dc: number, dr: number): void {
    const { size, z, scanOf } = this
    if (Math.abs(dc) >= size || Math.abs(dr) >= size) {
      z.fill(EMPTY_Z)
      scanOf.fill(0)
      this.filled = 0
      return
    }
    const nz = new Float32Array(size * size).fill(EMPTY_Z)
    const ns = new Uint32Array(size * size)
    const c0 = Math.max(0, dc)
    const c1 = Math.min(size, size + dc) // source columns that stay inside
    let filled = 0
    for (let r = Math.max(0, dr); r < Math.min(size, size + dr); r++) {
      const src = r * size
      const dst = (r - dr) * size - dc
      nz.set(z.subarray(src + c0, src + c1), dst + c0)
      ns.set(scanOf.subarray(src + c0, src + c1), dst + c0)
      for (let c = c0; c < c1; c++) if (z[src + c] !== EMPTY_Z) filled++
    }
    z.set(nz)
    scanOf.set(ns)
    this.filled = filled
  }
}
