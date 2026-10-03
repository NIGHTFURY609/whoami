// Turns a canonical Perception Port mask (GW x GH, classes {0,1,2}) into everything the UI shows beyond
// the overlay: a flat-ground costmap, a local path preview and the stats. Pure functions, no I/O.
// Flat-ground projection (camera height CAM_H over level ground) is an approximation, so the ground map
// and path are previews, not Nav2's plan. The live view holds the line across frames in pathhold.ts.
import {
  CAM_H, CELL_M, GH, GW, PERCEPTION_MAX_AGE_MS, TH, TW, Z_MIN,
  type Analysis, type FrameMeta,
} from '../types'

const HORIZON = Math.round(GH * 0.42)

export interface CameraGrid {
  fx: number
  fy: number
  cx: number
  vh: number // image row of the horizon / optical centre, in grid space
}

// Image position of a ground path. z is metres ahead; x is metres to the right.
export function pathPixels(path: { x: number; z: number }[], meta: FrameMeta): { u: number; v: number }[] {
  const { fx, fy, cx, vh } = cameraGrid(meta)
  return path.map(({ x, z }) => ({
    u: (cx + (fx * x) / z) / GW,
    v: (vh + (CAM_H * fy) / z) / GH,
  }))
}

export function cameraGrid(meta: FrameMeta): CameraGrid {
  const fx = (meta.K.fx * GW) / meta.width
  const fy = (meta.K.fy * GH) / meta.height
  const cx = (meta.K.cx * GW) / meta.width
  const vh = meta.kAssumed ? HORIZON : Math.round((meta.K.cy * GH) / meta.height)
  return { fx, fy, cx, vh }
}

// Flat-ground projection only holds where something touches the floor. Keep the bottom
// CONTACT_PX rows of each hazard run in a column and mark the rest SKIP, so a tall trunk
// or rock is not smeared across the ground map.
const CONTACT_PX = 2
const SKIP = 255

function contactMask(mask: Uint8Array): Uint8Array {
  const out = mask.slice()
  for (let u = 0; u < GW; u++) {
    let run = 0
    for (let v = GH - 1; v >= 0; v--) {
      const i = v * GW + u
      run = mask[i] === 2 ? run + 1 : 0
      if (run > CONTACT_PX) out[i] = SKIP
    }
  }
  return out
}

// Sample the mask into a ground grid by projecting each cell back into the image.
function groundGrid(rawMask: Uint8Array, fx: number, fy: number, cx: number, vh: number): Uint8Array {
  const mask = contactMask(rawMask)
  const grid = new Uint8Array(TW * TH)
  for (let gz = 0; gz < TH; gz++) {
    const z = Z_MIN + gz * CELL_M
    const v = vh + (CAM_H * fy) / z
    const dv = Math.abs((CAM_H * fy) / z - (CAM_H * fy) / (z + CELL_M))
    const wv = Math.max(1, Math.round(dv / 2))
    const wu = Math.max(1, Math.round((fx * CELL_M) / z / 2))
    for (let gx = 0; gx < TW; gx++) {
      const x = (gx - (TW - 1) / 2) * CELL_M
      const u = cx + (fx * x) / z
      let trav = 0, haz = 0, total = 0
      for (let vv = Math.round(v - wv); vv <= Math.round(v + wv); vv++) {
        for (let uu = Math.round(u - wu); uu <= Math.round(u + wu); uu++) {
          if (uu < 0 || uu >= GW || vv <= vh || vv >= GH) continue
          const c = mask[vv * GW + uu]
          if (c === SKIP) continue
          if (c === 1) trav++
          else if (c === 2) haz++
          total++
        }
      }
      // unseen or unsure cells stay 1 (inflated): unknown is never free
      grid[gz * TW + gx] = total === 0 ? 1 : haz / total > 0.15 ? 2 : trav / total > 0.5 ? 0 : 1
    }
  }

  const inflated = grid.slice()
  for (let gz = 0; gz < TH; gz++) {
    for (let gx = 0; gx < TW; gx++) {
      if (grid[gz * TW + gx] !== 2) continue
      for (let dz = -2; dz <= 2; dz++) {
        for (let dx = -2; dx <= 2; dx++) {
          const z = gz + dz, x = gx + dx
          if (z < 0 || z >= TH || x < 0 || x >= TW) continue
          const near = Math.max(Math.abs(dz), Math.abs(dx)) <= 1
          const i = z * TW + x
          inflated[i] = Math.max(inflated[i], near ? 2 : 1)
        }
      }
    }
  }
  return inflated
}

// Min-heap of (distance, cell) for findPath. Entries are never updated in place: a cell pushed again with a lower
// distance leaves a stale entry behind, skipped on pop (already done, or no longer its current distance).
class CellHeap {
  private d: number[] = []
  private c: number[] = []

  private less(a: number, b: number): boolean {
    return this.d[a] < this.d[b] || (this.d[a] === this.d[b] && this.c[a] < this.c[b])
  }

  private swap(a: number, b: number): void {
    ;[this.d[a], this.d[b]] = [this.d[b], this.d[a]]
    ;[this.c[a], this.c[b]] = [this.c[b], this.c[a]]
  }

  push(dist: number, cell: number): void {
    this.d.push(dist)
    this.c.push(cell)
    for (let i = this.d.length - 1; i > 0;) {
      const p = (i - 1) >> 1
      if (!this.less(i, p)) break
      this.swap(i, p)
      i = p
    }
  }

  // The undone cell with the lowest current distance (ties: lowest index), or -1.
  pop(done: Uint8Array, dist: Float32Array): number {
    while (this.d.length > 0) {
      const dist0 = this.d[0]
      const cell = this.c[0]
      const lastD = this.d.pop()!
      const lastC = this.c.pop()!
      if (this.d.length > 0) {
        this.d[0] = lastD
        this.c[0] = lastC
        for (let i = 0; ;) {
          const l = 2 * i + 1
          const r = l + 1
          let m = i
          if (l < this.d.length && this.less(l, m)) m = l
          if (r < this.d.length && this.less(r, m)) m = r
          if (m === i) break
          this.swap(i, m)
          i = m
        }
      }
      if (!done[cell] && dist0 === dist[cell]) return cell
    }
    return -1
  }
}

export function findPath(grid: Uint8Array): { x: number; z: number }[] {
  const start = (TW - 1) / 2
  const dist = new Float32Array(TW * TH).fill(Infinity)
  const prev = new Int32Array(TW * TH).fill(-1)
  const done = new Uint8Array(TW * TH)
  dist[start] = 0
  // Dijkstra on a binary heap keyed (distance, cell index): it pops cells in exactly the order a scan for the lowest
  // distance (ties: lowest index) would, so the path is the same, at O(n log n) instead of O(n^2) per frame.
  const heap = new CellHeap()
  heap.push(0, start)

  for (;;) {
    const cur = heap.pop(done, dist)
    if (cur < 0) break
    done[cur] = 1
    const cz = Math.floor(cur / TW), cxg = cur % TW
    for (let dz = -1; dz <= 1; dz++) {
      for (let dx = -1; dx <= 1; dx++) {
        const z = cz + dz, x = cxg + dx
        if ((dz === 0 && dx === 0) || z < 0 || z >= TH || x < 0 || x >= TW) continue
        const ni = z * TW + x
        if (grid[ni] === 2) continue
        const step = (dz !== 0 && dx !== 0 ? 1.414 : 1) * (grid[ni] === 0 ? 1 : 6)
        if (dist[cur] + step < dist[ni]) {
          dist[ni] = dist[cur] + step
          prev[ni] = cur
          heap.push(dist[ni], ni) // the stored (float32) distance, the one the scan compared
        }
      }
    }
  }

  let goal = start, best = -Infinity
  for (let i = 0; i < grid.length; i++) {
    if (grid[i] !== 0 || dist[i] === Infinity) continue
    const score = Math.floor(i / TW) - 0.08 * dist[i]
    if (score > best) { best = score; goal = i }
  }
  if (goal === start) return []

  const cells: number[] = []
  for (let i = goal; i >= 0; i = prev[i]) cells.push(i)
  cells.reverse()
  const pts = cells.map((i) => ({ x: ((i % TW) - (TW - 1) / 2) * CELL_M, z: Z_MIN + Math.floor(i / TW) * CELL_M }))
  return pts.map((p, i) => {
    if (i === 0 || i === pts.length - 1) return p
    return { x: (pts[i - 1].x + p.x + pts[i + 1].x) / 3, z: p.z }
  })
}

export interface MaskInput {
  meta: FrameMeta
  mask: Uint8Array // GW*GH, strictly {0,1,2}
  depth: Float32Array | null // GW*GH metres, or null when there is no depth channel
  latencyMs: number
  reasons?: string[] // anything already wrong with the sample (stale, degraded, invalid)
}

export function buildAnalysis({ meta, mask, depth, latencyMs, reasons: given = [] }: MaskInput): Analysis {
  const { fx, fy, cx, vh } = cameraGrid(meta)
  const grid = groundGrid(mask, fx, fy, cx, vh)
  const path = findPath(grid)
  const pathPx = pathPixels(path, meta)

  const counts = [0, 0, 0]
  for (const c of mask) counts[c]++
  const classPct = counts.map((n) => (100 * n) / mask.length) as [number, number, number]

  let depthStats: Analysis['depthStats'] = null
  if (depth) {
    const ground = Array.from(depth.subarray(vh + 1 < GH ? (vh + 1) * GW : 0)).filter(Number.isFinite).sort((a, b) => a - b)
    if (ground.length) {
      depthStats = { min: ground[0], median: ground[Math.floor(ground.length / 2)], max: ground[ground.length - 1] }
    }
  }

  const ageMs = Math.max(0, Date.now() - (meta.receivedAt ?? meta.stamp))
  const reasons = [...given]
  if (ageMs > PERCEPTION_MAX_AGE_MS && meta.streaming && !reasons.includes('MASK STALE')) reasons.push('MASK STALE')
  if (classPct[1] < 3) reasons.push('NO TRAVERSABLE GROUND')

  return {
    meta, mask, depth, grid, path, pathPx, classPct, depthStats, ageMs, latencyMs,
    degraded: reasons.length > 0,
    reasons,
  }
}
