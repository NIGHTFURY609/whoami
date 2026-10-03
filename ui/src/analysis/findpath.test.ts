import { describe, expect, it } from 'vitest'
import { CELL_M, TH, TW, Z_MIN } from '../types'
import { findPath } from './groundmap'

// The previous findPath search: a full scan for the lowest-distance undone cell (ties: lowest index) per step. The heap
// version must pop cells in the same order, so both give the same path on any grid.
function referenceDistances(grid: Uint8Array): { dist: Float32Array; prev: Int32Array } {
  const start = (TW - 1) / 2
  const dist = new Float32Array(TW * TH).fill(Infinity)
  const prev = new Int32Array(TW * TH).fill(-1)
  const done = new Uint8Array(TW * TH)
  dist[start] = 0
  for (;;) {
    let cur = -1
    for (let i = 0; i < dist.length; i++) if (!done[i] && dist[i] < Infinity && (cur < 0 || dist[i] < dist[cur])) cur = i
    if (cur < 0) break
    done[cur] = 1
    const cz = Math.floor(cur / TW), cx = cur % TW
    for (let dz = -1; dz <= 1; dz++) {
      for (let dx = -1; dx <= 1; dx++) {
        const z = cz + dz, x = cx + dx
        if ((dz === 0 && dx === 0) || z < 0 || z >= TH || x < 0 || x >= TW) continue
        const ni = z * TW + x
        if (grid[ni] === 2) continue
        const step = (dz !== 0 && dx !== 0 ? 1.414 : 1) * (grid[ni] === 0 ? 1 : 6)
        if (dist[cur] + step < dist[ni]) { dist[ni] = dist[cur] + step; prev[ni] = cur }
      }
    }
  }
  return { dist, prev }
}

function referencePath(grid: Uint8Array) {
  const start = (TW - 1) / 2
  const { dist, prev } = referenceDistances(grid)
  let goal = start, best = -Infinity
  for (let i = 0; i < grid.length; i++) {
    if (grid[i] !== 0 || dist[i] === Infinity) continue
    const score = Math.floor(i / TW) - 0.08 * dist[i]
    if (score > best) { best = score; goal = i }
  }
  if (goal === start) return []
  const cells: number[] = []
  for (let i = goal; i >= 0; i = prev[i]) cells.push(i)
  return cells.reverse()
}

function rng(seed: number) {
  return () => {
    seed = (seed * 1664525 + 1013904223) >>> 0
    return seed / 2 ** 32
  }
}

describe('findPath', () => {
  it('finds the same path as the full-scan search on random grids', () => {
    const r = rng(7)
    for (let n = 0; n < 80; n++) {
      const pFree = r(), pLethal = r() * (1 - pFree)
      const grid = Uint8Array.from({ length: TW * TH }, () => {
        const x = r()
        return x < pFree ? 0 : x < pFree + pLethal ? 2 : 1
      })
      grid[(TW - 1) / 2] = 0
      // the same cells, through the same mapping and smoothing as findPath
      const pts = referencePath(grid).map((i) => ({ x: ((i % TW) - (TW - 1) / 2) * CELL_M, z: Z_MIN + Math.floor(i / TW) * CELL_M }))
      const want = pts.map((p, i) => (i === 0 || i === pts.length - 1 ? p : { x: (pts[i - 1].x + p.x + pts[i + 1].x) / 3, z: p.z }))
      expect(findPath(grid)).toEqual(want)
    }
  })
})
