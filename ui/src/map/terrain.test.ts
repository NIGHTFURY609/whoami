import { describe, expect, it } from 'vitest'
import { EMPTY_Z, TERRAIN_CELL_M, TERRAIN_SIZE, Terrain } from './terrain'

const pts = (...p: [number, number, number][]) => Float32Array.from(p.flat())
// a small window: 10 x 10 cells of 1 m, re-centred within 2 m of an edge
const small = () => new Terrain(1, 10, 2)
const at = (t: Terrain, x: number, y: number) =>
  t.z[Math.floor((y - t.originY) / t.cellM) * t.size + Math.floor((x - t.originX) / t.cellM)]

describe('Terrain', () => {
  it('defaults to a 30 m window of 5 cm cells, all empty', () => {
    const t = new Terrain()
    expect([TERRAIN_CELL_M, TERRAIN_SIZE]).toEqual([0.05, 600])
    expect(t.z.length).toBe(600 * 600)
    expect(t.z.every((v) => v === EMPTY_Z)).toBe(true)
    expect(t.filled).toBe(0)
  })

  it('refuses a window it cannot keep the robot inside', () => {
    expect(() => new Terrain(1, 10, 5)).toThrow(RangeError)
    expect(() => new Terrain(0, 10, 1)).toThrow(RangeError)
  })

  it('centres the window on the robot, on whole cells', () => {
    const t = small()
    expect(t.follow(0.4, -3.6)).toBe(true)
    expect([t.originX, t.originY]).toEqual([-5, -9])
    expect(t.follow(1, -3)).toBe(false) // well inside: nothing moves
    expect(t.follow(Number.NaN, 0)).toBe(false)
  })

  it('keeps the highest point of a scan per cell and lets a later scan replace it', () => {
    const t = small()
    t.follow(0, 0)
    t.ingest(pts([0.2, 0.2, 0.1], [0.7, 0.9, 0.6], [0.5, 0.5, 0.3], [2.5, 0.5, -0.1]))
    expect(at(t, 0.5, 0.5)).toBeCloseTo(0.6)
    expect(at(t, 2.5, 0.5)).toBeCloseTo(-0.1)
    expect(t.filled).toBe(2)
    t.ingest(pts([0.5, 0.5, 0.05])) // the obstacle moved away: the new, lower scan wins
    expect(at(t, 0.5, 0.5)).toBeCloseTo(0.05)
    expect(t.filled).toBe(2)
  })

  it('reports the range of cells it wrote, and null when nothing landed', () => {
    const t = small()
    t.follow(0, 0)
    const r = t.ingest(pts([0.5, 0.5, 0], [-2.5, -3.5, 0]))!
    const idx = (x: number, y: number) => (y - t.originY) * t.size + (x - t.originX)
    expect(r).toEqual({ start: idx(-3, -4), end: idx(0, 0) + 1 })
    expect(t.ingest(pts([50, 50, 0], [Number.NaN, 0, 0], [0, 0, Number.POSITIVE_INFINITY]))).toBeNull()
  })

  it('places itself on the first point when no robot position came first', () => {
    const t = small()
    expect(t.ingest(pts([Number.NaN, 0, 0], [12.3, 4.5, 1]))).not.toBeNull()
    expect([t.originX, t.originY]).toEqual([7, -1])
    expect(at(t, 12.3, 4.5)).toBe(1)
  })

  it('shifts by whole cells when the robot nears an edge, keeping what is still inside', () => {
    const t = small()
    t.follow(0, 0) // window x, y in [-5, 5)
    t.ingest(pts([1.5, 0.5, 7], [-4.5, 0.5, 9]))
    expect(t.follow(3.5, 0)).toBe(true) // 1.5 m from the right edge
    expect([t.originX, t.originY]).toEqual([-2, -5])
    expect(at(t, 1.5, 0.5)).toBe(7) // still inside, moved with the window
    expect(t.filled).toBe(1) // the point at x = -4.5 left the window
    expect(Array.from(t.z).filter((v) => v !== EMPTY_Z)).toEqual([7])
    expect(t.follow(60, 60)).toBe(true) // a jump past the whole window empties it
    expect(t.filled).toBe(0)
    expect(t.z.every((v) => v === EMPTY_Z)).toBe(true)
  })

  it('clear() forgets everything and is placed again by the next position', () => {
    const t = small()
    t.follow(0, 0)
    t.ingest(pts([0.5, 0.5, 1]))
    t.clear()
    expect(t.filled).toBe(0)
    expect(t.z.every((v) => v === EMPTY_Z)).toBe(true)
    expect(t.follow(20, 20)).toBe(true)
    expect([t.originX, t.originY]).toEqual([15, 15])
  })

  it('merges a 300 000 point scan into the full-size window', () => {
    const t = new Terrain()
    t.follow(0, 0)
    const n = 300_000
    const xyz = new Float32Array(3 * n)
    for (let i = 0; i < n; i++) {
      xyz[3 * i] = ((i % 600) - 300) * 0.04
      xyz[3 * i + 1] = (Math.floor(i / 600) - 250) * 0.04
      xyz[3 * i + 2] = 0.01 * (i % 7)
    }
    const r = t.ingest(xyz)
    expect(r).not.toBeNull()
    expect(t.filled).toBeGreaterThan(100_000)
  })
})
