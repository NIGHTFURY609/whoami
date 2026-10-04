import { describe, expect, it } from 'vitest'
import type { GridFrame } from './codec'
import { buildGridTexture, depthToRgba, ribbonStrip, writeRamp } from './geometry'

const NaN_ = Number.NaN

function rampAt(t: number): number[] {
  const out = new Uint8Array(3)
  writeRamp(t, out, 0)
  return Array.from(out)
}

const luminance = ([r, g, b]: number[]) => 0.2126 * r + 0.7152 * g + 0.0722 * b

describe('colour ramp', () => {
  it('ends are distinct and the middle sits between them', () => {
    const lo = rampAt(0)
    const hi = rampAt(1)
    expect(lo).not.toEqual(hi)
    expect(rampAt(0.5)).not.toEqual(lo)
    expect(rampAt(0.5)).not.toEqual(hi)
  })

  it('is perceptually ordered: luminance never decreases from low to high', () => {
    let prev = -1
    for (let i = 0; i <= 100; i++) {
      const y = luminance(rampAt(i / 100))
      expect(y).toBeGreaterThanOrEqual(prev - 0.5) // rounding to bytes may wobble by well under 1
      prev = y
    }
    expect(luminance(rampAt(1))).toBeGreaterThan(luminance(rampAt(0)) + 100)
  })

  it('clamps outside 0..1 and sends non-finite input to the low colour', () => {
    expect(rampAt(-5)).toEqual(rampAt(0))
    expect(rampAt(7)).toEqual(rampAt(1))
    expect(rampAt(NaN_)).toEqual(rampAt(0))
    expect(rampAt(Infinity)).toEqual(rampAt(1))
  })

  it('writes three bytes at the offset and nothing else', () => {
    const out = new Uint8Array(9).fill(9)
    writeRamp(1, out, 3)
    expect(Array.from(out.subarray(0, 3))).toEqual([9, 9, 9])
    expect(Array.from(out.subarray(3, 6))).toEqual(rampAt(1))
    expect(Array.from(out.subarray(6))).toEqual([9, 9, 9])
  })
})

describe('ribbonStrip', () => {
  const path = (...p: [number, number, number][]) => Float32Array.from(p.flat())
  const vertex = (r: { positions: Float32Array }, i: number) => Array.from(r.positions.subarray(3 * i, 3 * i + 3))

  it('puts a left and a right vertex half the width either side of each point, lifted', () => {
    const r = ribbonStrip(path([0, 0, 0], [2, 0, 0]), 0.2, 0.01)!
    expect(r.positions.length).toBe(12)
    const close = (v: number[], w: number[]) => v.forEach((x, i) => expect(x).toBeCloseTo(w[i], 6))
    close(vertex(r, 0), [0, 0.1, 0.01]) // left of +x is +y
    close(vertex(r, 1), [0, -0.1, 0.01])
    close(vertex(r, 2), [2, 0.1, 0.01])
    close(vertex(r, 3), [2, -0.1, 0.01])
    expect(Array.from(r.index)).toEqual([0, 1, 2, 1, 3, 2])
    expect(r.lengthM).toBeCloseTo(2, 6)
  })

  it('turns with the path, using the direction through each point', () => {
    const r = ribbonStrip(path([0, 0, 0], [1, 0, 0], [1, 1, 0]), 2)!
    const [x, y] = vertex(r, 2) // the corner: direction (1, 1) / sqrt 2, left normal (-1, 1) / sqrt 2
    expect(x).toBeCloseTo(1 - Math.SQRT1_2, 6)
    expect(y).toBeCloseTo(Math.SQRT1_2, 6)
    expect(r.index.length).toBe(12)
    expect(r.lengthM).toBeCloseTo(2, 6)
  })

  it('skips repeated and non-finite points, and is null below two points', () => {
    const r = ribbonStrip(path([0, 0, 0], [0, 0, 0], [0.001, 0, 0], [Number.NaN, 0, 0], [1, 0, 0]), 0.2)!
    expect(r.positions.length).toBe(12) // two points kept
    expect(ribbonStrip(path([0, 0, 0], [0, 0.001, 0]), 0.2)).toBeNull()
    expect(ribbonStrip(new Float32Array(0), 0.2)).toBeNull()
  })

  it('keeps a finite width on a path that doubles straight back', () => {
    const r = ribbonStrip(path([0, 0, 0], [1, 0, 0], [0, 0, 0]), 0.2)!
    expect(r.positions.every(Number.isFinite)).toBe(true)
    const [, y] = vertex(r, 2)
    expect(Math.abs(y)).toBeCloseTo(0.1, 6)
  })
})

function grid(width: number, height: number, cells: number[]): GridFrame {
  return { epoch: 1, seq: 1, stampS: 0, width, height, resolution: 0.05, originX: 0, originY: 0, originYaw: 0, cells: Int8Array.from(cells) }
}

const texel = (t: Uint8ClampedArray, i: number) => Array.from(t.subarray(4 * i, 4 * i + 4))

describe('buildGridTexture', () => {
  it('is RGBA, width * height * 4, as a Uint8ClampedArray', () => {
    const t = buildGridTexture(grid(3, 2, [0, 0, 0, 0, 0, 0]))
    expect(t).toBeInstanceOf(Uint8ClampedArray)
    expect(t.length).toBe(3 * 2 * 4)
  })

  it('makes unknown (-1) cells fully transparent and paints free (0) cells as a translucent green path', () => {
    const t = buildGridTexture(grid(2, 1, [-1, 0]))
    expect(texel(t, 0)[3]).toBe(0)
    const [r, g, b, a] = texel(t, 1)
    expect(g).toBeGreaterThan(r)
    expect(g).toBeGreaterThan(b)
    expect(a).toBeGreaterThan(0)
    expect(a).toBeLessThan(255)
  })

  it('treats any other negative value as unknown', () => {
    const t = buildGridTexture(grid(2, 1, [-2, -128]))
    expect(texel(t, 0)[3]).toBe(0)
    expect(texel(t, 1)[3]).toBe(0)
  })

  it('draws the inflation halo (1..98) translucent, getting stronger with the cost', () => {
    const t = buildGridTexture(grid(4, 1, [1, 30, 70, 98]))
    const alphas = [0, 1, 2, 3].map((i) => texel(t, i)[3])
    for (const a of alphas) {
      expect(a).toBeGreaterThan(0)
      expect(a).toBeLessThan(255)
    }
    expect(alphas[1]).toBeGreaterThan(alphas[0])
    expect(alphas[2]).toBeGreaterThan(alphas[1])
    expect(alphas[3]).toBeGreaterThan(alphas[2])
  })

  it('draws 99 (inscribed) and 100 (lethal) opaque in two different strong colours', () => {
    const t = buildGridTexture(grid(4, 1, [98, 99, 100, 50]))
    expect(texel(t, 1)[3]).toBe(255)
    expect(texel(t, 2)[3]).toBe(255)
    expect(texel(t, 1).slice(0, 3)).not.toEqual(texel(t, 2).slice(0, 3))
    expect(texel(t, 0)[3]).toBeLessThan(255)
    // lethal reads as the hottest colour (red channel above the others)
    const [r, g, b] = texel(t, 2)
    expect(r).toBeGreaterThan(g)
    expect(r).toBeGreaterThan(b)
  })

  it('shares the ramp with the height colours for the halo', () => {
    const t = buildGridTexture(grid(1, 1, [50]))
    expect(texel(t, 0).slice(0, 3)).toEqual(rampAt(0.5))
  })

  it('treats a value above 100 as lethal rather than dropping it', () => {
    const t = buildGridTexture(grid(1, 1, [127]))
    expect(texel(t, 0)).toEqual(texel(buildGridTexture(grid(1, 1, [100])), 0))
  })

  it('keeps texture row 0 = grid row 0, row-major (a cell at row 1, col 2 of a 3 x 2 grid is texel 5)', () => {
    const cells = [-1, -1, -1, -1, -1, 100]
    const t = buildGridTexture(grid(3, 2, cells))
    for (let i = 0; i < 5; i++) expect(texel(t, i)[3]).toBe(0)
    expect(texel(t, 5)[3]).toBe(255)
    const top = buildGridTexture(grid(3, 2, [100, -1, -1, -1, -1, -1]))
    expect(texel(top, 0)[3]).toBe(255)
  })

  it('returns an empty array for an empty grid', () => {
    expect(buildGridTexture(grid(0, 0, [])).length).toBe(0)
  })

  it('handles a 512 x 512 grid', () => {
    const cells = new Array<number>(512 * 512).fill(-1)
    cells[512 * 512 - 1] = 100
    const t = buildGridTexture(grid(512, 512, cells))
    expect(t.length).toBe(512 * 512 * 4)
    expect(t[t.length - 1]).toBe(255)
  })
})

const metres = (...m: number[]) => Float32Array.from(m)

describe('depthToRgba', () => {
  it('is RGBA, width * height * 4, as a Uint8ClampedArray', () => {
    const t = depthToRgba(metres(1, 2, 3, 4, 5, 6), 3, 2, 8)
    expect(t).toBeInstanceOf(Uint8ClampedArray)
    expect(t.length).toBe(24)
  })

  it('makes holes (NaN, infinite, zero or negative) fully transparent', () => {
    const t = depthToRgba(metres(NaN_, Infinity, 0, -1, 1), 5, 1, 8)
    for (let i = 0; i < 4; i++) expect(texel(t, i)).toEqual([0, 0, 0, 0])
    expect(texel(t, 4)[3]).toBe(255)
  })

  it('is a grey ramp over 0..maxRangeM, near bright and far dark', () => {
    const t = depthToRgba(metres(0.5, 2, 4, 8), 4, 1, 8)
    const greys = [0, 1, 2, 3].map((i) => texel(t, i))
    for (const [r, g, b, a] of greys) {
      expect(r).toBe(g)
      expect(g).toBe(b)
      expect(a).toBe(255)
    }
    expect(greys[0][0]).toBeGreaterThan(greys[1][0])
    expect(greys[1][0]).toBeGreaterThan(greys[2][0])
    expect(greys[2][0]).toBeGreaterThan(greys[3][0])
    expect(greys[3][0]).toBe(0) // at max range
    expect(greys[2][0]).toBeCloseTo(128, -1) // half range, about mid grey
  })

  it('clamps beyond the maximum range to the far colour and keeps the pixel visible', () => {
    expect(texel(depthToRgba(metres(65.5), 1, 1, 8), 0)).toEqual([0, 0, 0, 255])
  })

  it('stays row-major: image row 0 first', () => {
    const t = depthToRgba(metres(NaN_, NaN_, NaN_, 1), 2, 2, 8)
    expect(texel(t, 3)[3]).toBe(255)
    for (let i = 0; i < 3; i++) expect(texel(t, i)[3]).toBe(0)
  })

  it('does not throw on a degenerate range and still marks holes transparent', () => {
    const t = depthToRgba(metres(NaN_, 1), 2, 1, 0)
    expect(texel(t, 0)[3]).toBe(0)
    expect(texel(t, 1)).toEqual([128, 128, 128, 255])
  })

  it('leaves pixels the data does not reach transparent and handles 640 x 480', () => {
    expect(depthToRgba(metres(), 0, 0, 8).length).toBe(0)
    expect(texel(depthToRgba(metres(1), 2, 1, 8), 1)).toEqual([0, 0, 0, 0])
    expect(depthToRgba(new Float32Array(640 * 480).fill(2.5), 640, 480, 8).length).toBe(640 * 480 * 4)
  })
})
