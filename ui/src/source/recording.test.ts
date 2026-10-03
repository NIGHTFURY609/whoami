import { afterEach, describe, expect, it, vi } from 'vitest'
import { fetchRecordedFrame } from './recording'
import { parseVideoStatus } from './videobridge'

const camera = { width: 640, height: 480, k: [500, 0, 320, 0, 500, 240, 0, 0, 1], frameId: 'camera_optical_frame' }

const status = {
  state: 'paused', frame: 10, frames: 486, fps: 30, speed: 0.9, sync: true, loop: false, perception: true, note: '',
  cache: { frames: 12, masks: 11, depths: 10, total: 486, complete: false,
    camera: { width: 640, height: 480, k: camera.k, d: [], distortion_model: 'plumb_bob', frame_id: 'camera_optical_frame' } },
}

describe('parseVideoStatus', () => {
  it('reads the recording summary and its camera', () => {
    const s = parseVideoStatus(status)
    expect(s?.cache).toEqual({ frames: 12, masks: 11, depths: 10, total: 486, complete: false, camera })
  })

  it('has no cache for a bridge that does not record, and refuses a fake K', () => {
    expect(parseVideoStatus({ ...status, cache: undefined })?.cache).toBeNull()
    const zeroK = { ...status, cache: { ...status.cache, camera: { ...status.cache.camera, k: Array(9).fill(0) } } }
    expect(parseVideoStatus(zeroK)?.cache?.camera).toBeNull()
  })
})

function bundle(jpeg: Uint8Array, mask: Uint8Array | null, mw: number, mh: number, depth: Float32Array | null, dw: number, dh: number) {
  const m = mask ?? new Uint8Array(0)
  const d = new Uint8Array(depth ? depth.buffer : new ArrayBuffer(0))
  const out = new Uint8Array(20 + jpeg.length + m.length + d.length)
  const v = new DataView(out.buffer)
  ;[jpeg.length, mask ? mw : 0, mask ? mh : 0, depth ? dw : 0, depth ? dh : 0].forEach((x, i) => v.setUint32(i * 4, x, true))
  out.set(jpeg, 20)
  out.set(m, 20 + jpeg.length)
  out.set(d, 20 + jpeg.length + m.length)
  return new Response(out)
}

const jpeg = new Uint8Array([0xff, 0xd8, 0xff, 0xd9])

describe('fetchRecordedFrame', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('returns the frame with its mask, depth and the camera K scaled to the frame', async () => {
    const fetchMock = vi.fn(async (_url: string) => bundle(jpeg, new Uint8Array([0, 1, 2, 1]), 2, 2, new Float32Array([1.5, NaN, 3, 4]), 2, 2))
    vi.stubGlobal('fetch', fetchMock)
    vi.stubGlobal('createImageBitmap', vi.fn(async () => ({ width: 320, height: 240, close: () => {} })))
    const f = await fetchRecordedFrame('http://bridge', 7, camera)
    expect(String(fetchMock.mock.calls[0][0])).toBe('http://bridge/video/cache/bundle/7?w=160&h=120')
    expect(f?.index).toBe(7)
    expect(f?.K).toEqual({ fx: 250, fy: 250, cx: 160, cy: 120 })
    expect(Array.from(f!.mask!.data)).toEqual([0, 1, 2, 1])
    expect(f!.depth!.data[0]).toBeCloseTo(1.5)
    expect(Number.isNaN(f!.depth!.data[1])).toBe(true)
    expect(f!.mask!.frameId).toBe('camera_optical_frame')
  })

  it('shows a frame without an overlay when none was saved, and nothing for a frame that was not recorded', async () => {
    vi.stubGlobal('createImageBitmap', vi.fn(async () => ({ width: 640, height: 480, close: () => {} })))
    vi.stubGlobal('fetch', vi.fn(async () => bundle(jpeg, null, 0, 0, null, 0, 0)))
    const f = await fetchRecordedFrame('http://bridge', 1, camera)
    expect(f?.mask).toBeNull()
    expect(f?.depth).toBeNull()
    vi.stubGlobal('fetch', vi.fn(async () => new Response('{}', { status: 404 })))
    expect(await fetchRecordedFrame('http://bridge', 2, camera)).toBeNull()
  })

  it('refuses a truncated bundle', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(new Uint8Array(30))))
    expect(await fetchRecordedFrame('http://bridge', 3, camera)).toBeNull()
  })
})
