import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ApiError, LAYERS, type Layer, type MapStatus } from '../source/api'
import {
  MIN_INTERVAL_MS, STATUS_POLL_LIVE_MS, STATUS_POLL_MS, STATUS_STALE_MS, createMapSession, isStale, layerKey, nextFetches,
  schedule, statusPollMs, type Frame,
} from './useMapData'

const none = <T>(v: T): Record<Layer, T> => Object.fromEntries(LAYERS.map((l) => [l, v])) as Record<Layer, T>
const only = (...on: Layer[]): Record<Layer, boolean> => ({ ...none(false), ...Object.fromEntries(on.map((l) => [l, true])) })
const ALL = none(true)

const mapStatus = (epoch: number, seq: Partial<Record<Layer, number>> = {}): MapStatus => ({ epoch, seq: { ...none(0), ...seq }, stats: {} })

describe('layerKey', () => {
  it('is the epoch and the layer sequence', () => {
    expect(layerKey(mapStatus(7, { live: 3 }), 'live')).toBe('7:3')
    expect(layerKey(mapStatus(7, { live: 3 }), 'grid')).toBe('7:0')
  })
})

describe('nextFetches', () => {
  it('fetches a layer whose epoch:seq key differs from the one held, and only that one', () => {
    const status = mapStatus(5, { grid: 3, trajectory: 2 })
    const have = { ...none<string | null>(null), grid: '5:3', trajectory: '5:1' }
    expect(nextFetches(status, have, ALL)).toEqual(['trajectory'])
  })

  it('fetches nothing when every held key matches', () => {
    const status = mapStatus(5, { live: 3, grid: 9 })
    const have = { ...none<string | null>(null), live: '5:3', grid: '5:9' }
    expect(nextFetches(status, have, ALL)).toEqual([])
  })

  it('fetches a layer that has never been held', () => {
    expect(nextFetches(mapStatus(5, { grid: 1 }), none<string | null>(null), ALL)).toEqual(['grid'])
  })

  it('refetches every enabled layer when the epoch changes even though no seq did', () => {
    const before = mapStatus(5, { trajectory: 4, grid: 1, live: 8 })
    const have = Object.fromEntries(LAYERS.map((l) => [l, layerKey(before, l)])) as Record<Layer, string | null>
    expect(nextFetches(before, have, ALL)).toEqual([])
    const restarted = { ...before, epoch: 6 }
    expect(nextFetches(restarted, have, ALL)).toEqual([...LAYERS])
    expect(nextFetches(restarted, have, only('trajectory', 'live'))).toEqual(['trajectory', 'live'])
  })

  it('never fetches a layer whose seq is 0, whatever is held', () => {
    const status = mapStatus(5, { live: 0, grid: 2 })
    expect(nextFetches(status, none<string | null>(null), ALL)).toEqual(['grid'])
    expect(nextFetches(status, { ...none<string | null>(null), live: '4:9' }, ALL)).toEqual(['grid'])
  })

  it('never fetches a disabled layer', () => {
    const status = mapStatus(5, { grid: 1, trajectory: 1, live: 1 })
    expect(nextFetches(status, none<string | null>(null), only('trajectory'))).toEqual(['trajectory'])
    expect(nextFetches(status, none<string | null>(null), none(false))).toEqual([])
  })

  it('returns layers in LAYERS order', () => {
    const status = mapStatus(1, { live: 1, grid: 1, trajectory: 1 })
    expect(nextFetches(status, none<string | null>(null), ALL)).toEqual(['trajectory', 'grid', 'live'])
  })

  it('does not treat a held key from another epoch with the same seq as current', () => {
    const status = mapStatus(2, { live: 3 })
    expect(nextFetches(status, { ...none<string | null>(null), live: '1:3' }, ALL)).toEqual(['live'])
  })
})

describe('MIN_INTERVAL_MS', () => {
  it('is the specified minimum between fetch starts of one layer', () => {
    expect(MIN_INTERVAL_MS).toEqual({ trajectory: 1000, grid: 1000, live: 200 })
  })
})

describe('statusPollMs', () => {
  it('is a quarter second while the live scan is wanted and a second otherwise', () => {
    expect([STATUS_POLL_LIVE_MS, STATUS_POLL_MS]).toEqual([250, 1000])
    expect(statusPollMs(only('live'))).toBe(250)
    expect(statusPollMs(ALL)).toBe(250)
    expect(statusPollMs(only('trajectory', 'grid'))).toBe(1000)
  })
})

describe('schedule', () => {
  const never = none<number | null>(null)
  const idle = none(false)

  it('starts every wanted layer that has never been started', () => {
    expect(schedule(['trajectory', 'live'], 10_000, never, idle)).toEqual({ start: ['trajectory', 'live'], retryInMs: null })
  })

  it('does nothing when nothing is wanted', () => {
    expect(schedule([], 10_000, never, idle)).toEqual({ start: [], retryInMs: null })
  })

  it('never starts a second fetch for a layer that is in flight, and does not ask to be woken for it', () => {
    const plan = schedule(['trajectory', 'grid'], 10_000, never, { ...idle, trajectory: true })
    expect(plan).toEqual({ start: ['grid'], retryInMs: null })
  })

  it('holds a layer back until its own minimum interval since the last start has passed', () => {
    const lastStart = { ...never, live: 10_000, trajectory: 10_000 }
    // 150 ms later live (200) needs 50 ms more, trajectory (1000) 850 ms
    expect(schedule(['live', 'trajectory'], 10_150, lastStart, idle)).toEqual({ start: [], retryInMs: 50 })
    // exactly the interval later live may start; trajectory still waits another 800 ms
    expect(schedule(['live', 'trajectory'], 10_200, lastStart, idle)).toEqual({ start: ['live'], retryInMs: 800 })
    expect(schedule(['trajectory'], 11_000, lastStart, idle)).toEqual({ start: ['trajectory'], retryInMs: null })
  })

  it('asks to be woken at the earliest moment any held-back layer becomes due', () => {
    const lastStart = { ...never, grid: 10_000, live: 10_300, trajectory: 9_600 }
    const plan = schedule(['grid', 'live', 'trajectory'], 10_400, lastStart, idle)
    expect(plan.start).toEqual([])
    expect(plan.retryInMs).toBe(100) // grid: 600, live: 10_300 + 200 - 10_400 = 100, trajectory: 9_600 + 1000 - 10_400 = 200
  })

  it('does not count an in-flight layer toward the wake-up even when its interval has not passed', () => {
    const lastStart = { ...never, trajectory: 10_000, grid: 10_000 }
    const plan = schedule(['trajectory', 'grid'], 10_100, lastStart, { ...idle, grid: true })
    expect(plan).toEqual({ start: [], retryInMs: 900 })
  })

  it('keeps the order it was given and mixes started and held-back layers', () => {
    const lastStart = { ...never, trajectory: 9_900 }
    const plan = schedule(['grid', 'trajectory', 'live'], 10_000, lastStart, idle)
    expect(plan).toEqual({ start: ['grid', 'live'], retryInMs: 900 })
  })

  it('never waits longer than the layer interval if the clock went backwards', () => {
    const plan = schedule(['trajectory'], 5_000, { ...never, trajectory: 10_000 }, idle)
    expect(plan).toEqual({ start: [], retryInMs: 1000 })
  })
})

describe('isStale', () => {
  it('is true before any poll succeeded', () => {
    expect(isStale(null, false, 10_000)).toBe(true)
  })

  it('is true when the last poll failed, however recent the last success', () => {
    expect(isStale(9_990, true, 10_000)).toBe(true)
  })

  it('is false until the last success is older than 3000 ms', () => {
    expect(STATUS_STALE_MS).toBe(3000)
    expect(isStale(10_000, false, 10_000)).toBe(false)
    expect(isStale(10_000, false, 13_000)).toBe(false)
    expect(isStale(10_000, false, 13_001)).toBe(true)
  })
})

// ---- the session: scheduling, state and cleanup, with fake timers and a fake gateway ----------------
interface Pending {
  path: string
  signal: AbortSignal
  settled: boolean
  resolve: (v: ArrayBuffer | null) => void
  reject: (e: unknown) => void
}

const bytes = (text: string): ArrayBuffer => Uint8Array.from(new TextEncoder().encode(text)).buffer
const statusBody = (epoch: number, seq: Partial<Record<Layer, number>> = {}, stats: Record<string, unknown> = {}) =>
  JSON.stringify({ epoch, seq: { ...none(0), ...seq }, stats })

function harness(initialEnabled: Record<Layer, boolean> = ALL) {
  const pending: Pending[] = []
  const frames: { layer: Layer; text: string }[] = []
  const statuses: MapStatus[] = []
  const stale: boolean[] = []
  const decodeGate: { hold: boolean; waiting: (() => void)[] } = { hold: false, waiting: [] }

  const getBuffer = (path: string, signal: AbortSignal) =>
    new Promise<ArrayBuffer | null>((resolve, reject) => {
      const p: Pending = { path, signal, settled: false, resolve: (v) => { p.settled = true; resolve(v) }, reject: (e) => { p.settled = true; reject(e) } }
      pending.push(p)
      signal.addEventListener('abort', () => p.reject(new DOMException('aborted', 'AbortError')))
    })

  // a body is "decoded" to its own text; the text 'bad' is an invalid body
  const decode = async (layer: Layer, buf: ArrayBuffer): Promise<Frame | null> => {
    const text = new TextDecoder().decode(buf)
    if (decodeGate.hold) await new Promise<void>((r) => decodeGate.waiting.push(r))
    return text === 'bad' ? null : ({ layer, text } as unknown as Frame)
  }

  const session = createMapSession({
    getBuffer,
    decode,
    enabled: initialEnabled,
    onStatus: (s) => statuses.push(s),
    onFrame: (layer, frame) => frames.push({ layer, text: (frame as unknown as { text: string }).text }),
    onStale: (s) => stale.push(s),
    now: () => Date.now(),
  })

  const paths = () => pending.map((p) => p.path)
  const count = (path: string) => pending.filter((p) => p.path === path).length

  // Answers the oldest unanswered request for `path` (a string body, null for 503, or an Error to fail it).
  async function answer(path: string, body: string | null | Error) {
    const p = pending.find((q) => q.path === path && !q.settled)
    if (!p) throw new Error(`no pending request for ${path}; have ${paths().join(', ')}`)
    if (body instanceof Error) p.reject(body)
    else p.resolve(body === null ? null : bytes(body))
    await vi.advanceTimersByTimeAsync(0)
  }

  return {
    session, pending, frames, statuses, stale, paths, count, answer,
    holdDecode: (hold: boolean) => {
      decodeGate.hold = hold
      if (!hold) for (const r of decodeGate.waiting.splice(0)) r()
    },
  }
}

describe('createMapSession', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.setSystemTime(1_000_000)
  })
  afterEach(() => {
    vi.useRealTimers()
  })

  it('polls the status at once, then once a second, and stops entirely on stop()', async () => {
    expect(STATUS_POLL_MS).toBe(1000)
    const h = harness(only('trajectory', 'grid'))
    h.session.start()
    expect(h.paths()).toEqual(['/map'])
    await h.answer('/map', statusBody(1))
    await vi.advanceTimersByTimeAsync(999)
    expect(h.paths()).toEqual(['/map'])
    await vi.advanceTimersByTimeAsync(1)
    expect(h.paths()).toEqual(['/map', '/map'])
    h.session.stop()
    expect(h.pending[1].signal.aborted).toBe(true) // the in-flight poll is aborted
    await vi.advanceTimersByTimeAsync(10_000)
    expect(h.paths()).toEqual(['/map', '/map']) // and nothing polls again
    h.session.dispose()
  })

  it('polls four times a second while the live scan is wanted, and slows down again when it is not', async () => {
    const h = harness(only('live'))
    h.session.start()
    await h.answer('/map', statusBody(1))
    await vi.advanceTimersByTimeAsync(249)
    expect(h.count('/map')).toBe(1)
    await vi.advanceTimersByTimeAsync(1)
    expect(h.count('/map')).toBe(2)
    h.session.setEnabled(only('grid'))
    await h.answer('/map', statusBody(1))
    await vi.advanceTimersByTimeAsync(999)
    expect(h.count('/map')).toBe(2)
    await vi.advanceTimersByTimeAsync(1)
    expect(h.count('/map')).toBe(3)
    h.session.dispose()
  })

  it('does not start a second poll loop when start() is called twice', async () => {
    const h = harness()
    h.session.start()
    h.session.start()
    expect(h.count('/map')).toBe(1)
    h.session.dispose()
  })

  it('can be started again after a stop and polls at once', async () => {
    const h = harness()
    h.session.start()
    h.session.stop()
    h.session.start()
    expect(h.count('/map')).toBe(2)
    h.session.dispose()
  })

  it('fetches a changed layer once, then again only when its key changes', async () => {
    const h = harness()
    h.session.start()
    await h.answer('/map', statusBody(1, { trajectory: 3 }))
    expect(h.paths()).toEqual(['/map', '/map/trajectory'])
    await h.answer('/map/trajectory', 'e3')
    expect(h.frames).toEqual([{ layer: 'trajectory', text: 'e3' }])

    await vi.advanceTimersByTimeAsync(1000)
    await h.answer('/map', statusBody(1, { trajectory: 3 }))
    expect(h.count('/map/trajectory')).toBe(1) // same epoch:seq, held

    await vi.advanceTimersByTimeAsync(1000)
    await h.answer('/map', statusBody(1, { trajectory: 4 }))
    expect(h.count('/map/trajectory')).toBe(2)
    await h.answer('/map/trajectory', 'e4')
    expect(h.frames.map((f) => f.text)).toEqual(['e3', 'e4'])
    h.session.dispose()
  })

  it('refetches a layer after an epoch change even though its seq did not', async () => {
    const h = harness(only('grid'))
    h.session.start()
    await h.answer('/map', statusBody(1, { grid: 2 }))
    await h.answer('/map/grid', 'g-epoch1')
    await vi.advanceTimersByTimeAsync(1000)
    await h.answer('/map', statusBody(2, { grid: 2 }))
    expect(h.count('/map/grid')).toBe(2)
    await h.answer('/map/grid', 'g-epoch2')
    expect(h.frames.map((f) => f.text)).toEqual(['g-epoch1', 'g-epoch2'])
    h.session.dispose()
  })

  it('does not fetch disabled layers or layers with seq 0, and fetches a layer at once when it is enabled later', async () => {
    const h = harness(only('grid'))
    h.session.start()
    await h.answer('/map', statusBody(1, { grid: 1, trajectory: 1, live: 0 }))
    expect(h.paths()).toEqual(['/map', '/map/grid'])
    h.session.setEnabled(only('grid', 'trajectory', 'live'))
    expect(h.paths()).toEqual(['/map', '/map/grid', '/map/trajectory']) // live has nothing yet
    h.session.dispose()
  })

  it('keeps at most one request per layer in flight, then fetches the newer version as soon as the first lands', async () => {
    const h = harness(only('live'))
    h.session.start()
    await h.answer('/map', statusBody(1, { live: 1 }))
    expect(h.count('/map/live')).toBe(1)

    await vi.advanceTimersByTimeAsync(1000) // the live request is still unanswered
    await h.answer('/map', statusBody(1, { live: 2 }))
    expect(h.count('/map/live')).toBe(1)

    await h.answer('/map/live', 'live-1')
    expect(h.frames).toEqual([{ layer: 'live', text: 'live-1' }])
    expect(h.count('/map/live')).toBe(2) // the version seen while the first was in flight
    h.session.dispose()
  })

  it('marks a frame as held with the key of the status that triggered the fetch, not a newer one', async () => {
    const h = harness(only('live'))
    h.session.start()
    await h.answer('/map', statusBody(1, { live: 1 }))
    await vi.advanceTimersByTimeAsync(1000)
    await h.answer('/map', statusBody(1, { live: 2 })) // seen while seq 1 is in flight
    await h.answer('/map/live', 'live-1') // lands, holds 1:1, so 1:2 is still due
    expect(h.count('/map/live')).toBe(2)
    await h.answer('/map/live', 'live-2')
    // now both are held: a further poll with seq 2 fetches nothing
    await vi.advanceTimersByTimeAsync(1000)
    await h.answer('/map', statusBody(1, { live: 2 }))
    expect(h.count('/map/live')).toBe(2)
    h.session.dispose()
  })

  it('waits out the layer minimum interval and starts the fetch exactly when it has passed', async () => {
    const h = harness(only('grid'))
    h.session.start()
    const t0 = Date.now()
    await vi.advanceTimersByTimeAsync(500) // a slow first status: the grid fetch starts 500 ms in
    await h.answer('/map', statusBody(1, { grid: 1 }))
    await h.answer('/map/grid', 'g1')
    await vi.advanceTimersByTimeAsync(500)
    await h.answer('/map', statusBody(1, { grid: 2 })) // 500 ms after the first start: too early (grid: 1000)
    expect(h.count('/map/grid')).toBe(1)
    await vi.advanceTimersByTimeAsync(499)
    expect(h.count('/map/grid')).toBe(1)
    await vi.advanceTimersByTimeAsync(1)
    expect(h.count('/map/grid')).toBe(2)
    expect(Date.now() - t0).toBe(1500)
    h.session.dispose()
  })

  it('leaves the previous frame in place and does not mark the layer held when the gateway answers 503', async () => {
    const h = harness(only('trajectory'))
    h.session.start()
    await h.answer('/map', statusBody(1, { trajectory: 1 }))
    await h.answer('/map/trajectory', 't1')
    await vi.advanceTimersByTimeAsync(1000)
    await h.answer('/map', statusBody(1, { trajectory: 2 }))
    await h.answer('/map/trajectory', null) // 503
    expect(h.frames).toEqual([{ layer: 'trajectory', text: 't1' }])
    await vi.advanceTimersByTimeAsync(1000) // still due: tried again
    expect(h.count('/map/trajectory')).toBeGreaterThanOrEqual(3)
    h.session.dispose()
  })

  it('treats a body the decoder rejects like a 503: no frame, not held, tried again', async () => {
    const h = harness(only('grid'))
    h.session.start()
    await h.answer('/map', statusBody(1, { grid: 1 }))
    await h.answer('/map/grid', 'bad')
    expect(h.frames).toEqual([])
    await vi.advanceTimersByTimeAsync(1000)
    expect(h.count('/map/grid')).toBe(2)
    await h.answer('/map/grid', 'good')
    expect(h.frames).toEqual([{ layer: 'grid', text: 'good' }])
    h.session.dispose()
  })

  it('survives a failed layer request (gateway error) and retries it', async () => {
    const h = harness(only('live'))
    h.session.start()
    await h.answer('/map', statusBody(1, { live: 1 }))
    await h.answer('/map/live', new ApiError({ title: 'Gateway unreachable', status: 0, detail: 'cannot reach /api/v1' }))
    expect(h.frames).toEqual([])
    await vi.advanceTimersByTimeAsync(500)
    expect(h.count('/map/live')).toBe(2)
    h.session.dispose()
  })

  it('aborts in-flight layer fetches on stop() and treats the abort as no error, no frame', async () => {
    const h = harness(only('trajectory', 'live'))
    h.session.start()
    await h.answer('/map', statusBody(1, { trajectory: 1, live: 1 }))
    const inFlight = h.pending.filter((p) => p.path.startsWith('/map/'))
    expect(inFlight).toHaveLength(2)
    h.session.stop()
    await vi.advanceTimersByTimeAsync(0)
    expect(inFlight.every((p) => p.signal.aborted)).toBe(true)
    expect(h.frames).toEqual([])
    await vi.advanceTimersByTimeAsync(10_000)
    expect(h.pending).toHaveLength(3) // nothing restarts
    h.session.dispose()
  })

  it('drops a body that finishes decoding after stop()', async () => {
    const h = harness(only('live'))
    h.holdDecode(true)
    h.session.start()
    await h.answer('/map', statusBody(1, { live: 1 }))
    await h.answer('/map/live', 'scan') // fetched, decode pending
    h.session.stop()
    h.holdDecode(false)
    await vi.advanceTimersByTimeAsync(0)
    expect(h.frames).toEqual([])
    h.session.dispose()
  })

  it('reports status changes, but not a repeat of the same document', async () => {
    const h = harness()
    h.session.start()
    await h.answer('/map', statusBody(1, {}, { keyframes: 1 }))
    await vi.advanceTimersByTimeAsync(1000)
    await h.answer('/map', statusBody(1, {}, { keyframes: 1 }))
    await vi.advanceTimersByTimeAsync(1000)
    await h.answer('/map', statusBody(1, {}, { keyframes: 2 }))
    expect(h.statuses.map((s) => s.stats.keyframes)).toEqual([1, 2])
    h.session.dispose()
  })

  it('is stale after a failed poll and fresh again after a good one', async () => {
    const h = harness()
    h.session.start()
    await h.answer('/map', new ApiError({ title: 'Gateway unreachable', status: 0, detail: '' }))
    expect(h.stale.at(-1)).toBe(true)
    await vi.advanceTimersByTimeAsync(1000)
    await h.answer('/map', statusBody(1))
    expect(h.stale.at(-1)).toBe(false)
    h.session.dispose()
  })

  it('counts an unreadable, mistyped or 503 status as a failed poll and fetches nothing from it', async () => {
    const h = harness()
    h.session.start()
    await h.answer('/map', statusBody(1))
    expect(h.stale.at(-1)).toBe(false)
    for (const body of ['not json', '{"epoch":1}', null]) {
      await vi.advanceTimersByTimeAsync(1000)
      await h.answer('/map', body)
      expect(h.stale.at(-1)).toBe(true)
    }
    expect(h.paths().every((p) => p === '/map')).toBe(true)
    h.session.dispose()
  })

  it('turns stale on its own 3 s after the last successful poll, even when polling stopped', async () => {
    const h = harness()
    h.session.start()
    await h.answer('/map', statusBody(1))
    expect(h.stale.at(-1)).toBe(false)
    h.session.stop()
    await vi.advanceTimersByTimeAsync(3000)
    expect(h.stale.at(-1)).toBe(false) // exactly 3000 ms is not "older than"
    await vi.advanceTimersByTimeAsync(2)
    expect(h.stale.at(-1)).toBe(true)
    h.session.dispose()
  })

  it('dispose() cancels the stale timer as well as the polling', async () => {
    const h = harness()
    h.session.start()
    await h.answer('/map', statusBody(1))
    const before = h.stale.length
    h.session.dispose()
    await vi.advanceTimersByTimeAsync(10_000)
    expect(h.stale).toHaveLength(before)
    expect(vi.getTimerCount()).toBe(0)
  })

  it('keeps what it holds across stop() and start(): an unchanged layer is not fetched again', async () => {
    const h = harness(only('grid'))
    h.session.start()
    await h.answer('/map', statusBody(1, { grid: 1 }))
    await h.answer('/map/grid', 'g1')
    h.session.stop()
    h.session.start()
    await h.answer('/map', statusBody(1, { grid: 1 }))
    expect(h.count('/map/grid')).toBe(1)
    h.session.dispose()
  })
})
