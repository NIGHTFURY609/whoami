import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  API_BASE, ApiError, LAYERS, TELEMETRY_MAX_AGE_MS, WATCH_NAMES, applyEvent, asCommand, asGoal, asMapStatus, asNavigation,
  asPose, asProblem, asSafety, getBinary, isLive,
} from './api'

// Payloads in the exact shape ugv_api's schemas.py serialises (camelCase).
const watches = (trip?: string) =>
  WATCH_NAMES.map((name) => ({ name, ok: name !== trip, reason: name === trip ? 'stale (0.90 s > 0.50 s)' : 'ok', ageS: 0.05 }))

const safety = (trip?: string) => ({
  ok: !trip,
  watches: watches(trip),
  arbiter: { status: null, ageS: null, present: false },
  eStop: { asserted: false, assertedByGateway: false, lastSeen: null, lastSeenAgeS: null },
})

const goal = { id: 'abc', state: 'executing', frameId: 'map', x: 3, y: 0, yaw: 0, distanceRemaining: 2.5, recoveries: 0, errorCode: null, errorMessage: null }

// MapStatus / Pose exactly as the gateway's map-json-contract.md serialises them.
const mapStatus = () => ({
  epoch: 123456789,
  seq: { trajectory: 2, grid: 1, live: 4 },
  stats: { keyframes: 12, depth_hz: 3.2, mode: 'mapping', calibration_placeholder: false, last_update_age_s: null },
})
const pose = () => ({ available: true, x: 1, y: 2, z: 0, qx: 0, qy: 0, qz: 0, qw: 1, ageS: 0.05 })
const noPose = { available: false, x: null, y: null, z: null, qx: null, qy: null, qz: null, qw: null, ageS: null }

describe('gateway payload checks', () => {
  it('accepts a healthy §12 table', () => {
    const s = asSafety(safety())
    expect(s?.ok).toBe(true)
    expect(s?.watches.map((w) => w.name)).toEqual([...WATCH_NAMES])
  })

  it('never trusts an ok summary that contradicts a tripped row', () => {
    const forged = { ...safety('nav2'), ok: true }
    expect(asSafety(forged)?.ok).toBe(false)
  })

  it('rejects a table missing a §12 row or with an unknown row', () => {
    const missing = { ...safety(), watches: watches().filter((w) => w.name !== 'tf') }
    expect(asSafety(missing)).toBeNull()
    const unknown = { ...safety(), watches: [...watches(), { name: 'gps', ok: true, reason: 'ok', ageS: 0 }] }
    expect(asSafety(unknown)).toBeNull()
  })

  it('checks /cmd_vel consistency', () => {
    expect(asCommand({ available: false, linear: null, angular: null, ageS: null })?.available).toBe(false)
    expect(asCommand({ available: true, linear: null, angular: null, ageS: 0.1 })).toBeNull()
    expect(asCommand({ available: true, linear: { x: 0.2, y: 0, z: 0 }, angular: { x: 0, y: 0, z: 0.1 }, ageS: 0.1 })).not.toBeNull()
  })

  it('validates goals and navigation', () => {
    expect(asGoal(goal)?.id).toBe('abc')
    expect(asGoal({ ...goal, state: 'flying' })).toBeNull()
    expect(asGoal({ ...goal, x: 'north' })).toBeNull()
    const nav = { heartbeat: true, heartbeatAgeS: 0.02, status: 'ok', actionServerReady: true, activeGoal: goal }
    expect(asNavigation(nav)?.activeGoal?.id).toBe('abc')
    expect(asNavigation({ ...nav, activeGoal: { id: 1 } })).toBeNull()
  })

  it('reads RFC 9457 problems, including goal-gate reasons', () => {
    const p = asProblem({ type: 'about:blank', title: 'Goal gate closed', status: 409, detail: 'x', reasons: ['e_stop: asserted'] }, 409)
    expect(p).toMatchObject({ title: 'Goal gate closed', status: 409, reasons: ['e_stop: asserted'] })
    expect(asProblem('<html>', 502)).toEqual({ title: 'HTTP 502', status: 502, detail: '' })
  })
})

describe('map status and pose checks', () => {
  it('names the layers', () => {
    expect([...LAYERS]).toEqual(['trajectory', 'grid', 'live'])
  })

  it('accepts a valid map status, passing every stats key through', () => {
    const m = asMapStatus(mapStatus())
    expect(m?.epoch).toBe(123456789)
    expect(m?.seq).toEqual({ trajectory: 2, grid: 1, live: 4 })
    expect(m?.stats).toEqual({ keyframes: 12, depth_hz: 3.2, mode: 'mapping', calibration_placeholder: false, last_update_age_s: null })
  })

  it('accepts empty stats and tolerates unknown stats keys', () => {
    expect(asMapStatus({ ...mapStatus(), stats: {} })?.stats).toEqual({})
    expect(asMapStatus({ ...mapStatus(), stats: { future_key: 1.5, another: 'x' } })?.stats).toEqual({ future_key: 1.5, another: 'x' })
  })

  it('rejects a map status with a missing or non-numeric epoch', () => {
    const { epoch: _epoch, ...noEpoch } = mapStatus()
    expect(asMapStatus(noEpoch)).toBeNull()
    expect(asMapStatus({ ...mapStatus(), epoch: '1' })).toBeNull()
    expect(asMapStatus({ ...mapStatus(), epoch: null })).toBeNull()
    expect(asMapStatus({ ...mapStatus(), epoch: Number.NaN })).toBeNull()
  })

  it('rejects a map status whose seq lacks a layer or holds a non-numeric value', () => {
    for (const layer of LAYERS) {
      const { [layer]: _gone, ...rest } = mapStatus().seq
      expect(asMapStatus({ ...mapStatus(), seq: rest })).toBeNull()
    }
    expect(asMapStatus({ ...mapStatus(), seq: { ...mapStatus().seq, live: '4' } })).toBeNull()
    expect(asMapStatus({ ...mapStatus(), seq: { ...mapStatus().seq, grid: null } })).toBeNull()
    expect(asMapStatus({ ...mapStatus(), seq: null })).toBeNull()
    expect(asMapStatus({ ...mapStatus(), seq: [] })).toBeNull()
  })

  it('rejects a map status whose stats is not a flat object of scalars', () => {
    expect(asMapStatus({ ...mapStatus(), stats: null })).toBeNull()
    expect(asMapStatus({ ...mapStatus(), stats: [1] })).toBeNull()
    expect(asMapStatus({ ...mapStatus(), stats: 'none' })).toBeNull()
    expect(asMapStatus({ ...mapStatus(), stats: { nested: { a: 1 } } })).toBeNull()
    const { stats: _stats, ...noStats } = mapStatus()
    expect(asMapStatus(noStats)).toBeNull()
    expect(asMapStatus('map')).toBeNull()
    expect(asMapStatus(null)).toBeNull()
  })

  it('accepts an available pose and the not-yet-seen pose (all null), rejects anything inconsistent', () => {
    expect(asPose(pose())).toMatchObject({ available: true, x: 1, y: 2, qw: 1, ageS: 0.05 })
    expect(asPose(noPose)?.available).toBe(false)
    expect(asPose({ ...pose(), available: false })).toBeNull() // "unavailable" must carry no coordinates
    expect(asPose({ ...pose(), x: null })).toBeNull() // "available" must carry all of them
    expect(asPose({ ...pose(), qw: undefined })).toBeNull()
  })

  it('rejects a malformed pose', () => {
    expect(asPose({ ...pose(), y: '2' })).toBeNull()
    expect(asPose({ ...pose(), ageS: 'fresh' })).toBeNull()
    expect(asPose({ ...pose(), available: 1 })).toBeNull()
    const { available: _available, ...noFlag } = pose()
    expect(asPose(noFlag)).toBeNull()
    expect(asPose(null)).toBeNull()
    expect(asPose([])).toBeNull()
  })
})

describe('telemetry stream', () => {
  it('applies valid events and ignores malformed ones', () => {
    const t1 = applyEvent(null, 'safety', JSON.stringify(safety()), 1000)
    expect(t1?.safety?.ok).toBe(true)
    expect(applyEvent(t1, 'safety', '{"ok": true}', 1100)).toBe(t1)
    expect(applyEvent(t1, 'safety', 'not json', 1100)).toBe(t1)
    expect(applyEvent(t1, 'unknown', '{}', 1100)).toBe(t1)
  })

  it('applies map and pose events alongside the existing four, ignoring malformed ones', () => {
    const t1 = applyEvent(null, 'safety', JSON.stringify(safety()), 1000)
    const t2 = applyEvent(t1, 'map', JSON.stringify(mapStatus()), 1100)
    const t3 = applyEvent(t2, 'pose', JSON.stringify(pose()), 1200)
    expect(t3?.safety?.ok).toBe(true) // earlier events survive
    expect(t3?.map?.epoch).toBe(123456789)
    expect(t3?.pose?.x).toBe(1)
    expect(t3?.at).toBe(1200) // freshness is tracked the same way: newest accepted event
    expect(applyEvent(t3, 'map', '{"epoch": 1}', 1300)).toBe(t3)
    expect(applyEvent(t3, 'pose', JSON.stringify({ ...pose(), x: null }), 1300)).toBe(t3)
    expect(applyEvent(t3, 'pose', 'not json', 1300)).toBe(t3)
    const unseen = applyEvent(null, 'pose', JSON.stringify(noPose), 1000)
    expect(unseen?.pose?.available).toBe(false)
  })

  it('goes NO SIGNAL once the stream stops', () => {
    const t = applyEvent(null, 'safety', JSON.stringify(safety()), 1000)
    expect(isLive(t, 1000 + TELEMETRY_MAX_AGE_MS)).toBe(true)
    expect(isLive(t, 1001 + TELEMETRY_MAX_AGE_MS)).toBe(false)
    expect(isLive(null, 0)).toBe(false)
  })
})

describe('binary fetch', () => {
  afterEach(() => vi.unstubAllGlobals())
  const problem = { type: 'about:blank', title: 'No data yet', status: 503, detail: 'live has no data' }
  const stubFetch = (res: Response | Error) => {
    const fetchMock = vi.fn(async (_url: string, _init?: RequestInit) => {
      if (res instanceof Error) throw res
      return res
    })
    vi.stubGlobal('fetch', fetchMock)
    return fetchMock
  }

  it('returns the body bytes on 200 and passes the URL and abort signal through', async () => {
    const fetchMock = stubFetch(new Response(new Uint8Array([1, 2, 3, 4]), { status: 200 }))
    const ctl = new AbortController()
    const buf = await getBinary('/map/live', ctl.signal)
    expect(buf).not.toBeNull()
    expect([...new Uint8Array(buf as ArrayBuffer)]).toEqual([1, 2, 3, 4])
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(fetchMock.mock.calls[0][0]).toBe(`${API_BASE}/map/live`)
    expect(fetchMock.mock.calls[0][1]?.signal).toBe(ctl.signal)
  })

  it('returns null on 503 (layer has no data yet)', async () => {
    stubFetch(new Response(JSON.stringify(problem), { status: 503, headers: { 'Content-Type': 'application/problem+json' } }))
    expect(await getBinary('/map/live', new AbortController().signal)).toBeNull()
  })

  it('throws the gateway problem for any other failure', async () => {
    stubFetch(new Response(JSON.stringify({ ...problem, title: 'Not found', status: 404, detail: 'no such layer' }), { status: 404 }))
    const err = await getBinary('/map/nope', new AbortController().signal).catch((e: unknown) => e)
    expect(err).toBeInstanceOf(ApiError)
    expect((err as ApiError).problem).toMatchObject({ title: 'Not found', status: 404, detail: 'no such layer' })
    stubFetch(new Response('<html>', { status: 502 }))
    const err2 = await getBinary('/map/live', new AbortController().signal).catch((e: unknown) => e)
    expect((err2 as ApiError).problem).toEqual({ title: 'HTTP 502', status: 502, detail: '' })
  })

  it('reports an unreachable gateway like every other request', async () => {
    stubFetch(new TypeError('Failed to fetch'))
    const err = await getBinary('/map/live', new AbortController().signal).catch((e: unknown) => e)
    expect(err).toBeInstanceOf(ApiError)
    expect((err as ApiError).problem).toMatchObject({ title: 'Gateway unreachable', status: 0 })
  })

  it('lets an abort through untouched so callers can tell it from a failure', async () => {
    const ctl = new AbortController()
    stubFetch(new DOMException('aborted', 'AbortError'))
    ctl.abort()
    const err = await getBinary('/map/live', ctl.signal).catch((e: unknown) => e)
    expect(err).not.toBeInstanceOf(ApiError)
    expect((err as DOMException).name).toBe('AbortError')
  })
})

describe('operator boundary', () => {
  const sources = import.meta.glob('/src/**/*.{ts,tsx}', { query: '?raw', import: 'default', eager: true }) as Record<string, string>
  const production = Object.entries(sources).filter(([path]) => !/\.test\.tsx?$/.test(path))
  // The camera view (the main page, its rosbridge reader, source picker and perception widgets) is display-only,
  // by the owner's call.
  const LIVE_VIEW = /\/src\/(components\/(CameraView|Viewport|TopDownMap|Inspector|SourcePanel)\.tsx|source\/(rosbridge|rosimage|camera|useCameraSource)\.ts|analysis\/.*|types\.ts)$/
  // A direct costmap topic name ('/global_costmap/costmap', "/local_costmap/costmap", a namespaced one, a template
  // literal, or a bare '/local_costmap') is banned outside the live view; the plain word may appear in labels and prose.
  const COSTMAP_TOPIC = /['"`]\/[\w/]*costmap\b/

  it('bans direct costmap topic names but not the word in a label', () => {
    expect(COSTMAP_TOPIC.test(`'/global_costmap/costmap'`)).toBe(true)
    expect(COSTMAP_TOPIC.test(`"/local_costmap/costmap"`)).toBe(true)
    expect(COSTMAP_TOPIC.test(`'/costmap/costmap_updates'`)).toBe(true)
    expect(COSTMAP_TOPIC.test(`'/ugv/local_costmap/costmap'`)).toBe(true) // namespaced
    expect(COSTMAP_TOPIC.test('`/global_costmap/costmap`')).toBe(true) // template literal
    expect(COSTMAP_TOPIC.test('`/ugv/local_costmap/costmap`')).toBe(true)
    expect(COSTMAP_TOPIC.test(`'/local_costmap'`)).toBe(true) // no trailing slash
    expect(COSTMAP_TOPIC.test(`"/costmap"`)).toBe(true)
    expect(COSTMAP_TOPIC.test(`'Costmap'`)).toBe(false)
    expect(COSTMAP_TOPIC.test(`label: 'Costmap grid'`)).toBe(false)
    expect(COSTMAP_TOPIC.test('// the costmap is built server-side')).toBe(false)
  })

  it('never commands motion and never publishes to ROS', () => {
    expect(API_BASE).toBe('/api/v1')
    const forbidden = [/op:\s*['"]publish/, /op:\s*['"](advertise|call_service|send_action_goal)/, /cmd_vel_nav2/]
    const offenders = production.flatMap(([path, code]) => forbidden.filter((re) => re.test(code)).map((re) => `${path}: ${re}`))
    expect(offenders).toEqual([])
  })

  it('controls go only through the gateway: rosbridge is confined to the read-only live view', () => {
    const forbidden = [/new WebSocket/, /rosbridge/i, /['"]\/segmentation\//, /['"]\/perception\//, /['"]\/odom['"]/, /['"]\/tf['"]/,
      /['"]\/plan['"]/, COSTMAP_TOPIC]
    const offenders = production
      .filter(([path]) => !LIVE_VIEW.test(path))
      .flatMap(([path, code]) => forbidden
        .filter((re) => !(re.source === 'new WebSocket' && PHONE_SENDER.test(path))) // checked on its own below
        .filter((re) => re.test(code)).map((re) => `${path}: ${re}`))
    expect(offenders).toEqual([])
  })

  // The phone camera page (phone.html) is a camera, not a console: its one socket goes to the camera bridge's
  // ingest (/phone/ingest, frames only), never to rosbridge or the gateway's controls.
  const PHONE_SENDER = /\/src\/phone\/PhoneSender\.tsx$/
  it('the phone camera page only opens its socket to the camera ingest', () => {
    const phone = production.filter(([path]) => /\/src\/phone\//.test(path))
    expect(phone.some(([path]) => PHONE_SENDER.test(path))).toBe(true)
    for (const [path, code] of phone) {
      const sockets = code.match(/new WebSocket\([^)]*\)/g) ?? []
      expect(sockets.every((s) => s === 'new WebSocket(ingestUrl(window.location)'), path).toBe(true)
      expect(code, path).not.toMatch(/\/api\/|rosbridge|:9090/i)
    }
  })
})
