import { useEffect, useState } from 'react'
import { LAYERS, asMapStatus, getBinary, type Layer, type MapStatus } from '../source/api'
import {
  decodeCloud, decodeGrid, decodeTrajectory, type CloudFrame, type GridFrame, type TrajectoryFrame,
} from './codec'

// The map view's data. GET /map is both the layer status and the gateway's demand heartbeat, so it is polled once a
// second (four times a second while the live scan is wanted, so a new scan is seen within a quarter second) for
// exactly as long as the view is active; each layer's bytes are fetched only when that layer's epoch:seq
// changed, at a limited rate, one request at a time per layer. Read only: nothing here sends anything but GETs.
//
// The scheduling lives in plain functions (nextFetches, schedule, isStale) and createMapSession, which are tested
// without React; useMapData below only wires a session to component state.

export type Have = Record<Layer, string | null> // the epoch:seq key of the frame held for each layer
export type Enabled = Record<Layer, boolean>

export const STATUS_POLL_MS = 1000
export const STATUS_POLL_LIVE_MS = 250 // while the live layer is wanted: the status is a small JSON document
export const STATUS_STALE_MS = 3000

// Minimum time between the starts of two fetches of the same layer.
export const MIN_INTERVAL_MS: Record<Layer, number> = {
  trajectory: 1000, grid: 1000, live: 200,
}

// How long after one status poll started the next one starts.
export const statusPollMs = (enabled: Enabled): number => (enabled.live ? STATUS_POLL_LIVE_MS : STATUS_POLL_MS)

// A layer's version: a gateway restart changes the epoch while sequences start over, so the pair is the identity.
export const layerKey = (status: MapStatus, layer: Layer): string => `${status.epoch}:${status.seq[layer]}`

// The layers to fetch for this status, in LAYERS order: enabled, with data (seq > 0), and not already held.
export function nextFetches(status: MapStatus, have: Have, enabled: Enabled): Layer[] {
  return LAYERS.filter((l) => enabled[l] && status.seq[l] > 0 && have[l] !== layerKey(status, l))
}

export interface Schedule {
  start: Layer[] // fetches to begin now
  retryInMs: number | null // when to look again for layers held back only by their minimum interval
}

// Which of the wanted layers may start at nowMs. A layer in flight never starts a second request (its completion
// schedules the next look); one started less than its minimum interval ago waits, and the earliest such wait is
// returned so the caller can wake up exactly then.
export function schedule(
  wanted: readonly Layer[], nowMs: number, lastStartMs: Record<Layer, number | null>, inFlight: Record<Layer, boolean>,
): Schedule {
  const start: Layer[] = []
  let retryInMs: number | null = null
  for (const layer of wanted) {
    if (inFlight[layer]) continue
    const last = lastStartMs[layer]
    const waitMs = last === null ? 0 : MIN_INTERVAL_MS[layer] - Math.max(0, nowMs - last)
    if (waitMs <= 0) start.push(layer)
    else retryInMs = retryInMs === null ? waitMs : Math.min(retryInMs, waitMs)
  }
  return { start, retryInMs }
}

// Stale: no successful status poll yet, the last poll failed, or the last success is older than STATUS_STALE_MS.
export function isStale(lastOkMs: number | null, lastPollFailed: boolean, nowMs: number): boolean {
  return lastPollFailed || lastOkMs === null || nowMs - lastOkMs > STATUS_STALE_MS
}

// ---- frames ---------------------------------------------------------------------------------------
type FrameOf = {
  trajectory: TrajectoryFrame; grid: GridFrame; live: CloudFrame
}
export type Frame = FrameOf[Layer]
type Frames = { [L in Layer]: FrameOf[L] | null }

const EMPTY_FRAMES: Frames = { trajectory: null, grid: null, live: null }

// A body becomes a frame, or null when it is not a valid one (the previous frame is then kept).
const DECODERS: { [L in Layer]: (buf: ArrayBuffer) => FrameOf[L] | null } = {
  trajectory: decodeTrajectory,
  grid: decodeGrid,
  live: decodeCloud,
}

const decodeLayer = (layer: Layer, buf: ArrayBuffer): Frame | null => (DECODERS[layer] as (b: ArrayBuffer) => Frame | null)(buf)

// ---- the session ----------------------------------------------------------------------------------
export interface MapSessionOptions {
  getBuffer: (path: string, signal: AbortSignal) => Promise<ArrayBuffer | null> // null = 503
  decode: (layer: Layer, buf: ArrayBuffer) => Promise<Frame | null> | Frame | null // null = not a valid body
  enabled: Enabled // the layers wanted at the start; changed later with setEnabled
  onStatus: (status: MapStatus) => void // a status document that differs from the previous one
  onFrame: (layer: Layer, frame: Frame) => void
  onStale: (stale: boolean) => void
  now?: () => number // milliseconds on a monotonic clock
}

export interface MapSession {
  start(): void // begin polling (no-op while already started)
  stop(): void // abort every request and cancel the polling; what is held is kept
  setEnabled(enabled: Enabled): void // change which layers are wanted; a newly wanted layer is fetched at once
  dispose(): void // stop, and cancel the staleness timer too
}

interface Run {
  ctl: AbortController
  status: MapStatus | null
  statusText: string
  lastStart: Record<Layer, number | null>
  inFlight: Record<Layer, boolean>
  pollTimer: ReturnType<typeof setTimeout> | undefined
  retryTimer: ReturnType<typeof setTimeout> | undefined
}

const perLayer = <T>(v: T): Record<Layer, T> => Object.fromEntries(LAYERS.map((l) => [l, v])) as Record<Layer, T>

export function createMapSession(opts: MapSessionOptions): MapSession {
  const { getBuffer, decode, onStatus, onFrame, onStale } = opts
  let enabled = opts.enabled
  const now = opts.now ?? (() => performance.now())

  // Kept across stop() / start(): what is held, and which fetch is the newest.
  const have: Have = perLayer(null)
  const startedId = perLayer(0)
  const appliedId = perLayer(0)
  let lastOk: number | null = null
  let lastFailed = false
  let staleTimer: ReturnType<typeof setTimeout> | undefined
  let run: Run | null = null

  const publishStale = () => onStale(isStale(lastOk, lastFailed, now()))

  // One timer, armed from the last success, that flips `stale` when nothing newer arrived. It outlives stop():
  // a view that is no longer polling still has a status that ages.
  function armStale() {
    clearTimeout(staleTimer)
    staleTimer = undefined
    if (lastOk === null || lastFailed) return
    const leftMs = lastOk + STATUS_STALE_MS - now()
    if (leftMs < 0) return
    staleTimer = setTimeout(() => {
      publishStale()
      armStale()
    }, leftMs + 1)
  }

  function pass(r: Run) {
    clearTimeout(r.retryTimer)
    r.retryTimer = undefined
    const status = r.status
    if (r.ctl.signal.aborted || !status) return
    const plan = schedule(nextFetches(status, have, enabled), now(), r.lastStart, r.inFlight)
    for (const layer of plan.start) void fetchLayer(r, layer, layerKey(status, layer))
    if (plan.retryInMs !== null) r.retryTimer = setTimeout(() => pass(r), plan.retryInMs)
  }

  // `key` is the version the status said was current when this fetch began; that is what the frame is held as.
  async function fetchLayer(r: Run, layer: Layer, key: string) {
    r.inFlight[layer] = true
    r.lastStart[layer] = now()
    const id = ++startedId[layer]
    try {
      const buf = await getBuffer(`/map/${layer}`, r.ctl.signal)
      if (buf === null) return // nothing to serve yet: keep what is held, ask again later
      const frame = await decode(layer, buf)
      if (frame === null) return // not a valid body: same
      if (r.ctl.signal.aborted || id < appliedId[layer]) return // stopped meanwhile, or a newer fetch already landed
      appliedId[layer] = id
      have[layer] = key
      onFrame(layer, frame)
    } catch {
      // gateway error or the abort from stop(): keep the previous frame; if the layer is
      // still due it is retried after its minimum interval
    } finally {
      r.inFlight[layer] = false
      pass(r)
    }
  }

  async function poll(r: Run) {
    const startedAt = now()
    let next: MapStatus | null = null
    let text = ''
    try {
      const buf = await getBuffer('/map', r.ctl.signal)
      if (buf !== null) {
        text = new TextDecoder().decode(buf)
        next = asMapStatus(JSON.parse(text))
      }
    } catch {
      // unreachable gateway or an unreadable document: a failed poll, unless it was aborted (checked next)
    }
    if (r.ctl.signal.aborted) return
    if (next) {
      lastOk = now()
      lastFailed = false
      r.status = next
      if (text !== r.statusText) {
        r.statusText = text
        onStatus(next)
      }
      armStale()
      publishStale()
      pass(r)
    } else {
      lastFailed = true
      armStale()
      publishStale()
    }
    r.pollTimer = setTimeout(() => void poll(r), Math.max(0, statusPollMs(enabled) - (now() - startedAt)))
  }

  function stop() {
    if (!run) return
    run.ctl.abort()
    clearTimeout(run.pollTimer)
    clearTimeout(run.retryTimer)
    run = null
  }

  return {
    start() {
      if (run) return
      run = {
        ctl: new AbortController(), status: null, statusText: '', lastStart: perLayer(null), inFlight: perLayer(false),
        pollTimer: undefined, retryTimer: undefined,
      }
      void poll(run)
    },
    stop,
    setEnabled(next) {
      enabled = next
      if (run) pass(run)
    },
    dispose() {
      stop()
      clearTimeout(staleTimer)
      staleTimer = undefined
    },
  }
}

// ---- the hook -------------------------------------------------------------------------------------
const enabledKey = (e: Enabled) => LAYERS.map((l) => (e[l] ? '1' : '0')).join('')
const enabledFromKey = (key: string): Enabled => Object.fromEntries(LAYERS.map((l, i) => [l, key[i] === '1'])) as Enabled

// Frames for the layers in `enabled`, kept current while `active`; a frame is null until first received. While not
// active nothing is requested at all (the gateway stops its ROS subscriptions when nobody polls). `stale` is true
// until the first status arrives, when the last poll failed, and when the last good one is over 3 s old.
export function useMapData(active: boolean, enabled: Enabled) {
  const [frames, setFrames] = useState<Frames>(EMPTY_FRAMES)
  const [status, setStatus] = useState<MapStatus | null>(null)
  const [stale, setStale] = useState(true)
  const key = enabledKey(enabled)
  // Creating a session starts nothing (no timer, no request), so a discarded StrictMode copy is harmless.
  const [map] = useState(() =>
    createMapSession({
      getBuffer: getBinary,
      decode: decodeLayer,
      enabled: enabledFromKey(key),
      onStatus: setStatus,
      onFrame: (layer, frame) => setFrames((prev) => ({ ...prev, [layer]: frame })),
      onStale: setStale,
    }))

  useEffect(() => {
    map.setEnabled(enabledFromKey(key)) // a layer that was just switched on does not wait for the next poll
  }, [key, map])

  useEffect(() => {
    if (!active) return
    map.start()
    return () => map.stop()
  }, [active, map])

  useEffect(() => () => map.dispose(), [map])

  return { ...frames, status, stale }
}
