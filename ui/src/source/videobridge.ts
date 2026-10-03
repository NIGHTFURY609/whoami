// Controls of the recorded-video bridge (ugv_bringup/scripts/webcam_stream.py --video), which serves an onboard
// camera recording to the camera driver in place of a live camera. Eval only. The bridge runs on the laptop next to
// this UI, on port 8090; the live webcam bridge has no /video routes, so the panel stays hidden then.
import { useCallback, useEffect, useState } from 'react'

export type VideoState = 'ready' | 'playing' | 'buffering' | 'paused' | 'ended'

export interface VideoStatus {
  state: VideoState
  frame: number // next frame to send
  frames: number // 0 when the file does not say
  fps: number
  speed: number
  sync: boolean // wait for perception before the next frame
  loop: boolean
  perception: boolean // the bridge can see perception's output (its sync gate is connected)
  note: string
  cache: CacheSummary | null // overlays saved so far (null: the bridge does not record)
}

// CameraInfo of the recorded camera, as the bridge saved it.
export interface RecordedCamera {
  width: number
  height: number
  k: number[]
  frameId: string
}

export interface CacheSummary {
  frames: number // video frames saved
  masks: number // of them with Dev 1's mask
  depths: number
  total: number
  complete: boolean
  camera: RecordedCamera | null
}

export type VideoCommand = 'play' | 'pause' | 'replay' | 'sync-on' | 'sync-off' | 'clear-recording'

const POLL_MS = 500
const STATES: VideoState[] = ['ready', 'playing', 'buffering', 'paused', 'ended']

function parseCamera(v: unknown): RecordedCamera | null {
  if (!v || typeof v !== 'object') return null
  const o = v as Record<string, unknown>
  const k = o.k
  if (typeof o.width !== 'number' || typeof o.height !== 'number' || !Array.isArray(k) || k.length !== 9) return null
  if (!k.every((x) => typeof x === 'number' && Number.isFinite(x)) || !(k[0] > 0 && k[4] > 0)) return null
  return { width: o.width, height: o.height, k: k as number[], frameId: typeof o.frame_id === 'string' ? o.frame_id : '' }
}

function parseCache(v: unknown): CacheSummary | null {
  if (!v || typeof v !== 'object') return null
  const o = v as Record<string, unknown>
  const n = (x: unknown) => typeof x === 'number' && Number.isFinite(x) && x >= 0
  if (!n(o.frames) || !n(o.masks) || !n(o.depths) || !n(o.total) || typeof o.complete !== 'boolean') return null
  return {
    frames: o.frames as number, masks: o.masks as number, depths: o.depths as number, total: o.total as number,
    complete: o.complete, camera: parseCamera(o.camera),
  }
}

export const videoBridgeUrl = () => `http://${window.location.hostname || 'localhost'}:8090`

export function parseVideoStatus(v: unknown): VideoStatus | null {
  if (!v || typeof v !== 'object') return null
  const o = v as Record<string, unknown>
  const n = (x: unknown) => typeof x === 'number' && Number.isFinite(x)
  if (!STATES.includes(o.state as VideoState) || !n(o.frame) || !n(o.frames) || !n(o.fps) || !n(o.speed)) return null
  if (typeof o.sync !== 'boolean' || typeof o.loop !== 'boolean' || typeof o.perception !== 'boolean') return null
  return {
    state: o.state as VideoState, frame: o.frame as number, frames: o.frames as number, fps: o.fps as number,
    speed: o.speed as number, sync: o.sync, loop: o.loop, perception: o.perception,
    note: typeof o.note === 'string' ? o.note : '',
    cache: parseCache(o.cache),
  }
}

const PATHS: Record<VideoCommand, string> = {
  play: '/video/play', pause: '/video/pause', replay: '/video/replay',
  'sync-on': '/video/sync?on=1', 'sync-off': '/video/sync?on=0', 'clear-recording': '/video/cache/clear',
}

// Polls the bridge; `status` is null while no video bridge answers (the webcam bridge, or none).
export function useVideoBridge(base: string = videoBridgeUrl()) {
  const [status, setStatus] = useState<VideoStatus | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    let cancelled = false
    const poll = async () => {
      try {
        const r = await fetch(`${base}/video/status`, { cache: 'no-store' })
        const s = r.ok ? parseVideoStatus(await r.json()) : null
        if (!cancelled) setStatus(s)
      } catch {
        if (!cancelled) setStatus(null)
      }
    }
    void poll()
    const t = window.setInterval(poll, POLL_MS)
    return () => {
      cancelled = true
      window.clearInterval(t)
    }
  }, [base])

  const send = useCallback(async (cmd: VideoCommand) => {
    try {
      const r = await fetch(`${base}${PATHS[cmd]}`, { method: 'POST' })
      const s = r.ok ? parseVideoStatus(await r.json()) : null
      if (s) setStatus(s)
      setError(r.ok ? '' : `video bridge: HTTP ${r.status}`)
    } catch (e) {
      setError(`video bridge: ${(e as Error).message}`)
    }
  }, [base])

  return { status, error, send }
}
