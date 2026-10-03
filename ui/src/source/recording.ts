// Recorded playback: a replayed video with the overlays saved while it went through the stack
// (webcam_stream.py --video records them in <video>.overlays/), fetched from the video bridge and played at the
// video's own frame rate. No ROS stack is needed. Frames are fetched ahead; when the next one is not here in time the
// playback waits for it (buffering) and then goes on from there, never skipping a frame.
import { useCallback, useEffect, useRef, useState } from 'react'
import { GH, GW, type Intrinsics } from '../types'
import type { RosDepth, RosMask } from './rosimage'
import type { CacheSummary, RecordedCamera } from './videobridge'

export interface RecordedFrame {
  index: number
  bitmap: ImageBitmap
  frameId: string
  K: Intrinsics
  mask: RosMask | null // null: perception made no mask for this frame
  depth: RosDepth | null
}

const AHEAD = 12 // frames fetched ahead of the one on screen
const PROGRESS_MS = 250 // the progress counter re-renders the console a few times a second, not every frame

// One recorded frame with its mask and depth (null when none was saved), or null when the frame itself was not
// recorded. One request per frame (the bridge's bundle: OverlayCache.bundle in webcam_stream.py), with mask and
// depth sampled onto the analysis grid by nearest pixel as the analyzer does, about 100 KB a frame.
export async function fetchRecordedFrame(
  base: string, index: number, camera: RecordedCamera, signal?: AbortSignal,
): Promise<RecordedFrame | null> {
  const r = await fetch(`${base}/video/cache/bundle/${index}?w=${GW}&h=${GH}`, { signal })
  if (!r.ok) return null
  const buf = await r.arrayBuffer()
  if (buf.byteLength < 20) return null
  const head = new DataView(buf, 0, 20)
  const [jpegLen, mw, mh, dw, dh] = [0, 4, 8, 12, 16].map((o) => head.getUint32(o, true))
  if (buf.byteLength !== 20 + jpegLen + mw * mh + dw * dh * 4) return null
  const bitmap = await createImageBitmap(new Blob([new Uint8Array(buf, 20, jpegLen)], { type: 'image/jpeg' }))
  const frameId = camera.frameId
  const maskAt = 20 + jpegLen
  const depthAt = maskAt + mw * mh
  // The saved CameraInfo, scaled to the frame as served (the same size unless the recording was made otherwise).
  const sx = bitmap.width / camera.width
  const sy = bitmap.height / camera.height
  return {
    index, bitmap, frameId,
    K: { fx: camera.k[0] * sx, fy: camera.k[4] * sy, cx: camera.k[2] * sx, cy: camera.k[5] * sy },
    mask: mw * mh > 0
      ? { stampMs: 0, frameId, width: mw, height: mh, data: new Uint8Array(buf.slice(maskAt, depthAt)), receivedAt: 0 }
      : null,
    depth: dw * dh > 0
      ? { stampMs: 0, frameId, width: dw, height: dh, data: new Float32Array(buf.slice(depthAt)), receivedAt: 0 }
      : null,
  }
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))

export function useRecordedPlayback(
  base: string, cache: CacheSummary | null, fps: number, show: (frame: RecordedFrame) => void,
) {
  const [playing, setPlaying] = useState(false)
  const [buffering, setBuffering] = useState(false)
  const [index, setIndex] = useState(0)
  const [session, setSession] = useState(0) // bumped by replay so playback restarts even while playing
  const indexRef = useRef(0)
  const showRef = useRef(show)
  const cameraRef = useRef<RecordedCamera | null>(null)
  useEffect(() => { showRef.current = show }, [show])
  useEffect(() => { cameraRef.current = cache?.camera ?? null }, [cache?.camera])
  const total = cache?.total ?? 0
  const hasCamera = !!cache?.camera

  useEffect(() => {
    const camera = cameraRef.current
    if (!playing || !camera || total <= 0 || !(fps > 0)) return
    const ctrl = new AbortController()
    const pending = new Map<number, Promise<RecordedFrame | null>>()
    const load = (i: number) => {
      let p = pending.get(i)
      if (!p) {
        p = fetchRecordedFrame(base, i, camera, ctrl.signal).catch(() => null)
        pending.set(i, p)
      }
      return p
    }
    let cancelled = false
    void (async () => {
      const period = 1000 / fps
      let i = indexRef.current
      let start = i
      let t0 = performance.now()
      let shownAt = 0
      while (!cancelled && i < total) {
        for (let k = i; k < Math.min(total, i + AHEAD); k++) void load(k)
        const p = load(i)
        let ready = false
        void p.then(() => { ready = true })
        await Promise.resolve()
        if (!ready) setBuffering(true)
        const frame = await p
        pending.delete(i)
        if (cancelled) {
          frame?.bitmap.close()
          break
        }
        if (!ready) {
          setBuffering(false)
          start = i // late: restart the clock here, never rush to catch up
          t0 = performance.now()
        }
        const wait = t0 + (i - start) * period - performance.now()
        if (wait > 0) await sleep(wait)
        if (cancelled) {
          frame?.bitmap.close()
          break
        }
        if (frame) showRef.current(frame)
        i += 1
        indexRef.current = i
        if (performance.now() - shownAt >= PROGRESS_MS || i >= total) {
          shownAt = performance.now()
          setIndex(i)
        }
      }
      if (!cancelled) setPlaying(false) // the end
    })()
    return () => {
      cancelled = true
      setIndex(indexRef.current)
      ctrl.abort()
      setBuffering(false)
      for (const p of pending.values()) void p.then((f) => f?.bitmap.close())
    }
  }, [playing, session, base, total, fps, hasCamera])

  const play = useCallback(() => {
    if (indexRef.current >= total) {
      indexRef.current = 0
      setIndex(0)
    }
    setPlaying(true)
  }, [total])
  const pause = useCallback(() => setPlaying(false), [])
  const replay = useCallback(() => {
    indexRef.current = 0
    setIndex(0)
    setSession((s) => s + 1)
    setPlaying(true)
  }, [])

  return { playing, buffering, index, total, available: hasCamera && (cache?.frames ?? 0) > 0, play, pause, replay }
}

export type RecordedPlayback = ReturnType<typeof useRecordedPlayback>
