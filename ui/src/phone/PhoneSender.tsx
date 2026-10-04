import { useEffect, useRef, useState } from 'react'
import {
  EXACT_VIDEO, PHONE_H, PHONE_VIDEO, PHONE_W, RttTracker, frameDue, ingestUrl, parseAck, shouldSend, sizeVerdict,
  verdictText, type BridgeStatus, type SizeVerdict,
} from './sender'

type Link = 'off' | 'connecting' | 'open' | 'closed'

interface Stats { fps: number; sent: number; skipped: number; rejected: number; rttMs: number | null; lastError: string }

const NO_STATS: Stats = { fps: 0, sent: 0, skipped: 0, rejected: 0, rttMs: null, lastError: '' }
const FPS_CHOICES = [5, 10, 15]

const initialFps = () => {
  const n = Number(new URLSearchParams(window.location.search).get('fps'))
  return FPS_CHOICES.includes(n) ? n : 15
}

// Focus and exposure held where the phone allows it (calibration needs a fixed lens; Android Chrome, not iOS).
// single-shot = adjust once, then hold; manual = hold where it is.
function lockModes(track: MediaStreamTrack): MediaTrackConstraintSet | null {
  const caps = (track.getCapabilities?.() ?? {}) as Record<string, unknown>
  const pick = (modes: unknown) =>
    Array.isArray(modes) ? ['single-shot', 'manual'].find((m) => modes.includes(m)) : undefined
  const set: Record<string, string> = {}
  const focus = pick(caps.focusMode)
  const exposure = pick(caps.exposureMode)
  if (focus) set.focusMode = focus
  if (exposure) set.exposureMode = exposure
  return Object.keys(set).length ? (set as MediaTrackConstraintSet) : null
}

// The phone as the robot camera: its 640x480 frames go to the laptop's camera bridge over /phone/ingest.
// Nothing is shown from the stack here; the overlays are in the laptop's console.
export default function PhoneSender() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const trackRef = useRef<MediaStreamTrack | null>(null)
  const [running, setRunning] = useState(false)
  const [hidden, setHidden] = useState(document.hidden)
  const [fps, setFps] = useState(initialFps)
  const fpsRef = useRef(fps) // the send loop reads the rate here, so changing it does not restart the camera
  const [error, setError] = useState('')
  const [verdict, setVerdict] = useState<SizeVerdict>({ kind: 'none' })
  const [link, setLink] = useState<Link>('off')
  const [stats, setStats] = useState<Stats>(NO_STATS)
  const [bridge, setBridge] = useState<BridgeStatus | null>(null)
  const [canLock, setCanLock] = useState(false)
  const [locked, setLocked] = useState('')

  useEffect(() => { fpsRef.current = fps }, [fps])

  useEffect(() => {
    const onVis = () => setHidden(document.hidden)
    document.addEventListener('visibilitychange', onVis)
    return () => document.removeEventListener('visibilitychange', onVis)
  }, [])

  // One session: camera + socket + send loop. A hidden page ends it (the camera stops anyway on phones, and the
  // bridge must see the phone leave rather than a frozen frame); it starts again when the page is visible.
  useEffect(() => {
    const video = videoRef.current
    if (!running || hidden || !video) return
    let stopped = false
    let stream: MediaStream | null = null
    let ws: WebSocket | null = null
    let retry: number | undefined
    let wake: WakeLockSentinel | null = null
    const rtt = new RttTracker()
    const canvas = document.createElement('canvas')
    canvas.width = PHONE_W
    canvas.height = PHONE_H
    const ctx = canvas.getContext('2d')
    const s = { ...NO_STATS }
    let lastSent = -Infinity
    let encoding = false

    const connect = (delayMs: number) => {
      if (stopped) return
      setLink('connecting')
      const sock = new WebSocket(ingestUrl(window.location))
      ws = sock
      sock.onopen = () => { rtt.reset(); setLink('open'); delayMs = 500 }
      sock.onmessage = (e) => {
        const ack = parseAck(e.data)
        if (!ack) return
        rtt.acked(performance.now())
        if (!ack.ok) {
          s.rejected += 1
          s.lastError = ack.error ?? 'rejected'
        }
      }
      sock.onclose = (e) => {
        if (ws === sock) ws = null
        if (stopped) return
        setLink('closed')
        if (e.reason) s.lastError = e.reason // e.g. another phone is already streaming
        retry = window.setTimeout(() => connect(Math.min(5000, delayMs * 2)), delayMs)
      }
    }

    const schedule = () => {
      if (stopped) return
      if ('requestVideoFrameCallback' in video) video.requestVideoFrameCallback((now) => tick(now))
      else requestAnimationFrame(tick)
    }

    const tick = (now: number) => {
      schedule()
      if (!frameDue(lastSent, now, fpsRef.current)) return
      if (sizeVerdict(video.videoWidth, video.videoHeight).kind !== 'ok' || encoding || !ctx) return
      const sock = ws
      if (!sock || sock.readyState !== WebSocket.OPEN) return
      lastSent = now
      if (!shouldSend(sock.bufferedAmount, rtt.inFlight)) {
        s.skipped += 1 // only the newest frame goes out; a backlog would arrive late and be stamped as current
        return
      }
      encoding = true
      ctx.drawImage(video, 0, 0) // 1:1, the gate above guarantees 640x480
      canvas.toBlob((blob) => {
        encoding = false
        if (!blob || stopped || sock.readyState !== WebSocket.OPEN) return
        sock.send(blob)
        rtt.sent(performance.now())
        s.sent += 1
      }, 'image/jpeg', 0.85)
    }

    let lastCount = 0
    let lastAt = performance.now()
    const ui = window.setInterval(() => {
      const now = performance.now()
      s.fps = ((s.sent - lastCount) * 1000) / (now - lastAt)
      lastCount = s.sent
      lastAt = now
      s.rttMs = rtt.median()
      setStats({ ...s })
      setVerdict(sizeVerdict(video.videoWidth, video.videoHeight))
    }, 500)

    void (async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: PHONE_VIDEO, audio: false })
        if (stopped) return stream.getTracks().forEach((t) => t.stop())
        const track = stream.getVideoTracks()[0]
        trackRef.current = track
        video.srcObject = stream
        video.muted = true
        video.playsInline = true
        await video.play()
        if (sizeVerdict(video.videoWidth, video.videoHeight).kind === 'wrong') {
          await track.applyConstraints(EXACT_VIDEO).catch(() => undefined) // the verdict then says what it gave
        }
        setCanLock(lockModes(track) !== null)
        wake = await navigator.wakeLock?.request('screen').catch(() => null)
        connect(500)
        schedule()
      } catch (e) {
        setError(`camera: ${(e as Error).message}`)
        setRunning(false)
      }
    })()

    return () => {
      stopped = true
      window.clearTimeout(retry)
      window.clearInterval(ui)
      ws?.close()
      stream?.getTracks().forEach((t) => t.stop())
      trackRef.current = null
      video.srcObject = null
      void wake?.release().catch(() => undefined)
      setLink('off')
      setLocked('')
    }
  }, [running, hidden])

  // The bridge's own view: is a phone connected, are frames accepted (polled through the dev server).
  useEffect(() => {
    if (!running) return
    let cancelled = false
    const poll = async () => {
      try {
        const r = await fetch('/phone/status', { cache: 'no-store' })
        const body = r.ok ? ((await r.json()) as BridgeStatus) : null
        if (!cancelled) setBridge(body)
      } catch {
        if (!cancelled) setBridge(null)
      }
    }
    void poll()
    const t = window.setInterval(poll, 1000)
    return () => { cancelled = true; window.clearInterval(t) }
  }, [running])

  const start = async () => {
    setError('')
    setRunning(true)
    try { // landscape where the browser allows locking it (Android, in fullscreen); elsewhere the user turns it
      await document.documentElement.requestFullscreen?.()
      const o = screen.orientation as ScreenOrientation & { lock?: (o: string) => Promise<void> }
      await o.lock?.('landscape')
    } catch { /* not supported: the size gate still holds frames until the phone is landscape */ }
  }

  const stop = () => {
    setRunning(false)
    if (document.fullscreenElement) void document.exitFullscreen().catch(() => undefined)
  }

  const lock = async () => {
    const track = trackRef.current
    const set = track && lockModes(track)
    if (!track || !set) return
    try {
      await track.applyConstraints({ advanced: [set] })
      const got = track.getSettings() as Record<string, unknown>
      setLocked(`focus ${String(got.focusMode ?? '?')}, exposure ${String(got.exposureMode ?? '?')}`)
    } catch (e) {
      setError(`lock: ${(e as Error).message}`)
    }
  }

  const sending = running && link === 'open' && verdict.kind === 'ok'
  return (
    <main className="phone">
      <div className="phone-view">
        <video ref={videoRef} muted playsInline />
        {running && verdict.kind !== 'ok' && <div className="phone-warn">{verdictText(verdict)}</div>}
        {!running && <div className="phone-idle">camera off</div>}
      </div>
      <section className="phone-side">
        <h1>UGV phone camera</h1>
        <p className="phone-dim">Hold it landscape. Only {PHONE_W}x{PHONE_H} frames are sent.</p>
        {running
          ? <button className="phone-btn stop" onClick={stop}>STOP</button>
          : <button className="phone-btn" onClick={() => void start()}>START</button>}
        <dl>
          <dt>camera</dt><dd className={verdict.kind === 'ok' ? 'ok' : 'warn'}>{running ? verdictText(verdict) : 'off'}</dd>
          <dt>link</dt><dd className={link === 'open' ? 'ok' : 'warn'}>{link}</dd>
          <dt>sending</dt><dd>{sending ? `${stats.fps.toFixed(1)} fps` : 'no'} (target {fps})</dd>
          <dt>sent / skipped</dt><dd>{stats.sent} / {stats.skipped}</dd>
          <dt>rejected</dt><dd className={stats.rejected ? 'warn' : ''}>{stats.rejected}</dd>
          <dt>round trip</dt><dd>{stats.rttMs === null ? '-' : `${Math.round(stats.rttMs)} ms (median)`}</dd>
          <dt>bridge</dt>
          <dd className={bridge?.ready ? 'ok' : 'warn'}>
            {bridge ? `${bridge.ready ? 'receiving' : bridge.connected ? 'connected, no fresh frame' : 'no phone'}; ${bridge.accepted} accepted` : running ? 'unreachable' : '-'}
          </dd>
        </dl>
        <label className="phone-dim">
          rate{' '}
          <select value={fps} onChange={(e) => setFps(Number(e.target.value))}>
            {FPS_CHOICES.map((f) => <option key={f} value={f}>{f} fps</option>)}
          </select>
        </label>
        {running && canLock && (
          <button className="phone-btn small" onClick={() => void lock()}>{locked ? 'LOCKED' : 'LOCK FOCUS + EXPOSURE'}</button>
        )}
        {locked && <p className="phone-dim">{locked}</p>}
        {(error || stats.lastError) && <p className="phone-err">{error || stats.lastError}</p>}
        <p className="phone-dim">
          Round trip / 2 is only a hint for transport_latency_s; measure it with the clock photo method
          (config/cameras/README.md).
        </p>
      </section>
    </main>
  )
}
