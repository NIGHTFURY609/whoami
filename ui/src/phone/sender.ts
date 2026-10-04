// The phone camera page's logic (phone.html), kept free of the DOM so it can be tested.
//
// The phone is the robot camera: its frames go to the Windows bridge (webcam_stream.py --phone) and from there to
// the camera driver, whose calibration (config/cameras/phone_640x480.yaml) is valid only for 640x480. So frames are
// sent only when the camera itself delivers exactly 640x480, never scaled or cropped (that would change K), and only
// the newest frame goes out: with a frame still in the socket buffer or too many unanswered, the next is skipped.

export const PHONE_W = 640
export const PHONE_H = 480
export const MAX_IN_FLIGHT = 2 // frames sent and not yet acked by the bridge; more would queue in the tunnel

export const PHONE_VIDEO: MediaTrackConstraints = {
  width: { ideal: PHONE_W },
  height: { ideal: PHONE_H },
  aspectRatio: { ideal: PHONE_W / PHONE_H },
  frameRate: { ideal: 15 },
  facingMode: 'environment',
}

export const EXACT_VIDEO: MediaTrackConstraints = { width: { exact: PHONE_W }, height: { exact: PHONE_H } }

export type SizeVerdict =
  | { kind: 'ok' }
  | { kind: 'none' } // no frame yet
  | { kind: 'portrait' } // 480x640: the phone is upright
  | { kind: 'wrong'; w: number; h: number }

export function sizeVerdict(w: number, h: number): SizeVerdict {
  if (!(w > 0 && h > 0)) return { kind: 'none' }
  if (w === PHONE_W && h === PHONE_H) return { kind: 'ok' }
  if (w === PHONE_H && h === PHONE_W) return { kind: 'portrait' }
  return { kind: 'wrong', w, h }
}

export function verdictText(v: SizeVerdict): string {
  switch (v.kind) {
    case 'ok': return `${PHONE_W}x${PHONE_H}`
    case 'none': return 'waiting for the camera'
    case 'portrait': return 'turn the phone to landscape (the camera gives 480x640 upright)'
    case 'wrong': return `the camera gives ${v.w}x${v.h}; only ${PHONE_W}x${PHONE_H} matches the calibration`
  }
}

export function shouldSend(bufferedAmount: number, inFlight: number): boolean {
  return bufferedAmount === 0 && inFlight < MAX_IN_FLIGHT
}

// True when a frame at `now` keeps the rate at about `fps`: a quarter period of slack, so a camera frame landing
// just short of the period is not skipped every other time.
export function frameDue(lastSentMs: number, nowMs: number, fps: number): boolean {
  return nowMs - lastSentMs >= 750 / fps
}

export function ingestUrl(loc: { protocol: string; host: string }): string {
  return `${loc.protocol === 'https:' ? 'wss' : 'ws'}://${loc.host}/phone/ingest`
}

// Round trips from the bridge's acks, which come back in send order (one per frame).
export class RttTracker {
  private pending: number[] = []
  private samples: number[] = []
  private readonly keep: number

  constructor(keep = 30) { this.keep = keep }

  get inFlight(): number { return this.pending.length }

  sent(nowMs: number): void { this.pending.push(nowMs) }

  acked(nowMs: number): number | null {
    const t = this.pending.shift()
    if (t === undefined) return null
    const rtt = nowMs - t
    this.samples.push(rtt)
    if (this.samples.length > this.keep) this.samples.shift()
    return rtt
  }

  median(): number | null {
    if (!this.samples.length) return null
    const s = [...this.samples].sort((a, b) => a - b)
    const m = s.length >> 1
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2
  }

  reset(): void {
    this.pending = []
    this.samples = []
  }
}

export interface Ack { seq: number; ok: boolean; w?: number; h?: number; error?: string }

export function parseAck(data: unknown): Ack | null {
  if (typeof data !== 'string') return null
  try {
    const a = JSON.parse(data) as Ack
    return typeof a.seq === 'number' && typeof a.ok === 'boolean' ? a : null
  } catch {
    return null
  }
}

export interface BridgeStatus {
  connected: boolean
  ready: boolean
  accepted: number
  rejected: number
  last_rejected: string
  last_frame_age_s: number | null
  refused_senders: number
  size: [number, number]
}
