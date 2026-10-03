// Holds robot camera frames until Dev 1's overlay for that exact frame has arrived, so a replayed video and its
// mask / depth are shown together instead of the picture running ahead of its overlay ("buffering"). Frames are
// matched by the image stamp, which the mask and the depth image copy (Perception Port §8.4).
//
// A frame is released once its mask is here and either its depth is too or `depthWaitMs` has passed since the mask
// (depth comes after the mask; a missing depth must not freeze the view). Releasing a frame drops every older one:
// a frame whose mask never arrives (throttled, degraded) is skipped, never shown late.

const key = (stampMs: number) => Math.round(stampMs * 1000) // microseconds: both sides compute the same float

export interface Released<T> {
  stampMs: number
  frame: T
}

export class FrameSync<T> {
  private frames = new Map<number, { stampMs: number; frame: T }>()
  private masks = new Map<number, number>() // stamp key -> when the mask arrived
  private depths = new Set<number>()

  private readonly dispose: (frame: T) => void
  private readonly depthWaitMs: number
  private readonly maxFrames: number

  constructor(dispose: (frame: T) => void, depthWaitMs = 600, maxFrames = 30) {
    this.dispose = dispose
    this.depthWaitMs = depthWaitMs
    this.maxFrames = maxFrames
  }

  get pending(): number {
    return this.frames.size
  }

  addFrame(stampMs: number, frame: T): void {
    const k = key(stampMs)
    const old = this.frames.get(k)
    if (old) this.dispose(old.frame)
    this.frames.set(k, { stampMs, frame })
    while (this.frames.size > this.maxFrames) {
      const oldest = Math.min(...this.frames.keys())
      this.dispose(this.frames.get(oldest)!.frame)
      this.frames.delete(oldest)
    }
  }

  noteMask(stampMs: number, now: number): void {
    if (!this.masks.has(key(stampMs))) this.masks.set(key(stampMs), now)
  }

  noteDepth(stampMs: number): void {
    this.depths.add(key(stampMs))
  }

  // The newest frame whose overlay is complete, or null. Older frames are disposed.
  take(now: number): Released<T> | null {
    const ready = [...this.frames.keys()]
      .filter((k) => {
        const maskAt = this.masks.get(k)
        return maskAt !== undefined && (this.depths.has(k) || now - maskAt >= this.depthWaitMs)
      })
      .sort((a, b) => b - a)[0]
    if (ready === undefined) return null
    const out = this.frames.get(ready)!
    for (const k of [...this.frames.keys()]) {
      if (k < ready) this.dispose(this.frames.get(k)!.frame)
      if (k <= ready) this.frames.delete(k)
    }
    for (const k of [...this.masks.keys()]) if (k <= ready) this.masks.delete(k)
    for (const k of [...this.depths]) if (k <= ready) this.depths.delete(k)
    return out
  }

  clear(): void {
    for (const f of this.frames.values()) this.dispose(f.frame)
    this.frames.clear()
    this.masks.clear()
    this.depths.clear()
  }
}
