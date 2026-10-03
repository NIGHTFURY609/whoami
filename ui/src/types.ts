// Perception Port contract (architecture.md §8): canonical 3-class mask + freshness + frame meta.

export const GW = 160 // analysis grid, image space
export const GH = 120
export const TW = 41 // top-down grid: lateral cells
export const TH = 48 // top-down grid: forward cells
export const CELL_M = 0.25
export const Z_MIN = 2.0 // ground distance of the first top-down row
export const CAM_H = 0.8 // assumed camera height above ground (m)
export const PERCEPTION_MAX_AGE_MS = 500 // safety_timeouts.yaml, perception mask

export type SourceKind = 'upload' | 'camera' | 'ros2' | 'recording'

export const CLASS_NAMES = ['UNKNOWN', 'TRAVERSABLE', 'HAZARD'] as const
export const CLASS_RGB: [number, number, number][] = [
  [110, 110, 110], // 0 unknown
  [143, 211, 255], // 1 traversable
  [255, 42, 42], // 2 hazard
]

export interface Intrinsics {
  fx: number
  fy: number
  cx: number
  cy: number
}

export interface FrameMeta {
  source: SourceKind
  frameId: string
  stamp: number // ms epoch, image time
  receivedAt?: number // ms epoch when received in browser (Date.now())
  width: number
  height: number
  K: Intrinsics // image pixels
  kAssumed: boolean // true when there is no CameraInfo
  streaming: boolean // true if this frame came from a live feed (camera live-detect, ROS 2);
  // false for an upload or a single take-photo still, which has no "going stale" concept
}

export interface Analysis {
  meta: FrameMeta
  mask: Uint8Array // GW*GH, values strictly {0,1,2}
  depth: Float32Array | null // GW*GH, metres; null when there is no depth channel
  grid: Uint8Array // TW*TH costmap, 0 free / 1 inflated / 2 lethal
  path: { x: number; z: number }[] // ground metres, x right, z forward
  pathPx: { u: number; v: number }[] // normalised 0..1 image coords
  classPct: [number, number, number]
  depthStats: { min: number; median: number; max: number } | null
  ageMs: number // now - stamp at analysis time
  latencyMs: number
  degraded: boolean
  reasons: string[]
  mock?: boolean // only ever set by the test-only fixture
}

export interface Layers {
  image: boolean
  mask: boolean
  depth: boolean
  path: boolean
}

export type Status = 'idle' | 'live' | 'still' | 'error'

export function assumedIntrinsics(width: number, height: number): Intrinsics {
  const f = 0.87 * width // ~60 deg horizontal FOV
  return { fx: f, fy: f, cx: width / 2, cy: height / 2 }
}
