// Client for Dev 5's operator gateway (ugv_nav/ugv_api, REST /api/v1 + SSE). This is the operator console's
// only connection to the robot: it never talks to Dev 1, 2 or 4 directly and never commands /cmd_vel.
// Every payload is checked before use; anything malformed is treated as "no data" (fail closed).

export const API_BASE = '/api/v1'

export const WATCH_NAMES = ['camera', 'perception', 'localization', 'tf', 'nav2', 'e_stop'] as const
export type WatchName = (typeof WATCH_NAMES)[number]

export interface Watch {
  name: WatchName
  ok: boolean
  reason: string
  ageS: number | null
}

export interface EStop {
  asserted: boolean
  assertedByGateway: boolean
  lastSeen: boolean | null
  lastSeenAgeS: number | null
}

export interface SafetyStatus {
  ok: boolean
  watches: Watch[]
  arbiter: { status: string | null; ageS: number | null; present: boolean }
  eStop: EStop
}

export interface Vector3 { x: number; y: number; z: number }

export interface BaseCommand {
  available: boolean
  linear: Vector3 | null
  angular: Vector3 | null
  ageS: number | null
}

export type Mode = 'mapping' | 'localize'

export interface Localization {
  poseValid: boolean | null
  poseValidAgeS: number | null
  status: string | null
  requestedMode: Mode | null
}

export const GOAL_STATES = ['pending', 'rejected', 'executing', 'canceling', 'succeeded', 'aborted', 'canceled', 'failed'] as const
export type GoalState = (typeof GOAL_STATES)[number]

export interface Goal {
  id: string
  state: GoalState
  frameId: string
  x: number
  y: number
  yaw: number
  distanceRemaining: number | null
  recoveries: number | null
  errorCode: number | null
  errorMessage: string | null
}

export interface Navigation {
  heartbeat: boolean | null
  heartbeatAgeS: number | null
  status: string | null
  actionServerReady: boolean
  activeGoal: Goal | null
}

// The map layers the gateway serves (GET /map/{layer}); MapStatus.seq always carries every one of them.
export const LAYERS = ['trajectory', 'grid', 'live'] as const
export type Layer = (typeof LAYERS)[number]

export type StatValue = number | string | boolean | null

export interface MapStatus {
  epoch: number // random per gateway process: a change means the layer sequences restarted
  seq: Record<Layer, number> // per-layer change counter, 0 = nothing received yet
  stats: Record<string, StatValue> // snake_case keys passed through from the ROS stats nodes; unknown keys are fine
}

// map -> base_link. Until a transform has been seen: available false and every coordinate null.
export interface Pose {
  available: boolean
  x: number | null
  y: number | null
  z: number | null
  qx: number | null
  qy: number | null
  qz: number | null
  qw: number | null
  ageS: number | null
}

// RFC 9457 problem details, as the gateway returns them.
export interface Problem {
  title: string
  status: number
  detail: string
  reasons?: string[]
  errors?: { loc: (string | number)[]; msg: string }[]
}

export class ApiError extends Error {
  readonly problem: Problem
  constructor(problem: Problem) {
    super(problem.detail || problem.title)
    this.problem = problem
  }
}

// ---- payload checks -------------------------------------------------------------------------------
type Obj = Record<string, unknown>
const isObj = (v: unknown): v is Obj => typeof v === 'object' && v !== null && !Array.isArray(v)
const isNum = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v)
const optNum = (v: unknown): v is number | null => v === null || isNum(v)
const optStr = (v: unknown): v is string | null => v === null || typeof v === 'string'
const optBool = (v: unknown): v is boolean | null => v === null || typeof v === 'boolean'
const isVec = (v: unknown): v is Vector3 => isObj(v) && isNum(v.x) && isNum(v.y) && isNum(v.z)
const isStat = (v: unknown): v is StatValue => v === null || isNum(v) || typeof v === 'string' || typeof v === 'boolean'
const POSE_COORDS = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'] as const

function asWatch(v: unknown): Watch | null {
  if (!isObj(v) || !WATCH_NAMES.includes(v.name as WatchName)) return null
  if (typeof v.ok !== 'boolean' || typeof v.reason !== 'string' || !optNum(v.ageS)) return null
  return v as unknown as Watch
}

export function asEStop(v: unknown): EStop | null {
  if (!isObj(v) || typeof v.asserted !== 'boolean' || typeof v.assertedByGateway !== 'boolean') return null
  if (!optBool(v.lastSeen) || !optNum(v.lastSeenAgeS)) return null
  return v as unknown as EStop
}

export function asSafety(v: unknown): SafetyStatus | null {
  if (!isObj(v) || typeof v.ok !== 'boolean' || !Array.isArray(v.watches) || !isObj(v.arbiter)) return null
  const watches = v.watches.map(asWatch)
  if (watches.some((w) => w === null)) return null
  const names = (watches as Watch[]).map((w) => w.name)
  if (WATCH_NAMES.some((n) => !names.includes(n))) return null // all six §12 rows or nothing
  const a = v.arbiter
  if (!optStr(a.status) || !optNum(a.ageS) || typeof a.present !== 'boolean') return null
  const eStop = asEStop(v.eStop)
  if (!eStop) return null
  // never trust a summary that contradicts its rows
  const ok = v.ok && (watches as Watch[]).every((w) => w.ok)
  return { ok, watches: watches as Watch[], arbiter: a as SafetyStatus['arbiter'], eStop }
}

export function asCommand(v: unknown): BaseCommand | null {
  if (!isObj(v) || typeof v.available !== 'boolean' || !optNum(v.ageS)) return null
  if (v.available ? !(isVec(v.linear) && isVec(v.angular)) : !(v.linear === null && v.angular === null)) return null
  return v as unknown as BaseCommand
}

export function asLocalization(v: unknown): Localization | null {
  if (!isObj(v) || !optBool(v.poseValid) || !optNum(v.poseValidAgeS) || !optStr(v.status)) return null
  if (!(v.requestedMode === null || v.requestedMode === 'mapping' || v.requestedMode === 'localize')) return null
  return v as unknown as Localization
}

export function asGoal(v: unknown): Goal | null {
  if (!isObj(v) || typeof v.id !== 'string' || !GOAL_STATES.includes(v.state as GoalState)) return null
  if (typeof v.frameId !== 'string' || !isNum(v.x) || !isNum(v.y) || !isNum(v.yaw)) return null
  if (!optNum(v.distanceRemaining) || !optNum(v.recoveries) || !optNum(v.errorCode) || !optStr(v.errorMessage)) return null
  return v as unknown as Goal
}

export function asNavigation(v: unknown): Navigation | null {
  if (!isObj(v) || !optBool(v.heartbeat) || !optNum(v.heartbeatAgeS) || !optStr(v.status)) return null
  if (typeof v.actionServerReady !== 'boolean') return null
  const activeGoal = v.activeGoal === null ? null : asGoal(v.activeGoal)
  if (v.activeGoal !== null && !activeGoal) return null
  return { ...(v as unknown as Navigation), activeGoal }
}

export function asMapStatus(v: unknown): MapStatus | null {
  if (!isObj(v) || !isNum(v.epoch) || !isObj(v.seq) || !isObj(v.stats)) return null
  const s = v.seq
  if (LAYERS.some((l) => !isNum(s[l]))) return null // every layer or nothing
  if (!Object.values(v.stats).every(isStat)) return null // flat scalars only
  const seq = Object.fromEntries(LAYERS.map((l) => [l, s[l] as number])) as Record<Layer, number>
  return { epoch: v.epoch, seq, stats: { ...(v.stats as Record<string, StatValue>) } }
}

export function asPose(v: unknown): Pose | null {
  if (!isObj(v) || typeof v.available !== 'boolean' || !optNum(v.ageS)) return null
  if (v.available ? !POSE_COORDS.every((k) => isNum(v[k])) : !POSE_COORDS.every((k) => v[k] === null)) return null
  return v as unknown as Pose
}

export function asProblem(v: unknown, status: number): Problem {
  if (isObj(v) && typeof v.title === 'string') {
    return {
      title: v.title,
      status: isNum(v.status) ? v.status : status,
      detail: typeof v.detail === 'string' ? v.detail : '',
      reasons: Array.isArray(v.reasons) ? v.reasons.filter((r): r is string => typeof r === 'string') : undefined,
      errors: Array.isArray(v.errors) ? (v.errors as Problem['errors']) : undefined,
    }
  }
  return { title: `HTTP ${status}`, status, detail: '' }
}

// ---- requests -------------------------------------------------------------------------------------
async function request<T>(method: string, path: string, parse: (v: unknown) => T | null, body?: unknown): Promise<T> {
  let res: Response
  try {
    res = await fetch(API_BASE + path, {
      method,
      headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
    })
  } catch {
    throw new ApiError({ title: 'Gateway unreachable', status: 0, detail: `cannot reach ${API_BASE}` })
  }
  let json: unknown = null
  try { json = await res.json() } catch { /* empty or non-JSON body */ }
  if (!res.ok) throw new ApiError(asProblem(json, res.status))
  const parsed = parse(json)
  if (parsed === null) throw new ApiError({ title: 'Bad response', status: res.status, detail: `unexpected payload from ${path}` })
  return parsed
}

// Binary layer download (GET /map/{layer}). Resolves null on 503: the layer has no data yet, which is a normal state, not
// an error. Any other failure throws like `request`; an abort via `signal` is rethrown untouched so callers can tell it
// from a failure.
export async function getBinary(path: string, signal: AbortSignal): Promise<ArrayBuffer | null> {
  let res: Response
  try {
    res = await fetch(API_BASE + path, { signal })
  } catch (e) {
    if (signal.aborted) throw e
    throw new ApiError({ title: 'Gateway unreachable', status: 0, detail: `cannot reach ${API_BASE}` })
  }
  if (res.status === 503) return null
  if (!res.ok) {
    let json: unknown = null
    try { json = await res.json() } catch { /* empty or non-JSON body */ }
    throw new ApiError(asProblem(json, res.status))
  }
  try {
    return await res.arrayBuffer()
  } catch (e) {
    if (signal.aborted) throw e
    throw new ApiError({ title: 'Bad response', status: res.status, detail: `truncated payload from ${path}` })
  }
}

export const api = {
  setEstop: (asserted: boolean) => request('PUT', '/safety/e-stop', asEStop, { asserted }),
  setMode: (mode: Mode) => request('PUT', '/localization/mode', (v) => (isObj(v) && v.mode === mode ? mode : null), { mode }),
  sendGoal: (x: number, y: number, yaw: number) =>
    request('POST', '/navigation/goals', asGoal, { x, y, yaw, frameId: 'map' }),
  cancelGoal: (id: string) => request('DELETE', `/navigation/goals/${encodeURIComponent(id)}`, asGoal),
}

// ---- telemetry (SSE) ------------------------------------------------------------------------------
export interface Telemetry {
  safety?: SafetyStatus
  command?: BaseCommand
  localization?: Localization
  navigation?: Navigation
  map?: MapStatus
  pose?: Pose
  at: number // browser ms of the newest accepted event
}

// The stream is the console's heartbeat: once no event has arrived for this long, everything reads NO SIGNAL.
export const TELEMETRY_MAX_AGE_MS = 2000

export function isLive(t: Telemetry | null, now: number): t is Telemetry {
  return !!t && now - t.at <= TELEMETRY_MAX_AGE_MS
}

const PARSERS = {
  safety: asSafety, command: asCommand, localization: asLocalization, navigation: asNavigation, map: asMapStatus, pose: asPose,
} as const

export function applyEvent(prev: Telemetry | null, event: string, data: string, now: number): Telemetry | null {
  const parse = PARSERS[event as keyof typeof PARSERS]
  if (!parse) return prev
  let json: unknown
  try { json = JSON.parse(data) } catch { return prev }
  const value = parse(json)
  if (value === null) return prev // malformed: keep waiting; staleness takes over if it persists
  return { ...(prev ?? { at: now }), [event]: value, at: now }
}

export function subscribeTelemetry(onUpdate: (t: Telemetry) => void, onConnection: (open: boolean) => void): () => void {
  let state: Telemetry | null = null
  const es = new EventSource(`${API_BASE}/telemetry/stream`)
  es.onopen = () => onConnection(true)
  es.onerror = () => onConnection(false) // EventSource reconnects on its own (server sends retry:)
  for (const name of Object.keys(PARSERS)) {
    es.addEventListener(name, (ev) => {
      const next = applyEvent(state, name, (ev as MessageEvent<string>).data, Date.now())
      if (next && next !== state) {
        state = next
        onUpdate(next)
      }
    })
  }
  return () => es.close()
}
