import type { StatValue } from '../source/api'

// Pure value formatting for the map widget (kept out of StatusWidgets.tsx so that file only exports components).
// Map stats are a flat bag whose keys may be absent, null or (from a misbehaving node) not numbers.
type Stats = Record<string, StatValue>

const MISSING = '—'

const stat = (s: Stats | undefined, key: string): number | null => {
  const v = s?.[key]
  return typeof v === 'number' && Number.isFinite(v) ? v : null
}
// A value that rounds to negative zero prints as zero ("-0.4" -> "0", "-0.04" -> "0.0"), never as "-0".
const count = (v: number | null) => {
  if (v === null) return MISSING
  const r = Math.round(v)
  return (r === 0 ? 0 : r).toLocaleString('en-US')
}
const fixed = (v: number | null, unit: string) => (v === null ? MISSING : `${v.toFixed(1).replace(/^-(0\.0)$/, '$1')} ${unit}`)
const megabytes = (bytes: number | null) => (bytes === null ? MISSING : fixed(bytes / 1e6, 'MB'))
// An age is never negative: a clock step backwards must not read "-0.3 s".
const age = (v: number | null) => (v === null ? null : Math.max(0, v))
// Only a finite number above zero is worth a row (a counter that has not moved says nothing).
const positive = (v: number | null): number | null => (v !== null && v > 0 ? v : null)

export interface MapRow {
  k: string
  v: string
  title?: string // secondary text, shown as the row's tooltip
}

export function mapRows(s: Stats | undefined): MapRow[] {
  const rows: MapRow[] = [
    { k: 'keyframes', v: count(stat(s, 'keyframes')) },
    { k: 'closure links', v: count(stat(s, 'loop_closures')) },
    { k: 'path length', v: fixed(stat(s, 'path_length_m'), 'm') },
    { k: 'database', v: megabytes(stat(s, 'db_bytes')) },
    { k: 'last update', v: fixed(age(stat(s, 'last_update_age_s')), 's') },
  ]
  const mode = s?.mode
  if (typeof mode === 'string' && mode !== '') rows.push({ k: 'mode', v: mode.toUpperCase() })
  // The gateway's own map-input health: listed only once something went wrong.
  const rejects = positive(stat(s, 'map_rejects'))
  if (rejects !== null) {
    const row: MapRow = { k: 'rejects', v: count(rejects) }
    const last = s?.map_last_reject
    if (typeof last === 'string' && last !== '') row.title = last // what the latest reject was
    rows.push(row)
  }
  const restarts = positive(stat(s, 'map_restarts'))
  if (restarts !== null) rows.push({ k: 'restarts', v: count(restarts) })
  return rows
}

export const placeholderCalibration = (s: Stats | undefined) => s?.calibration_placeholder === true

// The gateway reports that its map-input thread is gone or hung: the layers and statistics are frozen. Only an
// explicit false says so; absent, null or true shows nothing.
export const mapInputsStopped = (s: Stats | undefined) => s?.map_inputs_alive === false

const STOPPED_BANNER = 'MAP INPUTS STOPPED — layers are not updating'

export interface MapBannerInput {
  noMap: boolean // no layer has data yet
  statusStale: boolean // the map status document itself is stale (the poll failed or is old)
  telemetryLost: boolean // the telemetry stream is not live
  stats: Stats | undefined
}

// The one banner over the map view, or null.
// - A map with data and a stale status or lost telemetry: STALE, which takes precedence over the stopped banner.
// - An empty map shows no STALE banner (the NO MAP YET overlay says it), but a fresh status that reports the inputs
//   stopped is shown even then, with telemetry lost or not: it says why nothing arrives (inputs never started, or off
//   by a configuration error).
// - A stale status is not trusted for that health claim, so the stopped banner is never shown from one.
export function mapBanner({ noMap, statusStale, telemetryLost, stats }: MapBannerInput): string | null {
  const staleReason = statusStale ? 'map not updating' : telemetryLost ? 'telemetry lost' : null
  if (staleReason !== null && !noMap) return `STALE · ${staleReason}`
  if (statusStale) return null
  return mapInputsStopped(stats) ? STOPPED_BANNER : null
}
