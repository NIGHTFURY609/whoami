import { describe, expect, it } from 'vitest'
import { mapBanner, mapInputsStopped, mapRows, placeholderCalibration } from './mapStats'

const row = (rows: { k: string; v: string }[], k: string) => rows.find((r) => r.k === k)?.v

describe('map widget values', () => {
  it('formats every known stat', () => {
    const rows = mapRows({
      keyframes: 12, loop_closures: 3, path_length_m: 41.26,
      db_bytes: 52_428_800, last_update_age_s: 0.51, mode: 'mapping', calibration_placeholder: false,
    })
    expect(rows).toEqual([
      { k: 'keyframes', v: '12' },
      { k: 'closure links', v: '3' },
      { k: 'path length', v: '41.3 m' },
      { k: 'database', v: '52.4 MB' },
      { k: 'last update', v: '0.5 s' },
      { k: 'mode', v: 'MAPPING' },
    ])
  })

  it('shows the neutral placeholder for an absent, null or unusable value, never undefined or NaN', () => {
    const blank = mapRows({}).map((r) => r.v)
    expect(blank).toEqual(new Array(5).fill('—'))
    expect(mapRows(undefined).map((r) => r.v)).toEqual(blank)
    expect(mapRows({ keyframes: null, db_bytes: null }).map((r) => r.v)).toEqual(blank)
    const junk = mapRows({ keyframes: Number.NaN, path_length_m: Infinity, db_bytes: 'big', loop_closures: '3' })
    expect(junk.map((r) => r.v)).toEqual(blank)
    for (const r of [...blank, ...junk.map((j) => j.v)]) expect(r).not.toMatch(/undefined|NaN|Infinity/)
  })

  it('keeps zero, which is a real reading', () => {
    const rows = mapRows({ keyframes: 0, path_length_m: 0, db_bytes: 0, last_update_age_s: 0 })
    expect(row(rows, 'keyframes')).toBe('0')
    expect(row(rows, 'path length')).toBe('0.0 m')
    expect(row(rows, 'database')).toBe('0.0 MB')
    expect(row(rows, 'last update')).toBe('0.0 s')
  })

  it('lists the mode only when the node sent one', () => {
    expect(row(mapRows({}), 'mode')).toBeUndefined()
    expect(row(mapRows({ mode: '' }), 'mode')).toBeUndefined()
    expect(row(mapRows({ mode: null }), 'mode')).toBeUndefined()
    expect(row(mapRows({ mode: 7 }), 'mode')).toBeUndefined()
    expect(row(mapRows({ mode: 'localize' }), 'mode')).toBe('LOCALIZE')
  })

  it('flags a placeholder calibration only when it is exactly true', () => {
    expect(placeholderCalibration({ calibration_placeholder: true })).toBe(true)
    expect(placeholderCalibration({ calibration_placeholder: false })).toBe(false)
    expect(placeholderCalibration({ calibration_placeholder: null })).toBe(false)
    expect(placeholderCalibration({ calibration_placeholder: 'true' })).toBe(false)
    expect(placeholderCalibration({})).toBe(false)
    expect(placeholderCalibration(undefined)).toBe(false)
  })

  it('prints a value that rounds to negative zero as zero', () => {
    const rows = mapRows({ keyframes: -0.4, loop_closures: -0, path_length_m: -0.04, db_bytes: -1000 })
    expect(row(rows, 'keyframes')).toBe('0')
    expect(row(rows, 'closure links')).toBe('0')
    expect(row(rows, 'path length')).toBe('0.0 m')
    expect(row(rows, 'database')).toBe('0.0 MB')
    for (const r of rows) expect(r.v).not.toMatch(/-0/)
  })

  it('keeps a real negative reading negative', () => {
    expect(row(mapRows({ path_length_m: -1.26 }), 'path length')).toBe('-1.3 m')
    expect(row(mapRows({ keyframes: -3 }), 'keyframes')).toBe('-3')
  })

  it('shows a negative last-update age as 0.0 s', () => {
    expect(row(mapRows({ last_update_age_s: -0.3 }), 'last update')).toBe('0.0 s')
    expect(row(mapRows({ last_update_age_s: -12 }), 'last update')).toBe('0.0 s')
  })

})

describe('map input health rows', () => {
  it('lists rejects and restarts only when they are finite numbers above zero', () => {
    expect(mapRows({ map_rejects: 4, map_restarts: 2 }).slice(-2)).toEqual([
      { k: 'rejects', v: '4' },
      { k: 'restarts', v: '2' },
    ])
    for (const bad of [0, -1, -0, Number.NaN, Infinity, null, '3', true]) {
      const rows = mapRows({ map_rejects: bad, map_restarts: bad })
      expect(row(rows, 'rejects')).toBeUndefined()
      expect(row(rows, 'restarts')).toBeUndefined()
    }
    expect(row(mapRows({}), 'rejects')).toBeUndefined()
    expect(row(mapRows(undefined), 'restarts')).toBeUndefined()
  })

  it('keeps the base rows first and the optional rows after them', () => {
    const rows = mapRows({ keyframes: 1, mode: 'mapping', map_rejects: 3, map_restarts: 1 })
    expect(rows.map((r) => r.k)).toEqual([
      'keyframes', 'closure links', 'path length', 'database', 'last update',
      'mode', 'rejects', 'restarts',
    ])
  })

  it('attaches the last reject to the rejects row when there are rejects', () => {
    const rows = mapRows({ map_rejects: 3, map_last_reject: 'frame_grid: grid is in frame odom, not map' })
    expect(rows.find((r) => r.k === 'rejects')).toEqual({ k: 'rejects', v: '3', title: 'frame_grid: grid is in frame odom, not map' })
    expect(rows.find((r) => r.k === 'restarts')).toBeUndefined()
  })

  it('shows no last reject when it is empty, null, not a string or there are no rejects', () => {
    for (const last of ['', null, 7, true]) {
      expect(mapRows({ map_rejects: 3, map_last_reject: last }).find((r) => r.k === 'rejects')).toEqual({ k: 'rejects', v: '3' })
    }
    expect(mapRows({ map_rejects: 0, map_last_reject: 'x: y' }).find((r) => r.k === 'rejects')).toBeUndefined()
    expect(mapRows({ map_last_reject: 'x: y' }).find((r) => r.k === 'rejects')).toBeUndefined()
    // the restarts row never carries it
    expect(mapRows({ map_restarts: 2, map_last_reject: 'x: y' }).find((r) => r.k === 'restarts')).toEqual({ k: 'restarts', v: '2' })
  })
})

describe('map inputs stopped', () => {
  it('is true only when the gateway reports false', () => {
    expect(mapInputsStopped({ map_inputs_alive: false })).toBe(true)
    expect(mapInputsStopped({ map_inputs_alive: true })).toBe(false)
    expect(mapInputsStopped({ map_inputs_alive: null })).toBe(false)
    expect(mapInputsStopped({ map_inputs_alive: 'false' })).toBe(false)
    expect(mapInputsStopped({ map_inputs_alive: 0 })).toBe(false)
    expect(mapInputsStopped({})).toBe(false)
    expect(mapInputsStopped(undefined)).toBe(false)
  })
})

describe('map view banner', () => {
  const stopped = { map_inputs_alive: false }
  const STOPPED = 'MAP INPUTS STOPPED — layers are not updating'
  // A fresh status, telemetry live, a map with data, until a case says otherwise.
  const view = (o: Partial<Parameters<typeof mapBanner>[0]> = {}) =>
    mapBanner({ noMap: false, statusStale: false, telemetryLost: false, stats: {}, ...o })

  it('says nothing when the map is healthy', () => {
    expect(view()).toBeNull()
    expect(view({ stats: undefined })).toBeNull()
    expect(view({ stats: { map_inputs_alive: true } })).toBeNull()
    expect(view({ stats: { map_inputs_alive: null } })).toBeNull()
  })

  it('shows the stopped banner when the inputs are reported stopped', () => {
    expect(view({ stats: stopped })).toBe(STOPPED)
  })

  it('shows the stopped banner over an empty map too, so the operator learns why nothing arrives', () => {
    expect(view({ noMap: true, stats: stopped })).toBe(STOPPED)
  })

  it('shows the STALE banner for a stale status or lost telemetry', () => {
    expect(view({ statusStale: true })).toBe('STALE · map not updating')
    expect(view({ telemetryLost: true, stats: { map_inputs_alive: true } })).toBe('STALE · telemetry lost')
    expect(view({ statusStale: true, telemetryLost: true })).toBe('STALE · map not updating') // the status reason leads
  })

  it('lets STALE take precedence over the stopped banner on a map with data', () => {
    expect(view({ statusStale: true, stats: stopped })).toBe('STALE · map not updating')
    expect(view({ telemetryLost: true, stats: stopped })).toBe('STALE · telemetry lost')
  })

  it('keeps the existing rule that an empty map shows no STALE banner (the NO MAP YET overlay says it)', () => {
    expect(view({ noMap: true, statusStale: true })).toBeNull()
    expect(view({ noMap: true, telemetryLost: true })).toBeNull()
  })

  it('still reports stopped inputs on an empty map when only telemetry is lost: the status itself is fresh', () => {
    expect(view({ noMap: true, telemetryLost: true, stats: stopped })).toBe(STOPPED)
  })

  it('does not trust the health claim of a stale status on an empty map', () => {
    expect(view({ noMap: true, statusStale: true, stats: stopped })).toBeNull()
    expect(view({ noMap: true, statusStale: true, telemetryLost: true, stats: stopped })).toBeNull()
  })
})
