import { describe, expect, it } from 'vitest'
import {
  DEFAULT_TOGGLES, enabledLayers, parseToggles, parseView, slotOnError, slotOnProps, slotState, toggled, withLiveMode,
  type MapToggles,
} from './mapToggles'

describe('persisted toggles', () => {
  it('falls back to the defaults for nothing stored or anything unreadable', () => {
    expect(parseToggles(null)).toEqual(DEFAULT_TOGGLES)
    expect(parseToggles('')).toEqual(DEFAULT_TOGGLES)
    expect(parseToggles('not json')).toEqual(DEFAULT_TOGGLES)
    expect(parseToggles('[true, false]')).toEqual(DEFAULT_TOGGLES)
    expect(parseToggles('null')).toEqual(DEFAULT_TOGGLES)
    expect(parseToggles('42')).toEqual(DEFAULT_TOGGLES)
  })

  it('shows every layer by default', () => {
    expect(DEFAULT_TOGGLES).toEqual({ live: true, liveMode: 'terrain', trajectory: true, grid: true, images: true })
  })

  it('reads each stored value it understands and keeps the default for the rest', () => {
    // keys of removed layers (the map cloud, elevation and its colour mode) are ignored, like any other unknown key
    const t = parseToggles(JSON.stringify({ cloud: false, grid: 'no', images: false, elevation: true, mode: 'obstacle', liveMode: 'scan' }))
    expect(t).toEqual({ ...DEFAULT_TOGGLES, images: false, liveMode: 'scan' })
    expect(t).not.toHaveProperty('cloud')
    expect(parseToggles(JSON.stringify({ liveMode: 'mesh' })).liveMode).toBe('terrain')
  })

  it('round-trips what the view writes', () => {
    const t: MapToggles = { live: true, liveMode: 'scan', trajectory: false, grid: false, images: true }
    expect(parseToggles(JSON.stringify(t))).toEqual(t)
  })

  it('returns a new state and leaves the one it was given alone (React state must not be mutated)', () => {
    const given: MapToggles = { ...DEFAULT_TOGGLES }
    const next = toggled(given, 'grid')
    expect(next).not.toBe(given)
    expect(given).toEqual(DEFAULT_TOGGLES)
    expect(next.grid).toBe(!DEFAULT_TOGGLES.grid)
  })

  it('flips one toggle from the state it is given, so two flips in one render both count', () => {
    const once = toggled(DEFAULT_TOGGLES, 'trajectory')
    expect(once).toEqual({ ...DEFAULT_TOGGLES, trajectory: false })
    expect(toggled(once, 'trajectory')).toEqual(DEFAULT_TOGGLES)
    // two updaters queued against the same render: the second sees the first's result
    const queued = [(t: MapToggles) => toggled(t, 'grid'), (t: MapToggles) => toggled(t, 'live')]
    expect(queued.reduce((t, f) => f(t), DEFAULT_TOGGLES)).toEqual({ ...DEFAULT_TOGGLES, grid: false, live: false })
    expect(DEFAULT_TOGGLES.grid).toBe(true) // never mutated
  })

  it('switches the live mode without touching the rest', () => {
    const scan = withLiveMode(DEFAULT_TOGGLES, 'scan')
    expect(scan).toEqual({ ...DEFAULT_TOGGLES, liveMode: 'scan' })
    expect(DEFAULT_TOGGLES.liveMode).toBe('terrain')
  })

  it('turns toggles into the layers to fetch; the image panels fetch nothing', () => {
    const t: MapToggles = { ...DEFAULT_TOGGLES, live: true, trajectory: false, grid: false, images: false }
    expect(enabledLayers(t)).toEqual({ live: true, trajectory: false, grid: false })
    expect(enabledLayers({ ...t, liveMode: 'scan' })).toEqual(enabledLayers(t)) // both modes fetch the same layer
    expect(enabledLayers({ ...t, images: true })).toEqual(enabledLayers(t))
  })
})

describe('main view choice', () => {
  it('is the map only when the map was saved', () => {
    expect(parseView('map')).toBe('map')
    expect(parseView('camera')).toBe('camera')
    expect(parseView(null)).toBe('camera')
    expect(parseView('MAP')).toBe('camera')
    expect(parseView('"map"')).toBe('camera')
  })
})

describe('view slot error state', () => {
  it('starts without a failure, keyed to the view', () => {
    expect(slotState('map:0')).toEqual({ key: 'map:0', failed: null })
  })

  it.each([
    'Failed to fetch dynamically imported module: http://robot/assets/MapView-abc.js', // Chromium
    'error loading dynamically imported module: http://robot/assets/MapView-abc.js', // Firefox
    'Importing a module script failed.', // Safari
    'Unable to preload CSS for /assets/MapView-abc.css', // Vite's preload helper
  ])('tells a chunk that could not be downloaded from a crash (%s)', (message) => {
    expect(slotOnError(new TypeError(message))).toEqual({ failed: { load: true, detail: message } })
  })

  it('reports any other error by its message, and something readable for a non-Error throw', () => {
    expect(slotOnError(new RangeError('Invalid typed array length: -3'))).toEqual({ failed: { load: false, detail: 'Invalid typed array length: -3' } })
    expect(slotOnError('boom')).toEqual({ failed: { load: false, detail: 'boom' } })
    expect(slotOnError({ code: 7 })).toEqual({ failed: { load: false, detail: 'unexpected error' } })
    expect(slotOnError(new Error(''))).toEqual({ failed: { load: false, detail: 'Error' } })
    expect(slotOnError(null)).toEqual({ failed: { load: false, detail: 'unexpected error' } })
  })

  it('keeps a failure while the view stays the same and clears it when the view (or the attempt) changes', () => {
    const failed = { key: 'map:0', failed: { load: false, detail: 'x' } }
    expect(slotOnProps(failed, 'map:0')).toBeNull() // React: null = no change
    expect(slotOnProps(failed, 'camera:0')).toEqual({ key: 'camera:0', failed: null })
    expect(slotOnProps(failed, 'map:1')).toEqual({ key: 'map:1', failed: null })
    expect(slotOnProps(slotState('camera:0'), 'camera:0')).toBeNull()
  })
})
