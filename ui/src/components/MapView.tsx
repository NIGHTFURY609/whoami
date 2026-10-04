import { useCallback, useEffect, useRef, useState } from 'react'
import { depthToRgba } from '../map/geometry'
import {
  LIVE_MODES, enabledLayers, parseToggles, toggled, withLiveMode, type LiveMode, type MapToggles, type ToggleKey,
} from '../map/mapToggles'
import { MapScene } from '../map/scene'
import { useMapData } from '../map/useMapData'
import { LAYERS, type MapStatus, type Telemetry } from '../source/api'
import { GH, GW } from '../types'
import { mapBanner } from './mapStats'

// The map view, as in the owner's reference picture: the live depth scan as a height heat map (merged into a rolling
// terrain, or the current scan only), the cost grid with its free cells painted as the pathway, the travelled path as
// a ribbon the car's width and the car, with the depth image and the camera image stacked on its left edge. Display only: the 3D layers come from the gateway's map endpoints (GETs) and
// the telemetry stream's pose; the two image panels show the camera feed the console already receives for the camera
// view (passed in by App), so no image travels twice. Nothing here commands anything.

const TOGGLES_KEY = 'ugv.console.map.layers'
const CAMERA_PANEL_MAX_W = 480 // the camera panel is a thumbnail; no need to keep a full-size copy
const DEPTH_PANEL_MAX_M = 8 // the depth panel's grey ramp: white at 0 m, black from here on

// Short labels keep the bar on one row from a 1280 px window up (the view area is then about 580 px wide); the full
// name is the accessible name and the tooltip.
const LAYER_BUTTONS: { key: ToggleKey; label: string; name: string; title: string }[] = [
  { key: 'live', label: 'live', name: 'Live heat map', title: 'Live heat map: the depth scan coloured by height, blue low to red high' },
  { key: 'trajectory', label: 'path', name: 'Travelled path', title: 'Travelled path: a ribbon the car\'s width behind the car' },
  { key: 'grid', label: 'cost', name: 'Cost grid', title: 'Cost grid: free cells painted as the pathway (cells never seen count as free), obstacles and their halo' },
  { key: 'images', label: 'img', name: 'Image panels', title: 'Image panels: depth and camera' },
]
const LIVE_MODE_TITLES: Record<LiveMode, string> = {
  terrain: 'Terrain: every scan merged into a 30 m height map around the car, so ground already seen stays',
  scan: 'Scan: the current scan only, lighter and never old',
}
const count = (n: number) => n.toLocaleString('en-US')

const loadToggles = (): MapToggles => {
  try {
    return parseToggles(localStorage.getItem(TOGGLES_KEY))
  } catch {
    return parseToggles(null)
  }
}

const remember = (t: MapToggles) => {
  try { localStorage.setItem(TOGGLES_KEY, JSON.stringify(t)) } catch { /* not persisted */ }
}

// A frame from an earlier gateway process (its epoch is not the current status's) is not shown: after a restart the
// hook keeps old frames until each layer is fetched again, and a layer still at seq 0 never is.
function ofEpoch<T extends { epoch: number }>(frame: T | null, status: MapStatus | null): T | null {
  return frame && status && frame.epoch === status.epoch ? frame : null
}

// Polling pauses while the tab is hidden: nobody is looking, and the gateway drops its heavy subscriptions.
function usePageVisible(): boolean {
  const [visible, setVisible] = useState(() => !document.hidden)
  useEffect(() => {
    const onChange = () => setVisible(!document.hidden)
    document.addEventListener('visibilitychange', onChange)
    return () => document.removeEventListener('visibilitychange', onChange)
  }, [])
  return visible
}

interface Props {
  telemetry: Telemetry | null
  live: boolean // telemetry stream fresh
  image: ImageBitmap | null // the robot camera's latest frame, null when the console is not on the robot camera
  depth: Float32Array | null // its current depth, GW x GH metres (NaN = hole), null when there is none or it is stale
}

export function MapView({ telemetry, live, image, depth }: Props) {
  const [toggles, setToggles] = useState<MapToggles>(loadToggles)
  // Saved only once the operator changed something, so a later change of the defaults still reaches anyone who never
  // touched a toggle.
  const touched = useRef(false)
  useEffect(() => {
    if (touched.current) remember(toggles)
  }, [toggles])
  const pageVisible = usePageVisible()
  const data = useMapData(pageVisible, enabledLayers(toggles)) // a toggled-off layer is not fetched
  const [scene, setScene] = useState<MapScene | null>(null)
  const [glFailed, setGlFailed] = useState(false)

  // The scene lives exactly as long as its host element: one WebGL context per mount, created when React attaches the
  // element and disposed (context released) when it detaches. StrictMode attaches, detaches and attaches again in
  // development; the second attach gets a fresh scene after the first one is disposed. Stable (no deps), so a
  // re-render never re-attaches.
  const attachScene = useCallback((host: HTMLDivElement | null) => {
    if (!host) return
    let s: MapScene
    try {
      s = new MapScene(host)
    } catch {
      setGlFailed(true) // no WebGL: say so in the view instead of throwing
      return
    }
    s.resize(host.clientWidth, host.clientHeight)
    const ro = new ResizeObserver(() => s.resize(host.clientWidth, host.clientHeight))
    ro.observe(host)
    setScene(s)
    return () => {
      ro.disconnect()
      s.dispose()
      setScene(null)
    }
  }, [])

  const status = data.status
  const noMap = !status || LAYERS.every((l) => status.seq[l] === 0)
  const pose = telemetry?.pose ?? null
  const liveCloud = ofEpoch(data.live, status)
  const trajectory = ofEpoch(data.trajectory, status)
  const grid = ofEpoch(data.grid, status)

  // The robot is hidden while telemetry is not live: a pose from a dead stream is not where the robot is. The pose
  // goes first: the first layer to arrive frames the camera on it within the same commit.
  const shownPose = live ? pose : null
  useEffect(() => { scene?.setPose(shownPose) }, [scene, shownPose])
  const { live: showLive, liveMode, trajectory: showTrajectory, grid: showGrid } = toggles
  useEffect(() => {
    scene?.setLayers({ live: showLive, trajectory: showTrajectory, grid: showGrid })
  }, [scene, showLive, showTrajectory, showGrid])
  // The mode goes before the scan, so a mount in the scan mode never builds a terrain first.
  useEffect(() => { scene?.setLiveMode(liveMode) }, [scene, liveMode])
  useEffect(() => { scene?.setLive(liveCloud) }, [scene, liveCloud])
  const [terrainCells, setTerrainCells] = useState<number | null>(null)
  useEffect(() => {
    if (!scene) return
    scene.onInfo = (info) => setTerrainCells(info.terrainCells)
    return () => { scene.onInfo = null }
  }, [scene])
  useEffect(() => { scene?.setTrajectory(trajectory) }, [scene, trajectory])
  useEffect(() => { scene?.setGrid(grid) }, [scene, grid])

  // Functional updates: two clicks inside one render both count.
  const flip = (key: ToggleKey) => {
    touched.current = true
    setToggles((t) => toggled(t, key))
  }
  const chooseLiveMode = (mode: LiveMode) => {
    touched.current = true
    setToggles((t) => withLiveMode(t, mode))
  }

  const banner = mapBanner({ noMap, statusStale: data.stale, telemetryLost: !live, stats: status?.stats })
  const ok = !noMap && banner === null // the dot follows the banner: green only when nothing says the feed is wrong
  const slamMode = typeof status?.stats.mode === 'string' ? status.stats.mode : null
  const depthPanel = toggles.images ? depth : null // a panel shows only with data and with the images toggle on
  const cameraPanel = toggles.images ? image : null
  // Counts sit in the stage, not the bar, so the bar never cuts a number short.
  const info = [
    showLive && liveMode === 'terrain' && terrainCells !== null && terrainCells > 0 && `${count(terrainCells)} terrain cells`,
    showLive && liveMode === 'scan' && liveCloud && `${count(liveCloud.count)} live pts`,
    showTrajectory && trajectory && `${trajectory.lengthM.toFixed(1)} m path`,
  ].filter(Boolean).join(' · ')

  return (
    <section className="mapview">
      <header className="livefeed-bar">
        <span className={`dot ${ok ? 'ok' : 'bad'}`} />
        <span>
          3d map
          {slamMode && ` · ${slamMode}`}
        </span>
        <span className="livefeed-layers">
          {LAYER_BUTTONS.map(({ key, label, name, title }) => (
            <span key={key} className="livefeed-group">
              <button type="button" title={title} aria-label={name} className={toggles[key] ? 'on' : ''}
                aria-pressed={toggles[key]} onClick={() => flip(key)}>
                {label}
              </button>
              {key === 'live' && (
                <span className="livefeed-seg" role="group" aria-label="Live mode">
                  {LIVE_MODES.map((mode) => (
                    <button key={mode} type="button" title={LIVE_MODE_TITLES[mode]} disabled={!showLive}
                      className={liveMode === mode ? 'on' : ''} aria-pressed={liveMode === mode}
                      onClick={() => chooseLiveMode(mode)}>
                      {mode}
                    </button>
                  ))}
                </span>
              )}
            </span>
          ))}
          <button type="button" aria-label="Reset view" title="Reset view: back behind the robot" disabled={!scene}
            onClick={() => scene?.resetView()}>
            reset
          </button>
        </span>
      </header>
      <div className="mapview-stage">
        <div ref={attachScene} className="mapview-gl" role="img"
          aria-label="3D map around the robot: drag to orbit, right-drag to pan, scroll to zoom" />
        {!noMap && info && <div className="mapview-info">{info}</div>}
        {glFailed ? (
          <div className="nosignal">
            <b>NO 3D VIEW</b>
            <span>WebGL could not start in this browser</span>
          </div>
        ) : noMap ? (
          <div className="nosignal">
            <b>NO MAP YET</b>
            <span>{data.stale ? 'waiting for the gateway' : 'no map layer has data yet'}</span>
          </div>
        ) : null}
        {banner && <div className="banner stale">{banner}</div>}
        {(depthPanel || cameraPanel) && (
          <div className="mapview-insets">
            {depthPanel && <DepthPanel depthM={depthPanel} />}
            {cameraPanel && <CameraPanel frame={cameraPanel} />}
          </div>
        )}
      </div>
    </section>
  )
}

// Depth, grey: near is bright, holes show the panel behind them. GW x GH, the camera view's analysis grid.
function DepthPanel({ depthM }: { depthM: Float32Array }) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const c = ref.current
    if (!c) return
    if (c.width !== GW) c.width = GW
    if (c.height !== GH) c.height = GH
    // depthToRgba allocates a plain ArrayBuffer, which is what ImageData wants; the cast only says so.
    const rgba = depthToRgba(depthM, GW, GH, DEPTH_PANEL_MAX_M) as Uint8ClampedArray<ArrayBuffer>
    c.getContext('2d')?.putImageData(new ImageData(rgba, GW, GH), 0, 0)
  }, [depthM])
  return (
    <figure className="mapview-inset" title="Depth image: near is bright">
      <figcaption>depth</figcaption>
      <canvas ref={ref} role="img" aria-label="Depth image from the camera, near is bright" />
    </figure>
  )
}

function CameraPanel({ frame }: { frame: ImageBitmap }) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const c = ref.current
    if (!c || frame.width === 0 || frame.height === 0) return // a released bitmap reads 0 x 0
    const w = Math.min(frame.width, CAMERA_PANEL_MAX_W)
    const h = Math.round((w * frame.height) / frame.width)
    if (c.width !== w) c.width = w
    if (c.height !== h) c.height = h
    try {
      c.getContext('2d')?.drawImage(frame, 0, 0, w, h)
    } catch {
      // released between render and effect: the next frame draws
    }
  }, [frame])
  return (
    <figure className="mapview-inset">
      <figcaption>camera</figcaption>
      <canvas ref={ref} role="img" aria-label="Latest camera image" />
    </figure>
  )
}
