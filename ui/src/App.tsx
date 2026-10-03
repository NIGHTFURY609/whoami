import { Component, lazy, memo, Suspense, useEffect, useRef, useState, type ReactNode } from 'react'
import { CameraView } from './components/CameraView'
import { CommandPanel } from './components/CommandPanel'
import { Inspector } from './components/Inspector'
import { SafetyBoard } from './components/SafetyBoard'
import { SourcePanel } from './components/SourcePanel'
import { StatusWidgets } from './components/StatusWidgets'
import { TopBar } from './components/TopBar'
import { VideoPanel } from './components/VideoPanel'
import {
  parseView, slotOnError, slotOnProps, slotState, type MainView, type SlotFailure, type SlotState,
} from './map/mapToggles'
import { ApiError, api, isLive, subscribeTelemetry, type Mode, type Telemetry } from './source/api'
import { noteRobotPose, useCameraSource } from './source/useCameraSource'
import { useRecordedPlayback } from './source/recording'
import { useVideoBridge, videoBridgeUrl } from './source/videobridge'

const CLOCK_MS = 250 // re-check telemetry freshness at 4 Hz so a dead stream reads NO SIGNAL promptly

// three.js is large, so the map view is its own lazily loaded chunk: the camera view's first paint does not pay for it.
const MapView = lazy(() => import('./components/MapView').then((m) => ({ default: m.MapView })))

// Panels whose props only change with telemetry: kept out of the per-frame re-render of a playing video.
const TopBarM = memo(TopBar)
const SafetyBoardM = memo(SafetyBoard)
const StatusWidgetsM = memo(StatusWidgets)
const InspectorM = memo(Inspector)

// The latest value (by identity), at most every `ms`: the Inspector's widgets cost a full re-render each, so a 30 fps recorded video
// updates them a few times a second (the live robot camera arrives at about that rate anyway).
function useThrottled<T>(value: T, ms: number): T {
  const [shown, setShown] = useState(value)
  const last = useRef(0)
  useEffect(() => {
    const wait = last.current + ms - performance.now()
    const t = window.setTimeout(() => {
      last.current = performance.now()
      setShown(value)
    }, Math.max(0, wait))
    return () => window.clearTimeout(t)
  }, [value, ms])
  return shown
}

const VIEWS: MainView[] = ['camera', 'map']
const VIEW_KEY = 'ugv.console.view'

const savedView = (): MainView => {
  try {
    return parseView(localStorage.getItem(VIEW_KEY))
  } catch {
    return 'camera'
  }
}

interface SlotProps {
  resetKey: string // a new key (another view, another attempt) clears a failure
  fallback: (failure: SlotFailure) => ReactNode
  children: ReactNode
}

// Error boundary around the main view only. A view that fails - its code cannot be downloaded (a stale chunk after a
// redeploy, a network hiccup) or it throws while rendering or in an effect (say, a malformed frame) - shows its
// failure in the view area; without this React would unmount the whole console, the e-stop and safety board with it.
// The state logic is in mapToggles.ts (slotState / slotOnError / slotOnProps), where it is tested.
class ViewBoundary extends Component<SlotProps, SlotState> {
  state: SlotState = slotState(this.props.resetKey)

  static getDerivedStateFromError(error: unknown) {
    return slotOnError(error)
  }

  static getDerivedStateFromProps(props: SlotProps, state: SlotState) {
    return slotOnProps(state, props.resetKey)
  }

  render() {
    return this.state.failed ? this.props.fallback(this.state.failed) : this.props.children
  }
}

type Message = { text: string; reasons?: string[]; error: boolean } | null

const fromError = (e: unknown): Message =>
  e instanceof ApiError
    ? { text: `${e.problem.title}${e.problem.detail ? `: ${e.problem.detail}` : ''}`, reasons: e.problem.reasons, error: true }
    : { text: String(e), error: true }

// Operator console. The main area shows one of two views, picked in the top bar: the camera view (Dev 1's mask /
// depth / path over the live image) or the 3D map view; both are display only, and only the shown one is mounted.
// Left sidebar: the operator's commands (e-stop §3.1, mapping|localize §10, map-frame goal §11) through Dev 5's
// gateway (/api/v1), and the camera source. Right sidebar: §12 health table, robot status, perception details.
export default function App() {
  const [telemetry, setTelemetry] = useState<Telemetry | null>(null)
  const [connected, setConnected] = useState(false)
  const [now, setNow] = useState(() => Date.now())
  const [estopBusy, setEstopBusy] = useState(false)
  const [message, setMessage] = useState<Message>(null)
  const [view, setView] = useState<MainView>(savedView)
  const [viewAttempt, setViewAttempt] = useState(0) // bumped by "try again" after a failed view
  const cam = useCameraSource()
  const inspectedAnalysis = useThrottled(cam.analysis, 200)
  const inspectedFreshness = useThrottled(cam.freshness, 200)
  // A replayed video in sync mode: show each robot frame together with its own overlay.
  const video = useVideoBridge()
  const pair = !!video.status?.sync && !!video.status.perception
  const { setPairFrames } = cam
  useEffect(() => setPairFrames(pair), [pair, setPairFrames])
  // ...and its saved overlays, played back without the stack.
  const recorded = useRecordedPlayback(videoBridgeUrl(), video.status?.cache ?? null, video.status?.fps ?? 30, cam.showRecorded)
  const playLive = () => {
    recorded.pause()
    if (cam.source !== 'ros2') cam.pickSource('ros2')
    void video.send('play')
  }
  const playRecorded = () => {
    if (video.status?.state === 'playing' || video.status?.state === 'buffering') void video.send('pause')
    if (cam.source !== 'recording') cam.pickSource('recording')
    recorded.play()
  }
  // The map view's image panels show the robot camera only (not the browser camera or an upload), and its depth only
  // while that analysis is current: one feed for both views, nothing fetched twice.
  const onRobot = cam.source === 'ros2'
  const robotImage = onRobot ? cam.frame : null
  const robotDepth = onRobot && cam.freshness?.ok ? cam.analysis?.depth ?? null : null

  const pickView = (v: MainView) => {
    setView(v)
    try { localStorage.setItem(VIEW_KEY, v) } catch { /* not persisted */ }
  }

  const viewFailed = (f: SlotFailure) => (
    <section className="viewfail" role="alert">
      <b>{view} view {f.load ? 'failed to load' : 'stopped'}</b>
      <span>{f.detail}</span>
      <span>The rest of the console, the e-stop included, keeps working.</span>
      <div className="viewfail-actions">
        {view === 'map' && (
          <button type="button" className="btn primary" onClick={() => pickView('camera')}>back to camera view</button>
        )}
        {f.load ? (
          <button type="button" className="btn" onClick={() => window.location.reload()}>reload page</button>
        ) : (
          <button type="button" className="btn" onClick={() => setViewAttempt((a) => a + 1)}>try again</button>
        )}
      </div>
    </section>
  )

  useEffect(() => subscribeTelemetry(setTelemetry, setConnected), [])
  useEffect(() => {
    const t = window.setInterval(() => setNow(Date.now()), CLOCK_MS)
    return () => window.clearInterval(t)
  }, [])

  const live = isLive(telemetry, now)
  const safety = live ? telemetry.safety : undefined
  const pose = live ? telemetry.pose : undefined
  useEffect(() => {
    if (pose?.available && pose.x !== null && pose.y !== null && pose.qx !== null && pose.qy !== null && pose.qz !== null && pose.qw !== null) {
      noteRobotPose({ x: pose.x, y: pose.y, qx: pose.qx, qy: pose.qy, qz: pose.qz, qw: pose.qw })
    } else {
      noteRobotPose(null)
    }
  }, [pose])
  const gateReasons = safety ? safety.watches.filter((w) => !w.ok).map((w) => `${w.name}: ${w.reason}`) : []

  const run = async (fn: () => Promise<string>) => {
    try {
      setMessage({ text: await fn(), error: false })
    } catch (e) {
      setMessage(fromError(e))
    }
  }

  const setEstop = async (asserted: boolean) => {
    setEstopBusy(true)
    await run(async () => {
      const r = await api.setEstop(asserted)
      return r.assertedByGateway ? 'E-stop asserted.' : 'E-stop released by this console.'
    })
    setEstopBusy(false)
  }

  const setMode = (mode: Mode) => run(async () => `Localization mode set to ${await api.setMode(mode)}.`)

  const sendGoal = (x: number, y: number, yaw: number) =>
    run(async () => {
      const g = await api.sendGoal(x, y, yaw)
      return `Goal ${g.id.slice(0, 8)} sent: ${g.state}.`
    })

  const cancelGoal = (id: string) => run(async () => `Goal ${id.slice(0, 8)}: ${(await api.cancelGoal(id)).state}.`)

  return (
    <div className="app">
      <TopBarM connected={connected} live={live} safetyOk={safety?.ok ?? null} />
      <nav className="viewswitch" aria-label="Main view">
        {VIEWS.map((v) => (
          <button key={v} type="button" className={view === v ? 'on' : ''} aria-pressed={view === v} onClick={() => pickView(v)}>
            {v}
          </button>
        ))}
      </nav>
      <aside className="panel source">
        <CommandPanel
          live={live}
          estopAsserted={safety?.eStop.asserted ?? false}
          estopBusy={estopBusy}
          onEstop={setEstop}
          mode={live ? telemetry.localization?.requestedMode ?? null : null}
          onMode={setMode}
          gateReasons={gateReasons}
          activeGoal={live ? telemetry.navigation?.activeGoal ?? null : null}
          onSendGoal={sendGoal}
          onCancelGoal={cancelGoal}
          message={message}
        />
        <SourcePanel cam={cam} />
        {video.status && (
          <VideoPanel
            status={video.status}
            error={video.error}
            onCommand={(c) => void video.send(c)}
            recorded={recorded}
            onPlayLive={playLive}
            onPlayRecorded={playRecorded}
          />
        )}
      </aside>
      <ViewBoundary resetKey={`${view}:${viewAttempt}`} fallback={viewFailed}>
        {view === 'camera' ? (
          <CameraView cam={cam} />
        ) : (
          <Suspense fallback={<section className="mapview" aria-busy="true" />}>
            <MapView telemetry={telemetry} live={live} image={robotImage} depth={robotDepth} />
          </Suspense>
        )}
      </ViewBoundary>
      <aside className="panel inspector">
        <SafetyBoardM safety={safety} live={live} />
        <p className="inspector-hint">drag widgets to rearrange · alt + arrows on keyboard</p>
        <StatusWidgetsM
          live={live}
          command={telemetry?.command}
          navigation={telemetry?.navigation}
          localization={telemetry?.localization}
          eStop={safety?.eStop}
          map={telemetry?.map}
        />
        <InspectorM analysis={inspectedAnalysis} freshness={inspectedFreshness} />
      </aside>
    </div>
  )
}
