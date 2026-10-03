import { useCallback, useEffect, useRef, useState } from 'react'
import { unavailableAnalyzer, type Analyzer } from '../analysis/analyzer'
import { useFreshness } from '../analysis/freshness'
import { RosPerception } from '../analysis/ros-analyzer'
import type { RobotPose } from '../analysis/pathhold'
import { openCamera } from './camera'
import { FrameSync } from './framesync'
import type { RecordedFrame } from './recording'
import { connectRos, type CameraCalibration, type RosOptions } from './rosbridge'
import {
  assumedIntrinsics, type Analysis, type FrameMeta, type Intrinsics, type Layers, type SourceKind, type Status,
} from '../types'

const FRAME_INTERVAL_MS = 250

// ROS 2 frames are analysed from Dev 1's Perception Port outputs received over rosbridge. Uploads and the browser
// camera have no perception backend, so they stay unavailable rather than faked.
const rosPerception = new RosPerception()
// A recorded video plays Dev 1's saved mask and depth through the same analyzer as the live robot camera.
const recordedPerception = new RosPerception()
export const analyzerFor = (src: SourceKind): Analyzer =>
  src === 'ros2' ? rosPerception : src === 'recording' ? recordedPerception : unavailableAnalyzer
let recordedStamp = 0 // recorded frames get fresh, increasing stamps: the analyzer never takes a replay for stale

// The camera path holds still until this pose moves. Pass null when the gateway pose is not live.
export function noteRobotPose(pose: RobotPose | null): void {
  rosPerception.setPose(pose)
}

// CameraInfo K scaled to the received image. null when the frame isn't the calibrated camera's.
function scaledK(calib: CameraCalibration | null, frameId: string, w: number, h: number): Intrinsics | null {
  if (!calib) return null
  const norm = (s: string) => s.replace(/^\//, '')
  if (norm(calib.frameId) && norm(calib.frameId) !== norm(frameId)) return null
  if (!(calib.width > 0 && calib.height > 0)) return calib.K
  const sx = w / calib.width
  const sy = h / calib.height
  return { fx: calib.K.fx * sx, fy: calib.K.fy * sy, cx: calib.K.cx * sx, cy: calib.K.cy * sy }
}

// The camera view's state: which source feeds it (robot camera over rosbridge by default, the browser camera or
// an uploaded photo), the latest frame and its Perception Port analysis. Read only: nothing here publishes.
export function useCameraSource() {
  const [source, setSource] = useState<SourceKind>('ros2')
  const [frame, setFrame] = useState<ImageBitmap | null>(null)
  const [analysis, setAnalysis] = useState<Analysis | null>(null)
  const [layers, setLayers] = useState<Layers>({ image: true, mask: true, depth: false, path: true })
  const [status, setStatus] = useState<Status>('idle')
  const [note, setNote] = useState('')
  const [fps, setFps] = useState(0)
  const [live, setLive] = useState(false)
  const [rosOn, setRosOn] = useState(true)
  const [rosConnected, setRosConnected] = useState(false)
  const [pairFrames, setPairFrames] = useState(false) // show each robot frame with its own overlay (video replay)
  const [rosCfg, setRosCfg] = useState<RosOptions>({
    url: `ws://${window.location.hostname || 'localhost'}:9090`,
    // Dev 5's camera driver publishes a rate-limited JPEG stream for web UIs on /image_raw/compressed, with
    // CameraInfo on /camera_info (ugv_bringup README); /camera/image_raw itself is raw rgb8.
    imageTopic: '/image_raw/compressed',
    infoTopic: '/camera_info',
  })
  const lastFrameAt = useRef(0)
  const rosInfo = useRef<CameraCalibration | null>(null)
  const ingestSeq = useRef(0)
  const freshness = useFreshness(analysis)

  const ingest = useCallback(async (bmp: ImageBitmap, src: SourceKind, stamp: number, frameId: string, streaming: boolean, K?: Intrinsics | null) => {
    const meta: FrameMeta = {
      source: src, frameId, stamp, receivedAt: Date.now(), width: bmp.width, height: bmp.height,
      K: K ?? assumedIntrinsics(bmp.width, bmp.height), kAssumed: !K, streaming,
    }
    const seq = ++ingestSeq.current
    let result: Analysis | null = null
    try {
      result = await analyzerFor(src).analyze(bmp, meta)
    } catch {
      result = null
    }
    if (seq !== ingestSeq.current) { // a newer frame already took over
      bmp.close()
      return
    }
    // The frame and its analysis in one render (a 30 fps recorded video re-renders the console once per frame).
    const now = performance.now()
    const dt = now - lastFrameAt.current
    lastFrameAt.current = now
    setFps((f) => (dt > 0 && dt < 2000 ? f * 0.7 + (1000 / dt) * 0.3 : 0))
    setFrame(bmp)
    setAnalysis(result)
  }, [])

  useEffect(() => () => frame?.close(), [frame])

  const afterStop = frame ? 'still' : 'idle'
  const stopLive = () => { setLive(false); setRosOn(false) }

  const pickSource = (s: SourceKind) => {
    stopLive()
    setSource(s)
    setNote('')
    setStatus(afterStop)
    if (s === 'ros2') setRosOn(true)
  }

  const upload = async (file: File) => {
    stopLive()
    setNote('')
    try {
      await ingest(await createImageBitmap(file), 'upload', Date.now(), file.name, false)
      setStatus('still')
    } catch {
      setStatus('error')
      setNote(`cannot read ${file.name}`)
    }
  }

  const setLiveCamera = (on: boolean) => {
    setLive(on)
    if (!on) setStatus(afterStop)
  }

  const takePhoto = async () => {
    if (live) return setLiveCamera(false)
    try {
      const cam = await openCamera()
      await new Promise((r) => setTimeout(r, 500)) // let exposure settle
      await ingest(await createImageBitmap(cam.video), 'camera', Date.now(), 'camera', false)
      cam.stop()
      setStatus('still')
      setNote('')
    } catch (e) {
      setStatus('error')
      setNote(`camera: ${(e as Error).message}`)
    }
  }

  // Browser camera, live.
  useEffect(() => {
    if (!live) return
    let cancelled = false
    let busy = false
    let timer: number | undefined
    let stop = () => {}
    openCamera().then((cam) => {
      if (cancelled) return cam.stop()
      stop = cam.stop
      setStatus('live')
      setNote('')
      timer = window.setInterval(async () => {
        if (busy) return
        busy = true
        try {
          await ingest(await createImageBitmap(cam.video), 'camera', Date.now(), 'camera', true)
        } finally {
          busy = false
        }
      }, FRAME_INTERVAL_MS)
    }).catch((e: Error) => {
      setLive(false)
      setStatus('error')
      setNote(`camera: ${e.message}`)
    })
    return () => {
      cancelled = true
      window.clearInterval(timer)
      stop()
    }
  }, [live, ingest])

  // Robot camera + Dev 1's Perception Port over rosbridge. Paired (a replayed video in sync mode): each frame is held
  // until its own mask and depth are here and shown with them (framesync.ts); otherwise frames show as they come.
  useEffect(() => {
    if (!rosOn) return
    type Held = { bmp: ImageBitmap; frameId: string; K: Intrinsics }
    const sync = new FrameSync<Held>((h) => h.bmp.close())
    const release = () => {
      const r = sync.take(Date.now())
      if (!r) return
      void ingest(r.frame.bmp, 'ros2', r.stampMs, r.frame.frameId, true, r.frame.K)
      setStatus('live')
    }
    const timer = pairFrames ? window.setInterval(release, 100) : undefined // the depth wait runs out on its own
    const disconnect = connectRos(rosCfg, {
      onInfo: (calib) => { rosInfo.current = calib },
      onFrame: (bmp, stamp, frameId) => {
        const K = scaledK(rosInfo.current, frameId, bmp.width, bmp.height)
        if (!K) {
          bmp.close() // no CameraInfo for this frame yet: never draw geometry with an assumed K
          setNote('waiting for CameraInfo')
          return
        }
        if (pairFrames) {
          sync.addFrame(stamp, { bmp, frameId, K })
          release()
          return
        }
        void ingest(bmp, 'ros2', stamp, frameId, true, K)
        setStatus('live')
      },
      onStatus: (text, ok) => {
        setNote(text)
        setRosConnected(ok)
        if (!ok) setStatus('error')
      },
      onHealth: (patch) => rosPerception.setHealth(patch),
      onMask: (m) => {
        rosPerception.pushMask(m)
        if (pairFrames) {
          sync.noteMask(m.stampMs, Date.now())
          release()
        }
      },
      onDepth: (d) => {
        rosPerception.pushDepth(d)
        if (pairFrames) {
          sync.noteDepth(d.stampMs)
          release()
        }
      },
    }, pairFrames)
    return () => {
      window.clearInterval(timer)
      sync.clear()
      rosPerception.reset()
      setRosConnected(false)
      disconnect()
    }
    // rosCfg is locked while connected
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rosOn, ingest, pairFrames])

  // One frame of a recorded video with its saved overlay (recording.ts). A frame perception made no mask for is
  // shown without one: the previous frame's overlay is cleared, never carried over.
  const showRecorded = useCallback((f: RecordedFrame) => {
    const now = Date.now()
    const stamp = Math.max(now, recordedStamp + 1)
    recordedStamp = stamp
    recordedPerception.reset()
    recordedPerception.setHealth({ degraded: false, valid: true })
    if (f.mask) recordedPerception.pushMask({ ...f.mask, stampMs: stamp, receivedAt: now })
    if (f.depth) recordedPerception.pushDepth({ ...f.depth, stampMs: stamp, receivedAt: now })
    void ingest(f.bitmap, 'recording', stamp, f.frameId, true, f.K)
    setStatus('live')
    setNote(f.mask ? '' : 'no overlay saved for this frame')
  }, [ingest])

  const toggleRos = (on: boolean) => {
    setRosOn(on)
    if (!on) {
      rosInfo.current = null
      setNote('')
      setStatus(afterStop)
    }
  }

  return {
    source, pickSource, frame, analysis, freshness, layers, setLayers, status, note, fps,
    live, setLiveCamera, takePhoto, upload,
    rosCfg, setRosCfg, rosOn, toggleRos, rosConnected, pairFrames, setPairFrames, showRecorded,
  }
}

export type CameraSource = ReturnType<typeof useCameraSource>
