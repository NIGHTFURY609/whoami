// rosbridge v2 client (JSON over WebSocket) for the console camera view (architecture.md §4 live_cam / bag
// profiles). Reads only the camera and Dev 1's Perception Port outputs. It never publishes anything: no
// goals, no e-stop, no /cmd_vel* (operator commands live in the Dev 5 operator console, ui/).
import type { Intrinsics } from '../types'
import { parseDepth, parseMask, type RosDepth, type RosMask } from './rosimage'

export interface RosOptions {
  url: string
  imageTopic: string
  infoTopic: string
}

export interface CameraCalibration {
  K: Intrinsics
  width: number
  height: number
  frameId: string
}

export interface PerceptionHealthPatch {
  degraded?: boolean // /ugv/perception_degraded
  valid?: boolean // /segmentation/port_meta valid flag
}

export interface RosCallbacks {
  onFrame: (bitmap: ImageBitmap, stampMs: number, frameId: string) => void
  onInfo: (calib: CameraCalibration) => void
  onStatus: (text: string, ok: boolean) => void
  onHealth?: (patch: PerceptionHealthPatch) => void
  // Dev 1's Perception Port outputs, decoded and validated (see rosimage.ts)
  onMask?: (mask: RosMask) => void
  onDepth?: (depth: RosDepth) => void
}

const num = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v)

function isValidK(k: unknown): k is number[] {
  if (!Array.isArray(k) || k.length !== 9) return false
  for (let i = 0; i < 9; i++) if (!num(k[i])) return false
  // zero / fake intrinsics must never be accepted as "measured"
  return k[0] > 0 && k[4] > 0 && k[2] > 0 && k[5] > 0
}

// `paired`: every camera frame, mask and depth image is wanted (a replayed video shown frame by frame with its own
// overlay, framesync.ts), so nothing is throttled; the replay's own pace (one frame per perception cycle) bounds it.
export function connectRos(opts: RosOptions, cb: RosCallbacks, paired = false): () => void {
  const throttle = (ms: number) => ({ throttle_rate: paired ? 0 : ms, queue_length: paired ? 5 : 1 })
  let ws: WebSocket | null = null
  let closed = false
  let reconnectTimer: number | undefined
  let reconnectAttempts = 0

  let latestProcessedStamp = -Infinity
  let latestDispatchedSeq = 0
  let msgSeq = 0

  const send = (o: unknown) => { if (ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify(o)) }

  const scheduleReconnect = () => {
    if (closed || reconnectTimer !== undefined) return
    reconnectAttempts++
    const delay = Math.min(1000 * Math.pow(1.5, reconnectAttempts - 1), 10000)
    cb.onStatus(`reconnecting to ${opts.url} in ${(delay / 1000).toFixed(1)}s...`, false)
    reconnectTimer = window.setTimeout(() => {
      reconnectTimer = undefined
      if (!closed) setupWs()
    }, delay)
  }

  const sub = (topic: string, type: string, extra: object = {}) =>
    send({ op: 'subscribe', topic, type, ...extra })

  const handlePublish = async (m: any) => {
    const msg = m.msg
    if (!msg || typeof msg !== 'object') return
    switch (m.topic) {
      case '/ugv/perception_degraded':
        if (typeof msg.data === 'boolean') cb.onHealth?.({ degraded: msg.data })
        return
      case '/segmentation/port_meta': {
        const d = msg.data
        if (Array.isArray(d) && d.length >= 3 && d.slice(0, 3).every(num)) cb.onHealth?.({ valid: d[0] >= 0.5 })
        return
      }
      case '/segmentation/mask': {
        const mask = parseMask(msg, Date.now())
        if (mask) cb.onMask?.(mask)
        return
      }
      case '/perception/depth/image': {
        const depth = parseDepth(msg, Date.now())
        if (depth) cb.onDepth?.(depth)
        return
      }
    }

    if (m.topic === opts.infoTopic) {
      const width = num(msg.width) ? msg.width : 0
      const height = num(msg.height) ? msg.height : 0
      if (width <= 0 || height <= 0 || !isValidK(msg.k)) return
      const k = msg.k as number[]
      cb.onInfo({
        K: { fx: k[0], cx: k[2], fy: k[4], cy: k[5] },
        width, height,
        frameId: typeof msg.header?.frame_id === 'string' ? msg.header.frame_id : '',
      })
    } else if (m.topic === opts.imageTopic) {
      const stamp = msg.header?.stamp
      if (!stamp || !num(stamp.sec) || !num(stamp.nanosec)) return
      const stampMs = stamp.sec * 1000 + stamp.nanosec / 1e6
      const frameId = typeof msg.header.frame_id === 'string' ? msg.header.frame_id : ''
      const data = msg.data
      const format = typeof msg.format === 'string' ? msg.format : ''
      if (typeof data !== 'string' || !data) return
      if (stampMs <= latestProcessedStamp) return // older than the newest frame already shown

      const seq = ++msgSeq
      try {
        const binary = atob(data)
        const bytes = new Uint8Array(binary.length)
        for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
        const blob = new Blob([bytes], { type: format.toLowerCase().includes('png') ? 'image/png' : 'image/jpeg' })
        const bitmap = await createImageBitmap(blob)
        if (stampMs <= latestProcessedStamp || seq < latestDispatchedSeq || closed) {
          bitmap.close()
          return
        }
        latestProcessedStamp = stampMs
        latestDispatchedSeq = seq
        cb.onFrame(bitmap, stampMs, frameId)
      } catch (err) {
        console.warn('Failed to decode ROS 2 frame:', err)
      }
    }
  }

  const setupWs = () => {
    if (closed) return
    try {
      ws = new WebSocket(opts.url)
    } catch {
      cb.onStatus(`cannot reach ${opts.url}`, false)
      scheduleReconnect()
      return
    }

    ws.onopen = () => {
      reconnectAttempts = 0
      cb.onStatus(`ROS 2 connected ${opts.url}`, true)
      sub(opts.imageTopic, 'sensor_msgs/msg/CompressedImage', throttle(200))
      sub(opts.infoTopic, 'sensor_msgs/msg/CameraInfo')
      // Dev 1 Perception Port: degraded flag + the mask / depth the analyzer is built from
      sub('/ugv/perception_degraded', 'std_msgs/msg/Bool')
      sub('/segmentation/port_meta', 'std_msgs/msg/Float64MultiArray')
      sub('/segmentation/mask', 'sensor_msgs/msg/Image', throttle(200))
      sub('/perception/depth/image', 'sensor_msgs/msg/Image', throttle(500))
    }

    ws.onerror = () => cb.onStatus(`cannot reach ${opts.url}`, false)

    ws.onclose = () => {
      if (!closed) {
        cb.onStatus('ROS 2 disconnected', false)
        scheduleReconnect()
      }
    }

    ws.onmessage = async (ev) => {
      let m: any
      try {
        m = typeof ev.data === 'string' ? JSON.parse(ev.data) : JSON.parse(new TextDecoder().decode(ev.data))
      } catch {
        return
      }
      if (m && typeof m === 'object' && m.op === 'publish') await handlePublish(m)
    }
  }

  setupWs()

  return () => {
    closed = true
    if (reconnectTimer !== undefined) {
      window.clearTimeout(reconnectTimer)
      reconnectTimer = undefined
    }
    if (ws) {
      ws.onclose = null
      ws.onerror = null
      ws.onmessage = null
      ws.onopen = null
      ws.close()
      ws = null
    }
  }
}
