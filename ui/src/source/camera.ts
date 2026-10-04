import { PHONE_VIDEO } from '../phone/sender'

export interface Camera {
  video: HTMLVideoElement
  stop: () => void
}

export async function openCamera(): Promise<Camera> {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: PHONE_VIDEO, // 640x480 like the phone camera page (phone.html), the size the phone calibration is for
    audio: false,
  })
  const video = document.createElement('video')
  video.srcObject = stream
  video.muted = true
  video.playsInline = true
  await video.play()
  return { video, stop: () => stream.getTracks().forEach((t) => t.stop()) }
}
