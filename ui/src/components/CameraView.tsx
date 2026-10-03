import type { CameraSource } from '../source/useCameraSource'
import { analyzerFor } from '../source/useCameraSource'
import type { Layers } from '../types'
import { Viewport } from './Viewport'

const SOURCE_LABEL = { ros2: 'robot camera', camera: 'browser camera', upload: 'photo', recording: 'recorded video' } as const
const LAYER_KEYS: (keyof Layers)[] = ['image', 'mask', 'depth', 'path']

// The main page: the camera with Dev 1's mask / depth / path overlay. Display only: nothing here commands motion.
export function CameraView({ cam }: { cam: CameraSource }) {
  const streaming = cam.status === 'live'
  const ok = cam.source === 'ros2' ? cam.rosConnected && streaming : streaming || cam.status === 'still'
  const toggle = (k: keyof Layers) => cam.setLayers({ ...cam.layers, [k]: !cam.layers[k] })

  return (
    <section className="camera">
      <header className="livefeed-bar">
        <span className={`dot ${ok ? 'ok' : 'bad'}`} />
        <span>
          {SOURCE_LABEL[cam.source]}
          {streaming && ` · ${cam.fps.toFixed(1)} fps`}
          {cam.note && ` · ${cam.note}`}
        </span>
        {!analyzerFor(cam.source).available && (
          <span className="flag" title="No perception backend for this source: frames are shown without a mask, depth or path">
            NO ANALYZER
          </span>
        )}
        <span className="livefeed-layers">
          {LAYER_KEYS.map((k) => (
            <button key={k} type="button" className={cam.layers[k] ? 'on' : ''} aria-pressed={cam.layers[k]} onClick={() => toggle(k)}>
              {k}
            </button>
          ))}
        </span>
      </header>
      <Viewport frame={cam.frame} analysis={cam.analysis} layers={cam.layers} freshness={cam.freshness} />
    </section>
  )
}
