import { useEffect, useState } from 'react'
import type { RecordedPlayback } from '../source/recording'
import type { VideoCommand, VideoStatus } from '../source/videobridge'

const STATE_LABEL = {
  ready: 'READY',
  playing: 'PLAYING',
  buffering: 'BUFFERING',
  paused: 'PAUSED',
  ended: 'ENDED',
} as const

const clock = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`

function Progress({ at, total, fps }: { at: number; total: number; fps: number }) {
  const pct = total > 0 ? Math.min(100, (100 * at) / total) : 0
  return (
    <>
      <span className="dim-inline">
        {clock(at / fps)}{total > 0 && ` / ${clock(total / fps)}`} · frame {at}{total > 0 && ` of ${total}`}
      </span>
      <div className="video-progress" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(pct)}>
        <span style={{ width: `${pct}%` }} />
      </div>
    </>
  )
}

interface Props {
  status: VideoStatus
  error: string
  onCommand: (cmd: VideoCommand) => void // the live replay through the stack
  recorded: RecordedPlayback
  onPlayLive: () => void
  onPlayRecorded: () => void
}

// Recorded-video replay (eval only). Live: the onboard camera recording goes through the whole stack in place of a
// live camera (start, pause, replay; optionally waiting for perception on every frame), and the overlays it makes
// are saved. Recorded: those saved overlays played back with the video, any time, without the stack.
export function VideoPanel({ status: s, error, onCommand, recorded: r, onPlayLive, onPlayRecorded }: Props) {
  const running = s.state === 'playing' || s.state === 'buffering'
  const syncActive = s.sync && s.perception
  const c = s.cache
  const [armed, setArmed] = useState(false) // deleting the recording takes a second click
  useEffect(() => {
    if (!armed) return
    const t = window.setTimeout(() => setArmed(false), 4000)
    return () => window.clearTimeout(t)
  }, [armed])
  const recordedState = r.buffering ? 'buffering' : r.playing ? 'playing' : r.index >= r.total && r.total > 0 ? 'ended' : r.index > 0 ? 'paused' : 'ready'

  return (
    <section className="section">
      <h3>Video replay</h3>
      <div className="stack">
        <h4 className="video-sub">Through the stack</h4>
        <div className="video-state">
          <span className={`video-badge ${s.state}`}>{STATE_LABEL[s.state]}</span>
        </div>
        <Progress at={s.frame} total={s.frames} fps={s.fps} />
        <div className="video-buttons">
          <button type="button" className={`btn ${running ? '' : 'primary'}`} onClick={() => (running ? onCommand('pause') : onPlayLive())}>
            {running ? 'PAUSE' : s.state === 'ended' ? 'PLAY AGAIN' : 'PLAY'}
          </button>
          <button type="button" className="btn" onClick={() => { onPlayLive(); onCommand('replay') }} disabled={s.state === 'ready'}>
            REPLAY
          </button>
        </div>
        <label className="video-sync">
          <input type="checkbox" checked={s.sync} onChange={(e) => onCommand(e.target.checked ? 'sync-on' : 'sync-off')} />
          Wait for perception (buffer each frame until its overlay is ready)
        </label>
        <p className="dim">
          {s.sync && !s.perception
            ? 'The bridge cannot see perception yet: playing in real time, nothing saved.'
            : syncActive
              ? 'Every frame is analysed and its overlay saved; playback slows to the perception rate.'
              : 'Real time, like a live camera: frames perception cannot keep up with are skipped, nothing saved.'}
          {s.state === 'buffering' && ' Waiting for the overlay of this frame.'}
          {s.note && ` ${s.note}.`}
          {error && ` ${error}`}
        </p>

        {c && (
          <>
            <h4 className="video-sub">Recorded overlays</h4>
            <span className="dim-inline">
              {c.masks} of {c.total || '?'} frames saved{c.complete ? ' · complete' : ''}
              {c.depths < c.masks && ` · ${c.depths} with depth`}
            </span>
            {r.available ? (
              <>
                <div className="video-state">
                  <span className={`video-badge ${recordedState}`}>{STATE_LABEL[recordedState]}</span>
                </div>
                <Progress at={r.index} total={r.total} fps={s.fps} />
                <div className="video-buttons">
                  <button type="button" className={`btn ${r.playing ? '' : 'primary'}`} onClick={() => (r.playing ? r.pause() : onPlayRecorded())}>
                    {r.playing ? 'PAUSE' : 'PLAY RECORDED'}
                  </button>
                  <button type="button" className="btn" onClick={() => { onPlayRecorded(); r.replay() }} disabled={r.index === 0 && !r.playing}>
                    REPLAY
                  </button>
                </div>
                <p className="dim">
                  Plays the saved overlays with the video at its own speed, without the stack.
                  {!c.complete && ' Frames not saved yet show without an overlay; play it through the stack again to fill them.'}
                  {r.buffering && ' Loading frames.'}
                </p>
              </>
            ) : (
              <p className="dim">Nothing saved yet: play the video through the stack with "wait for perception" on.</p>
            )}
            <button
              type="button"
              className="btn danger"
              disabled={c.frames === 0 || r.playing}
              onClick={() => {
                if (armed) onCommand('clear-recording')
                setArmed(!armed)
              }}
            >
              {armed ? 'CLICK AGAIN TO DELETE' : 'DELETE RECORDING'}
            </button>
          </>
        )}
      </div>
    </section>
  )
}
