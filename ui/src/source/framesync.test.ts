import { describe, expect, it } from 'vitest'
import { FrameSync } from './framesync'

const make = (depthWaitMs = 600, maxFrames = 30) => {
  const disposed: string[] = []
  const sync = new FrameSync<string>((f) => disposed.push(f), depthWaitMs, maxFrames)
  return { sync, disposed }
}

describe('FrameSync', () => {
  it('holds a frame until its mask and depth arrive', () => {
    const { sync } = make()
    sync.addFrame(1000.5, 'f1')
    expect(sync.take(0)).toBeNull()
    sync.noteMask(1000.5, 0)
    expect(sync.take(10)).toBeNull() // depth not here yet
    sync.noteDepth(1000.5)
    expect(sync.take(20)).toEqual({ stampMs: 1000.5, frame: 'f1' })
    expect(sync.pending).toBe(0)
  })

  it('releases on the mask alone once the depth wait has passed', () => {
    const { sync } = make(600)
    sync.addFrame(2000, 'f')
    sync.noteMask(2000, 100)
    expect(sync.take(699)).toBeNull()
    expect(sync.take(700)?.frame).toBe('f')
  })

  it('ignores overlays of other frames', () => {
    const { sync } = make()
    sync.addFrame(3000, 'f')
    sync.noteMask(2999, 0)
    sync.noteDepth(2999)
    expect(sync.take(10_000)).toBeNull()
  })

  it('releases the newest ready frame and drops older ones', () => {
    const { sync, disposed } = make()
    sync.addFrame(1, 'a')
    sync.addFrame(2, 'b') // its mask never comes
    sync.addFrame(3, 'c')
    sync.addFrame(4, 'd') // newer than the overlay: stays pending
    sync.noteMask(3, 0)
    sync.noteDepth(3)
    expect(sync.take(0)?.frame).toBe('c')
    expect(disposed).toEqual(['a', 'b'])
    expect(sync.pending).toBe(1)
  })

  it('caps the frames it holds and disposes them on clear', () => {
    const { sync, disposed } = make(600, 2)
    sync.addFrame(1, 'a')
    sync.addFrame(2, 'b')
    sync.addFrame(3, 'c')
    expect(disposed).toEqual(['a'])
    sync.clear()
    expect(disposed).toEqual(['a', 'b', 'c'])
    expect(sync.pending).toBe(0)
  })
})
