import { describe, expect, it } from 'vitest'
import { MAX_IN_FLIGHT, RttTracker, frameDue, ingestUrl, parseAck, shouldSend, sizeVerdict, verdictText } from './sender'

describe('phone sender', () => {
  it('sends only a native 640x480 frame', () => {
    expect(sizeVerdict(640, 480)).toEqual({ kind: 'ok' })
    expect(sizeVerdict(480, 640)).toEqual({ kind: 'portrait' })
    expect(sizeVerdict(1280, 720)).toEqual({ kind: 'wrong', w: 1280, h: 720 })
    expect(sizeVerdict(1280, 960)).toEqual({ kind: 'wrong', w: 1280, h: 960 }) // 4:3 but not the calibrated size
    expect(sizeVerdict(0, 0)).toEqual({ kind: 'none' })
    expect(verdictText(sizeVerdict(480, 640))).toMatch(/landscape/)
    expect(verdictText(sizeVerdict(1280, 720))).toMatch(/1280x720/)
  })

  it('skips a frame while one is still buffered or too many are unanswered', () => {
    expect(shouldSend(0, 0)).toBe(true)
    expect(shouldSend(1, 0)).toBe(false)
    expect(shouldSend(0, MAX_IN_FLIGHT)).toBe(false)
    expect(shouldSend(0, MAX_IN_FLIGHT - 1)).toBe(true)
  })

  it('paces at the target rate', () => {
    expect(frameDue(0, 66.7, 15)).toBe(true)
    expect(frameDue(0, 40, 15)).toBe(false)
    expect(frameDue(0, 55, 15)).toBe(true) // slack for a camera landing just short of the period
  })

  it('reaches the ingest on the same origin', () => {
    expect(ingestUrl({ protocol: 'https:', host: 'x.trycloudflare.com' })).toBe('wss://x.trycloudflare.com/phone/ingest')
    expect(ingestUrl({ protocol: 'http:', host: 'localhost:5173' })).toBe('ws://localhost:5173/phone/ingest')
  })

  it('measures round trips from acks in send order and reports the median', () => {
    const r = new RttTracker(3)
    expect(r.median()).toBeNull()
    r.sent(0)
    r.sent(10)
    expect(r.inFlight).toBe(2)
    expect(r.acked(100)).toBe(100)
    expect(r.acked(60)).toBe(50)
    expect(r.acked(70)).toBeNull() // an ack without a frame
    expect(r.median()).toBe(75)
    for (const t of [0, 0, 0]) r.sent(t)
    for (const t of [1, 2, 3]) r.acked(t)
    expect(r.median()).toBe(2) // only the last `keep` samples
  })

  it('parses only well-formed acks', () => {
    expect(parseAck('{"seq":1,"ok":true,"w":640,"h":480}')).toEqual({ seq: 1, ok: true, w: 640, h: 480 })
    expect(parseAck('{"ok":true}')).toBeNull()
    expect(parseAck('nope')).toBeNull()
    expect(parseAck(new ArrayBuffer(2))).toBeNull()
  })
})
