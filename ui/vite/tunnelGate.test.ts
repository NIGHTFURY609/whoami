import { describe, expect, it } from 'vitest'
import { TOKEN_COOKIE, decide, isLocal, readCookie } from './tunnelGate.ts'

const T = 'tok3n-abc'
const req = (host: string, url = '/', extra: Record<string, string> = {}) => ({ url, headers: { host, ...extra } })

describe('tunnel gate', () => {
  it('without a token lets local and LAN requests through and refuses the tunnel, naming PHONE=1', () => {
    expect(decide(undefined, req('localhost:5173'))).toEqual({ kind: 'pass' })
    expect(decide('', req('192.168.1.20:5173'))).toEqual({ kind: 'pass' })
    expect(decide(undefined, req('x.trycloudflare.com'))).toEqual({ kind: 'deny', reason: 'no-token' })
    expect(decide(undefined, req('ugv.example.org', '/', { 'cf-ray': '1' }))).toEqual({ kind: 'deny', reason: 'no-token' })
  })

  it('lets the laptop itself through', () => {
    for (const host of ['localhost:5173', '127.0.0.1:5173', '[::1]:5173', 'LOCALHOST']) {
      expect(decide(T, req(host)).kind).toBe('pass')
    }
  })

  it('does not take a Cloudflare-relayed request for a local one, whatever its Host', () => {
    expect(isLocal(req('localhost:5173', '/', { 'cf-ray': '8a1b' }))).toBe(false)
    expect(decide(T, req('localhost:5173', '/', { 'cf-connecting-ip': '1.2.3.4' })).kind).toBe('deny')
  })

  it('refuses a tunnel request without the token', () => {
    expect(decide(T, req('x.trycloudflare.com', '/phone.html')).kind).toBe('deny')
    expect(decide(T, req('x.trycloudflare.com', '/api/v1/safety')).kind).toBe('deny')
  })

  it('swaps ?token= for a cookie and redirects to the clean URL', () => {
    const d = decide(T, req('x.trycloudflare.com', `/phone.html?token=${T}&fps=10`))
    expect(d.kind).toBe('grant')
    if (d.kind !== 'grant') return
    expect(d.location).toBe('/phone.html?fps=10')
    expect(d.cookie).toContain(`${TOKEN_COOKIE}=${T}`)
    expect(d.cookie).toMatch(/HttpOnly/)
    expect(d.cookie).toMatch(/Secure/)
    expect(d.cookie).toMatch(/SameSite=Lax/) // Strict is dropped on the redirect of a link opened from another app
  })

  it('refuses a wrong token in the query or the cookie', () => {
    expect(decide(T, req('x.trycloudflare.com', '/?token=nope')).kind).toBe('deny')
    expect(decide(T, req('x.trycloudflare.com', '/', { cookie: `${TOKEN_COOKIE}=nope` })).kind).toBe('deny')
  })

  it('passes with the cookie, including WebSocket upgrades to the ingest', () => {
    expect(decide(T, req('x.trycloudflare.com', '/phone/ingest', { cookie: `a=1; ${TOKEN_COOKIE}=${T}` })).kind)
      .toBe('pass')
  })

  it('reads cookies', () => {
    expect(readCookie('a=1; b=x%3Dy', 'b')).toBe('x=y')
    expect(readCookie(undefined, 'b')).toBeUndefined()
  })
})
