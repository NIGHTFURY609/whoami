// Token gate for reaching the dev server through a tunnel (run.sh PHONE=1: the phone camera page over Cloudflare).
// A tunnel URL is public, and behind this server sit the operator API (e-stop release, navigation goals; ugv_api v1
// has no authentication) and the phone camera ingest. So with UGV_TUNNEL_TOKEN set, a request that did not come
// from this machine needs the token: `?token=<T>` once sets an HttpOnly cookie and redirects to the clean URL,
// after that the cookie is enough. Loopback requests (the laptop's own browser) pass. Without a token every tunnel
// request is refused with a message naming PHONE=1 (Vite's own host check would only suggest editing allowedHosts);
// other requests pass as before.
import { timingSafeEqual } from 'node:crypto'
import type { IncomingMessage, ServerResponse } from 'node:http'
import type { Socket } from 'node:net'
import type { Connect, Plugin, ProxyOptions } from 'vite'

export const TOKEN_COOKIE = 'ugv_token'

type GateRequest = Pick<IncomingMessage, 'url' | 'headers'>

export type GateDecision =
  | { kind: 'pass' }
  | { kind: 'deny'; reason?: 'no-token' }
  | { kind: 'grant'; location: string; cookie: string }

const LOOPBACK = /^(localhost|127(\.\d{1,3}){3}|\[::1\])$/i

// Loopback by Host, and not relayed by Cloudflare (cloudflared adds cf-* headers; a tunnel configured to rewrite
// the Host to localhost must still not pass as local).
export function isLocal(req: GateRequest): boolean {
  if (req.headers['cf-connecting-ip'] || req.headers['cf-ray']) return false
  const host = (req.headers.host ?? '').replace(/:\d+$/, '')
  return LOOPBACK.test(host)
}

// Relayed by a Cloudflare tunnel: cloudflared's headers, or a quick-tunnel host.
export function viaTunnel(req: GateRequest): boolean {
  if (req.headers['cf-connecting-ip'] || req.headers['cf-ray']) return true
  return /\.trycloudflare\.com(:\d+)?$/i.test(req.headers.host ?? '')
}

export function readCookie(header: string | undefined, name: string): string | undefined {
  for (const part of (header ?? '').split(';')) {
    const [k, ...v] = part.trim().split('=')
    if (k === name) return decodeURIComponent(v.join('='))
  }
  return undefined
}

function same(a: string, b: string): boolean {
  const x = Buffer.from(a)
  const y = Buffer.from(b)
  return x.length === y.length && timingSafeEqual(x, y)
}

export function decide(token: string | undefined, req: GateRequest): GateDecision {
  if (!token) return viaTunnel(req) ? { kind: 'deny', reason: 'no-token' } : { kind: 'pass' }
  if (isLocal(req)) return { kind: 'pass' }
  const url = new URL(req.url ?? '/', 'http://gate')
  const given = url.searchParams.get('token')
  if (given !== null) {
    if (!same(given, token)) return { kind: 'deny' }
    url.searchParams.delete('token')
    return {
      kind: 'grant',
      location: url.pathname + url.search,
      // Lax, not Strict: the link is opened from another app or site (a chat, a QR scanner), and a Strict cookie is
      // not sent on the redirect that ends that cross-site navigation, so the page would answer 401 every time. Lax
      // still keeps it off cross-site POSTs and subresource requests.
      cookie: `${TOKEN_COOKIE}=${encodeURIComponent(token)}; Path=/; HttpOnly; Secure; SameSite=Lax`,
    }
  }
  const cookie = readCookie(req.headers.cookie, TOKEN_COOKIE)
  return cookie !== undefined && same(cookie, token) ? { kind: 'pass' } : { kind: 'deny' }
}

function middleware(token: string | undefined): Connect.NextHandleFunction {
  return (req: IncomingMessage, res: ServerResponse, next: Connect.NextFunction) => {
    const d = decide(token, req)
    if (d.kind === 'pass') return next()
    if (d.kind === 'grant') {
      res.writeHead(302, { Location: d.location, 'Set-Cookie': d.cookie, 'Cache-Control': 'no-store' })
      res.end()
      return
    }
    res.writeHead(d.reason ? 403 : 401, { 'Content-Type': 'text/plain', 'Cache-Control': 'no-store' })
    res.end(d.reason
      ? 'UGV console: tunnel access is off. Start the laptop side with PHONE=1 bash run.sh (Git Bash), then open\n'
        + 'the phone.html?token=... link it prints, on this tunnel host.\n'
      : 'UGV console: open the link with ?token=... that run.sh printed\n')
  }
}

// Ahead of Vite's own middlewares (the proxy included), on the dev server and on `vite preview`.
export function tunnelGate(token: string | undefined): Plugin {
  return {
    name: 'ugv-tunnel-gate',
    configureServer(server) { server.middlewares.use(middleware(token)) },
    configurePreviewServer(server) { server.middlewares.use(middleware(token)) },
  }
}

// WebSocket upgrades go to the proxy without passing the middlewares: refuse them there.
export function gateWs(token: string | undefined): ProxyOptions['configure'] {
  return (proxy) => {
    proxy.on('proxyReqWs', (_proxyReq, req, socket: Socket) => {
      if (decide(token, req).kind !== 'pass') socket.destroy()
    })
  }
}
