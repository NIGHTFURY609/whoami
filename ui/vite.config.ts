import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig, type ProxyOptions } from 'vite'
import { gateWs, tunnelGate } from './vite/tunnelGate.ts'

// The console talks only to Dev 5's operator gateway (ugv_nav/ugv_api). In dev the gateway is proxied so the
// browser stays same-origin; override the target with UGV_API_URL (e.g. a robot on the LAN).
const gateway = process.env.UGV_API_URL ?? 'http://127.0.0.1:8080'

// Phone camera (phone.html): frames go over /phone/ingest to the Windows bridge's WebSocket (webcam_stream.py
// --phone), its status is read from the bridge's HTTP port. Same origin, so the page works through one tunnel.
const phoneIngest = process.env.UGV_PHONE_INGEST_URL ?? 'ws://127.0.0.1:8091'
const phoneBridge = process.env.UGV_PHONE_BRIDGE_URL ?? 'http://127.0.0.1:8090'

// Set by run.sh PHONE=1: then every non-local request needs the token (vite/tunnelGate.ts). Without it the gate
// refuses tunnel requests itself, with a message naming PHONE=1. Extra hosts for a named tunnel:
// UGV_ALLOWED_HOSTS=ugv.example.org,...
const token = process.env.UGV_TUNNEL_TOKEN || undefined
const allowedHosts = ['.trycloudflare.com',
  ...(process.env.UGV_ALLOWED_HOSTS ?? '').split(',').map((h) => h.trim()).filter(Boolean)]

const proxy: Record<string, ProxyOptions> = {
  '/api': { target: gateway, changeOrigin: true, configure: gateWs(token) },
  '/phone/ingest': { target: phoneIngest, ws: true, configure: gateWs(token) },
  '/phone/status': { target: phoneBridge },
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss(), tunnelGate(token)],
  server: { proxy, allowedHosts },
  preview: { proxy, allowedHosts },
  build: { rolldownOptions: { input: { main: 'index.html', phone: 'phone.html' } } },
})
