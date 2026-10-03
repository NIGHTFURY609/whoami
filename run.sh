#!/usr/bin/env bash
# One-shot start of the full live_cam stack on the Windows laptop (run from Git Bash):
#   Docker container ugv-run (ROS) + rosbridge port forward + Windows webcam bridge + UI dev server.
#
#   bash run.sh
#
# Env: CAM_INDEX (DirectShow camera index, default 0).
#
#   VIDEO=/n/path/rc_car.mp4 bash run.sh     # a recorded video instead of the webcam (eval only)
#
# VIDEO mode env: RC_HFOV (assumed horizontal FOV, deg, default 90), RC_CAM_Z (camera height, m, default 0.10),
# RC_CAM_PITCH (deg, default 0), VIDEO_SPEED (default 1.0; 0.5 = slow motion), VIDEO_LOOP (any value: rewind at the
# end). Any resolution works: the video is scaled to 640 wide keeping its aspect, and a placeholder calibration for
# that size is written to ugv_nav/config/cameras/rc_car_<W>x<H>.yaml. Map database: ~/.ros/ugv/rc_video.db.
# Ctrl+C stops the UI and the webcam bridge (camera LED off). The ROS stack keeps running in ugv-run and holds
# (no camera -> zero /cmd_vel); rerunning this script restarts it.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER=ugv-run
FORWARD=ugv-rosbridge-port
LOGS="$REPO/.logs"
mkdir -p "$LOGS"

log() { printf '\n== %s\n' "$*"; }

log "Model weights"
# Each live model needs either the HuggingFace safetensors folder (CUDA) or the OpenVINO IR pair (Intel).
W="$REPO/turing/weights"
missing=()
for m in rugd-segformer da3metric-large; do
  if [ -s "$W/$m/model.safetensors" ] || { [ -s "$W/$m.xml" ] && [ -s "$W/$m.bin" ]; }; then
    echo "$m ok"
  else
    missing+=("$m")
  fi
done
if [ ${#missing[@]} -gt 0 ]; then
  echo "Missing weights: ${missing[*]}"
  echo "Git does not ship the model weights. Follow user_manual.md (section 1: download; section 2 for Intel/OpenVINO)."
  exit 1
fi

log "Docker"
if ! docker info >/dev/null 2>&1; then
  echo "Docker Desktop is not running; starting it"
  "/c/Program Files/Docker/Docker/Docker Desktop.exe" >/dev/null 2>&1 &
  for _ in $(seq 60); do docker info >/dev/null 2>&1 && break; sleep 2; done
  docker info >/dev/null 2>&1 || { echo "Docker did not come up"; exit 1; }
fi
docker start "$CONTAINER" >/dev/null
echo "$CONTAINER up"

log "rosbridge port forward (host :9090 -> $CONTAINER:9090)"
# The container IP can change across Docker restarts, so recreate the forwarder against the current one.
IP="$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$CONTAINER")"
docker rm -f "$FORWARD" >/dev/null 2>&1 || true
docker run -d --name "$FORWARD" -p 9090:9090 alpine/socat \
  tcp-listen:9090,fork,reuseaddr "tcp-connect:$IP:9090" >/dev/null
echo "$FORWARD -> $IP:9090"

STREAM="$REPO/ugv_nav/ugv_bringup/scripts/webcam_stream.py"
WEBCAM_PID=""
ROS_ENV=()  # restart_live.sh overrides; empty = the laptop webcam defaults
if [ -n "${VIDEO:-}" ]; then
  log "Video bridge (:8090): $VIDEO (eval only: placeholder calibration, open loop)"
  [ -f "$VIDEO" ] || { echo "VIDEO=$VIDEO is not a file"; exit 1; }
  if netstat -ano | grep -qE '[:.]8090 +[^ ]+ +LISTENING'; then
    echo ":8090 is already serving (the webcam bridge?); stop it first so the driver reads the video"
    exit 1
  fi
  # A placeholder calibration for exactly the size the video is served at (aspect kept, scaled to 640 wide).
  CAL_TMP="$REPO/ugv_nav/config/cameras/rc_car.tmp.yaml"
  OUT="$(python "$STREAM" --video "$VIDEO" --calibration-out "$CAL_TMP" --hfov-deg "${RC_HFOV:-90}")"
  CAL="rc_car_${OUT%% *}.yaml"
  mv -f "$CAL_TMP" "$REPO/ugv_nav/config/cameras/$CAL"
  echo "calibration: ugv_nav/config/cameras/$CAL (placeholder, assumed HFOV ${RC_HFOV:-90} deg)"
  python -u "$STREAM" --video "$VIDEO" ${VIDEO_SPEED:+--speed "$VIDEO_SPEED"} ${VIDEO_LOOP:+--loop} >"$LOGS/webcam.log" 2>&1 &
  WEBCAM_PID=$!
  ROS_ENV=(-e "UGV_CAL=/repo/ugv_nav/config/cameras/$CAL" -e UGV_ALLOW_PLACEHOLDER=true
           -e "UGV_CAM_Z=${RC_CAM_Z:-0.10}" -e "UGV_CAM_PITCH=${RC_CAM_PITCH:-0}"
           -e UGV_DB=/root/.ros/ugv/rc_video.db)
else
  log "Webcam bridge (:8090)"
  if netstat -ano | grep -qE '[:.]8090 +[^ ]+ +LISTENING'; then
    echo "already serving on :8090, reusing it"
  else
    python "$STREAM" --index "${CAM_INDEX:-0}" >"$LOGS/webcam.log" 2>&1 &
    WEBCAM_PID=$!
  fi
fi
if [ -n "$WEBCAM_PID" ]; then
  sleep 3
  kill -0 "$WEBCAM_PID" 2>/dev/null || { echo "camera bridge died:"; cat "$LOGS/webcam.log"; exit 1; }
  echo "started (log: .logs/webcam.log)"
fi

cleanup() {
  [ -n "$WEBCAM_PID" ] && kill "$WEBCAM_PID" 2>/dev/null && echo "webcam bridge stopped"
}
trap cleanup EXIT

log "ROS stack (sync + colcon build + live_cam launch + rosbridge)"
MSYS_NO_PATHCONV=1 docker exec ${ROS_ENV[@]+"${ROS_ENV[@]}"} "$CONTAINER" bash /ws/restart_live.sh  # keep Git Bash from rewriting /ws/...

# First free port from 5173 up, so the summary below shows the exact URL (an older dev server may hold 5173).
UI_PORT=5173
while netstat -ano | grep -qE "[:.]$UI_PORT +[^ ]+ +LISTENING"; do UI_PORT=$((UI_PORT + 1)); done

log "Running"
printf '  %-22s %s\n' \
  "UI"                    "http://localhost:$UI_PORT  (opens in your browser)" \
  "API gateway"           "http://localhost:8080" \
  "rosbridge (websocket)" "ws://localhost:9090" \
  "Camera stream"         "http://localhost:8090/cam.mjpg" \
  "ROS log"               "docker exec $CONTAINER tail -f /tmp/live.log" \
  "Stop"                  "Ctrl+C (stops UI + camera bridge; ROS holds in $CONTAINER)"

log "UI dev server"
cd "$REPO/ui"
[ -d node_modules ] || npm install
npm run dev -- --port "$UI_PORT" --strictPort --open
