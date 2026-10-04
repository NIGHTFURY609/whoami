# ugv_bringup: camera driver (Dev 5)

One capture (a V4L2 device or a network stream), published for the system and for the web UI. No second pipeline, no relay node.

```
ros2 launch ugv_bringup camera.launch.py calibration_file:=/path/to/real_calibration.yaml [device:=/dev/video0]
```

The stamp on every message is when the frame arrived from the capture, not the sensor's exposure time, minus
the parameter `transport_latency_s` (seconds, default 0; see "Phone camera over a tunnel").

| Topic | Type | QoS | For |
|---|---|---|---|
| `/camera/image_raw` | `sensor_msgs/Image` rgb8 | reliable, volatile, depth 1 | Dev 1 perception, Dev 2 RTAB-Map |
| `/camera/camera_info` | `sensor_msgs/CameraInfo` | reliable, transient local, depth 1 | Dev 1, Dev 2, safety arbiter. One per image, same stamp |
| `/image_raw/compressed` | `sensor_msgs/CompressedImage` jpeg | reliable, volatile, depth 1 | web UI (rosbridge), `compressed_rate_hz` (default 5) |
| `/camera_info` | `sensor_msgs/CameraInfo` | reliable, transient local, depth 1 | web UI, published with each compressed frame |

The internal names (`/camera/*`) are what Dev 1 and Dev 2 already use; the UI defaults to
`/image_raw/compressed` and `/camera_info` (`ui/src/App.tsx`). The driver publishes both sets from the
same capture, so neither side was renamed.

`calibration_file` has no default and the driver fails closed: no valid calibration (Dev 2's
`ugv_localization.camera` refuses zero or fake K), a missing, empty or malformed file, a camera whose
resolution differs from the calibration, or a calibration flagged `placeholder: true` (below), means it exits with
one clear message and publishes nothing.

**Placeholder calibrations are refused unless you opt in.** A YAML with `placeholder: true` holds another camera's
numbers. By default the driver logs an ERROR naming the file, the reason and the override, publishes no Image and no
CameraInfo, and exits; perception, RTAB-Map and the arbiter then see a silent camera, so nothing moves.
`allow_placeholder_calibration:=true` (on `camera.launch.py` or `bringup.launch.py`, default false) loads it anyway,
with a WARN at start-up and every 10 s. Use it only for bring-up (the robot on a cart or carried by hand, a recording
for debugging): depth, the depth cloud Nav2 marks and the map are all scaled wrong, so never for an autonomous run or a
mapping run that counts.

## Calibrating a real camera

The driver will not run without a calibration, and calibrating needs the camera's images, so there is a
calibration mode that breaks the loop. It publishes only raw `/camera/image_raw` (no CameraInfo, no UI
stream), which nothing downstream accepts, and the safety arbiter sees the camera as silent.

```
ros2 launch ugv_bringup camera.launch.py calibration_mode:=true width:=640 height:=480
ros2 run camera_calibration cameracalibrator --size 8x6 --square 0.025     --ros-args -r image:=/camera/image_raw -r camera:=/camera
# COMMIT writes the YAML; save it as ugv_nav/config/cameras/<camera_name>.yaml (see config/cameras/README.md)
# then stop calibration mode and launch normally with calibration_file:=...
```

Calibrate at the resolution you will run at: K is only valid there.

## Phone camera over a tunnel

The phone serves a stream OpenCV can open by URL (MJPEG over HTTP or RTSP) and the driver reads it through the
tunnel with `device:=http://...`. Any `device` that is not an index or a `/dev/...` path is read on its own
thread, and only the newest frame is published: a stalled stream that then delivers a burst of old frames yields
one frame, and while nothing new arrives the driver is silent (the safety arbiter sees a dead camera, never a
repeated old frame). V4L2 devices are read directly as before. The number of frames dropped this way is logged
every 10 s while it changes.

- Set `fps` above the stream's own rate (for example `fps:=30` for a 15 fps stream). The driver publishes on its
  `fps` timer, and if that is slower than the stream, frames are superseded in steady state and `dropped` stops
  being a clean indicator of a stall followed by a burst.
- A video file given as `device` is not paced: it is drained at decode speed and only the newest frames are
  published. To replay a recording, serve it as a stream instead ("Replaying a recorded video" below).
- A tunnel that stalls without erroring leaves a blocked read that counts no failure. The driver therefore
  watches the time since the last frame: after more than 2 s it logs a WARN `no new frame for N s` every 10 s
  while it lasts (saying whether the read is blocked or failing, and why), and an INFO when frames resume. After
  30 consecutive failed reads it also logs `camera is not delivering frames`, once per outage.

```
ros2 launch ugv_bringup camera.launch.py calibration_file:=<repo>/ugv_nav/config/cameras/phone_640x480.yaml \
    device:=http://<tunnel host>:<port>/<stream> transport_latency_s:=<measured seconds>
```

- `transport_latency_s` (seconds, finite, 0 to 5, default 0, refused at start-up otherwise, so `350` typed for
  milliseconds does not pass): a network stream has no capture timestamps, so the stamp is the frame's arrival
  time minus this measured delay. `bringup.launch.py` takes the same argument. How to measure it:
  `config/cameras/README.md`.
- `phone_640x480.yaml` is the owner's phone, calibrated 2026-10-03 through the browser page below (RMS 0.35 px; the
  header says how). It is valid only for that phone in that 640x480 mode: another phone starts again from a flagged
  placeholder (`placeholder: true`, refused unless `allow_placeholder_calibration:=true`).

## Phone browser camera over a Cloudflare tunnel

No app on the phone: it opens the web UI's `phone.html` through a tunnel to the laptop and sends its camera's frames,
which the Windows camera bridge serves on the usual `http://host.docker.internal:8090/cam.mjpg`. The driver path is
the one above, unchanged (newest frame only, stamp = arrival - `transport_latency_s`).

```
phone.html (getUserMedia 640x480) --JPEG over WebSocket /phone/ingest--> UI dev server (token gate, proxy)
  --> webcam_stream.py --phone (ingest ws 127.0.0.1:8091) --> :8090/cam.mjpg --> camera driver
```

From Git Bash on the laptop:

```
PHONE=1 bash run.sh                                   # bridge + UI; prints the UI port and an access token
cloudflared tunnel --url http://localhost:<UI port>   # second terminal; prints https://<name>.trycloudflare.com
# on the phone: https://<name>.trycloudflare.com/phone.html?token=<token>, hold it landscape, tap START
```

The ROS stack starts once the first phone frame arrives (the driver's first open needs one). The overlays show in
the laptop's console at `http://localhost:<UI port>`, as with the webcam. Env: `PHONE_CAM_Z`, `PHONE_CAM_PITCH`
(mount), `PHONE_LATENCY_S` (`transport_latency_s`), `PHONE_WAIT_S`; map database `~/.ros/ugv/phone.db`.

- **640x480 only, at three places.** The page asks the camera for 640x480 and sends a frame only when the camera
  itself delivers exactly that, never scaled or cropped (K is valid only for the calibrated mode). Upright, the
  camera gives 480x640: the page stops sending and asks for landscape (and locks landscape where the browser lets
  it, Android in fullscreen). The bridge drops every frame that is not 640x480 (`/phone/status` counts them), and
  the driver refuses a stream whose size differs from `phone_640x480.yaml`.
- **Newest frame only.** The page skips a frame while the previous one is still buffered or two are unanswered by
  the bridge, so a slow tunnel drops frames instead of delivering a late backlog stamped as current.
- **Phone drops.** A locked screen or another app ends the page's session, and the bridge ends the stream (also
  after `--stall-s`, 2 s, without a frame). The driver sees a dead camera (the arbiter holds) and reopens the URL
  every 20 failed reads; the bridge answers 503 until the phone sends again, then the stream resumes without a
  restart. One phone at a time: a second one is refused.
- **Access.** A tunnel URL is public, and the UI server also proxies the operator API (e-stop release, goals; no
  authentication in v1). `run.sh` therefore sets `UGV_TUNNEL_TOKEN`: any request that did not come from this machine
  (or that Cloudflare relayed) needs `?token=` once, which becomes an HttpOnly cookie; everything else gets 401,
  WebSocket upgrades included. Without the token (plain `npm run dev`) tunnel hosts are not allowed at all. Anyone
  holding the link and token can reach the console: stop the tunnel when done. Named tunnels: add their host with
  `UGV_ALLOWED_HOSTS=ugv.example.org`.
- **Calibration and latency.** `phone_640x480.yaml` is calibrated through this same page (2026-10-03; steps in
  `config/cameras/README.md`; its **LOCK FOCUS + EXPOSURE** button holds the lens where Android allows it). `run.sh`
  passes `allow_placeholder_calibration` only if the file is ever flagged as a placeholder again. The page shows the median round trip to the bridge; half of it is a rough hint only:
  measure `transport_latency_s` with the clock-photo method and rerun with `PHONE_LATENCY_S=<median s>`.
- Keep the page in front with the screen on (it holds a wake lock); a background tab gets no camera.

## Full stack: `bringup.launch.py profile:=live_cam`

```
ros2 launch ugv_bringup bringup.launch.py profile:=live_cam     calibration_file:=<camera yaml> device:=<V4L2 path or stream URL>     camera_x:=<m> camera_y:=<m> camera_z:=<m> camera_pitch_deg:=<deg>     perception_src:=<repo>/turing/src [mode:=mapping|localize] [robot:=primary]
```

Optional: `transport_latency_s:=<s>`, `allow_placeholder_calibration:=true` (bring-up only, see above),
`map_assembler:=true` (RTAB-Map's whole 3D map on `/rtabmap/cloud_map`, for RViz or other tools; the web viewer no
longer shows it, mindmap D27; off by default because its memory grows with the map: `docs/mapping/README.md`).

Starts camera driver, robot description (`ugv_robot_description`: base_link -> camera_optical_frame from the
measured mount, no defaults), Dev 1 perception, Dev 2 localization, Dev 3 semantic costmap (`ugv_costmap`),
Dev 4 Nav2, the safety arbiter and the operator API. `sim` and `bag` are refused until they are wired.

## Mapping run and the map view

`mode:=mapping` (default) builds the RTAB-Map database at `database_path` (default `~/.ros/ugv/rtabmap.db`, saved when
the stack stops); `mode:=localize` loads it read-only. The operator API this profile starts (`api_host`, `api_port`,
default `0.0.0.0:8080`) serves the 3D map to the web UI's map view (`GET /api/v1/map/...`), so open the UI
(`cd ui && npm run dev`) and pick **map**. Before a run that counts, have the real phone calibration (not the
placeholder), a measured `transport_latency_s` and a lit scene. What the map shows and where it is not to be trusted:
`docs/mapping/README.md`. Elevation mapping is not built yet.

## live_cam on a Windows laptop (ROS in Docker)

Docker on Windows cannot open a USB webcam, so the host serves it and the driver reads the stream.

1. Host (Windows Python + opencv-python): `python ugv_nav/ugv_bringup/scripts/webcam_stream.py` serves
   `http://<host>:8090/cam.mjpg` (newest frame only, no backlog). Stop it with Ctrl+C: it holds the camera.
2. Image: `docker build -t ugv-live -f ugv_nav/ugv_bringup/docker/live.Dockerfile ugv_nav/ugv_bringup/docker`
   (needs `ugv-lyrical-nav2` first, see that Dockerfile).
3. Container with the GPU, the repo and WSLg (for the calibrator window):
   ```
   docker run -it --gpus all -p 8080:8080 -v <repo>:/repo        -v /run/desktop/mnt/host/wslg/.X11-unix:/tmp/.X11-unix -e DISPLAY=:0 ugv-live
   ```
   then build the workspace from `/repo` and use `device:=http://host.docker.internal:8090/cam.mjpg`.

## Replaying a recorded video (eval only)

A video from another camera (an RC car's, say) goes through the same path as the webcam. `webcam_stream.py --video`
serves the file on `:8090` paced at its own frame rate, and the driver reads it like a live camera. From Git Bash on
the laptop:

```
VIDEO=/n/path/to/rc_car.mp4 bash run.sh
```

- Any resolution: the video keeps its aspect ratio and full field of view and is scaled to 640 wide (1920x1080
  becomes 640x360). `--calibration-out` writes a `placeholder: true` calibration for exactly that size to
  `config/cameras/rc_car_<W>x<H>.yaml`, from an assumed horizontal FOV (`RC_HFOV`, default 90 deg). No distortion is
  modelled. Depth and map scale are off by the FOV error: open loop only, never a run that counts.
- The mount is assumed, not measured: `RC_CAM_Z` (m, default 0.10) and `RC_CAM_PITCH` (deg, default 0). The map goes
  to `~/.ros/ugv/rc_video.db`, so the laptop map is left alone.
- The video waits at frame 0 until you press **PLAY** in the UI's **Video replay** panel, so the stack can come up
  first. Pause and replay are in the same panel.
- **Wait for perception** (on by default): each frame is sent only after Dev 1's mask for the previous one is back, so
  every frame is analysed and the camera view shows each frame with its own overlay ("buffering"; playback slows to
  the perception rate, about 4 frames/s on the laptop). Off: real time, like a live camera.
- **Recorded overlays:** during a play with "wait for perception" on, every frame's JPEG, mask and depth are saved in
  `<video stem>.overlays/` beside the video (masks are matched to frames by image stamp, with the container's clock
  offset measured through rosapi). **PLAY RECORDED** then shows the video with those overlays at its own speed,
  without the stack, as often as wanted. A later play only fills frames still missing; **DELETE RECORDING** starts
  over (do that after changing the model or its configuration).
- `VIDEO_SPEED=0.5` plays in slow motion if visual odometry loses a fast car (stamps are arrival times, so it just
  looks like a slower car). Paused, ready or ended, the frame on screen is sent again every 2 s: OpenCV drops a
  network stream that is silent for about 30 s and the driver never gets it back. `VIDEO_LOOP=1` rewinds at the end
  instead, which visual odometry sees as a teleport.
- Restarting the video bridge cuts the camera driver's stream for good: restart the stack after it (`bash run.sh`
  does both).
- Nothing moves: the arbiter keeps `/cmd_vel` at zero. Running `bash run.sh` without `VIDEO` goes back to the webcam
  and its calibration.

Not covered here: motor driver, wheel odometry, Gazebo `sim` profile.
