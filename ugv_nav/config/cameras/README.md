# config/cameras/ (Dev 2 owned — architecture §7)

One YAML per physical (or simulated) camera, in ROS `camera_info_manager` format. Dev 5's driver
loads it to publish `CameraInfo`; Dev 1 and Dev 2 only consume that topic.

**No hand-typed or placeholder calibrations** (they would give RTAB-Map confident wrong geometry).
`ugv_localization.camera.load_calibration` rejects zero K, principal points outside the image, and
wrong distortion lengths.

**The one exception: a flagged placeholder.** A new camera has to be running before it can be calibrated, and
the stack can only be brought up and checked with some K. For that, and only that, a file may carry the
top-level key `placeholder: true` (it must be a boolean; anything else is rejected). `phone_640x480.yaml` was
one (the laptop webcam's K and D copied unchanged) until the phone was calibrated on 2026-10-03; its header records
how. What happens to a flagged file:

- The loader accepts it and sets `CameraCalibration.placeholder`. Nothing else is relaxed: K is still
  validated, and the driver still refuses a camera that delivers another resolution.
- The camera driver **refuses it by default** (an ERROR naming the file, the reason and the override; no Image, no
  CameraInfo, exit), so the default path fails closed as for a fake K. Only `allow_placeholder_calibration:=true`
  (`camera.launch.py` or `bringup.launch.py`) loads it, for bring-up on a cart or by hand, never for an autonomous run;
  the driver then logs a WARN naming the file at start-up and again every 10 s for as long as it is loaded.
- It is not good enough for any mapping run that counts: depth is stretched along the viewing axis in
  proportion to the focal-length error, and RTAB-Map builds the map at that wrong scale with full confidence.
- A real calibration never carries the key. `test_camera_calib.py` fails if a file that copies the laptop
  webcam's K and D is not flagged.

Replacing the phone placeholder (hardware steps, done by the owner):

1. Lock the phone's focus and exposure (autofocus changes the focal length, so K would differ between views;
   auto-exposure changes brightness and motion blur while you calibrate). Mount it the way it will be used. With
   the browser page (`phone.html`, `ugv_bringup/README.md` "Phone browser camera over a Cloudflare tunnel"), its
   **LOCK FOCUS + EXPOSURE** button does this where the browser allows it (Android Chrome). iOS gives a web page
   no focus or exposure control: calibrate an iPhone in steady light and expect K to vary with focus distance.
2. Run the driver in calibration mode against the phone's stream; it publishes raw images only, no CameraInfo:
   `ros2 launch ugv_bringup camera.launch.py calibration_mode:=true device:=<stream URL> width:=640 height:=480`
   (browser page: `webcam_stream.py --phone` running and the page streaming, `device:=http://host.docker.internal:8090/cam.mjpg`;
   calibrate the browser stream, not the phone's camera app, because it is the 640x480 mode the page sends)
3. Calibrate with `cameracalibrator` (see "Real camera" below), at 640x480.
4. Save the result as `ugv_nav/config/cameras/phone_640x480.yaml`, with the header described below.
5. Remove the `placeholder` line (the flag only ever marks stand-in numbers). `allow_placeholder_calibration` is then
   no longer needed.

## Network cameras: transport latency
A stream reached through a tunnel carries no capture timestamps, and the driver stamps each frame with the time
it arrived on the laptop, which is later than the moment the phone captured it. The driver parameter
`transport_latency_s` (seconds, finite, 0 to 5, default 0; the driver refuses anything else, so milliseconds
typed by mistake are caught) is subtracted from that arrival time. Measure it once
per phone, stream setup and network route:

1. Show a clock with millisecond resolution on the laptop screen (any page or terminal that prints the time to
   the millisecond) and point the phone at it.
2. View the received frames (`/camera/image_raw`, in any image viewer) next to the live clock and take a
   screenshot of the two together.
3. Latency = the time the live clock shows minus the time shown inside the received frame. Repeat for 10 to 20
   screenshots and use the median: the stream's delay is not constant.
4. Set it where the driver is launched: `transport_latency_s:=0.35` on `camera.launch.py` or
   `bringup.launch.py`. Measure again after changing the network, the tunnel or the stream settings.

The figure includes the phone's encoding, the tunnel and the laptop's decoding; the screen refresh adds up to
one frame period (about 17 ms) of error, which is small against typical tunnel delays.

## Sim camera
Gazebo derives intrinsics from the SDF (`horizontal_fov`, width, height). Capture them — don't compute by hand:

```bash
ros2 run ugv_localization camera_info_to_yaml --topic /camera/camera_info \
    --name sim_front_mono --out ugv_nav/config/cameras/sim_front_mono.yaml
```

## Real camera
Calibrate with a checkerboard (`camera_calibration` package), then save its output here:

```bash
ros2 run camera_calibration cameracalibrator --size 8x6 --square 0.025 \
    --ros-args -r image:=/camera/image_raw -r camera:=/camera
# "COMMIT" writes the YAML; copy it to ugv_nav/config/cameras/<camera_name>.yaml
```

Record in the file header: camera model, lens, resolution, date, who calibrated, reprojection error.
