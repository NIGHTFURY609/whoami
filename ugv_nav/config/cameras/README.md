# config/cameras/ (Dev 2 owned — architecture §7)

One YAML per physical (or simulated) camera, in ROS `camera_info_manager` format. Dev 5's driver
loads it to publish `CameraInfo`; Dev 1 and Dev 2 only consume that topic.

**No hand-typed or placeholder calibrations** (they would give RTAB-Map confident wrong geometry).
`ugv_localization.camera.load_calibration` rejects zero K, principal points outside the image, and
wrong distortion lengths.

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
