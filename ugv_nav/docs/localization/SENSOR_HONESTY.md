# Sensor honesty — mono camera + wheel odometry (architecture §10, DoD item 7)

**No numbers in this file are measured yet.** Results rows are filled only from `drift_report` output of real runs.

## What mono + wheel odom can and cannot do

| Aspect | Mono + wheel odom (what we ship) | Stereo / RGB-D (recommended, §6) |
|---|---|---|
| Metric scale | From wheel odom only → wheel slip, tyre wear, wrong radius = scale error | From the camera itself |
| Odometry when wheels slip (mud, grass, sand) | Drifts silently; only a visual loop closure corrects it | Visual odometry keeps tracking |
| Loop closure / relocalization | Yes (bag-of-words), 3D words triangulated from motion | Yes, with direct depth |
| Occupancy grid `/map` | **None** (no depth) | Yes |
| Pure rotation in place | No parallax → no new 3D features | Fine |
| Low texture (flat dirt, sky, uniform grass) | Few features → no corrections → dead-reckon budget trips | Also weak, but better |
| Lighting change (sun/shade, dawn vs noon) | Relocalization against a map from different lighting may fail | Same issue for appearance-based loop closure |

Product consequence: `/ugv/pose_valid` goes false when we have driven `max_dead_reckon_m` without a visual constraint. Mono runs **will** hold more often than stereo would; that is the honest outcome.

## Measurement protocol (sim first, real later)

1. Record: `scripts/record_eval_bag.sh eval_bags/<run>` while driving a loop (sim ground truth available).
2. Map: `ros2 launch ugv_localization bag_eval.launch.py bag:=eval_bags/<run> mode:=mapping fresh_db:=true`
   with `ros2 run ugv_localization drift_eval --ros-args -r ground_truth:=<gt topic> -p use_sim_time:=true -p out_dir:=eval_out/<run>_map`.
3. Localize: replay a **different** run with `mode:=localize` against that db; same `drift_eval`.
4. Ablation: repeat mapping with `RGBD/ProximityBySpace:"false"` and loop closure disabled to show what vision adds over wheels alone.
5. Record the pose_valid duty cycle (% time true) from `/ugv/localization_status`.

## Results

| Run | Profile | Mode | Path m | ATE RMSE m | RPE 10 m % | Endpoint drift % | pose_valid % | Notes |
|---|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | — | not yet measured |
