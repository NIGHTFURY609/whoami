# architecture.md / dev.md vs what Dev 2 builds

| Source wording | Reality / architecture | What we do |
|---|---|---|
| architecture §2/§6 "RTAB-Map VO/SLAM" | RTAB-Map has no monocular VO (its VO nodes need stereo or depth). Sensor is mono (D2) | Wheel odom (Dev 5) = odom frame + metric scale (D5). RTAB-Map RGB-only = loop closure / relocalization correcting `map->odom` |
| architecture §10 "VO-lost hold" | No VO with mono → nothing to be "lost" in the classic sense | "VO-lost" = dead-reckoning past a budget since the last **visual constraint** (`max_dead_reckon_m` / `_s`) → `/ugv/pose_valid=false` |
| dev.md task 1 "stereo (recommended) / RGB-D / mono (minimum)" configs | Only mono hardware exists (D2) | Only `rtabmap_mono.yaml` ships. Stereo/RGB-D configs not built — adding one is a config + launch arg, no kernel change |
| dev.md task 1 "tune visual feature tracking, bundle adjustment, keyframing" | With external odom, RTAB-Map's own tracking is loop-closure / proximity only | Tuned: features (GFTT/BRIEF), graph node spacing (`RGBD/LinearUpdate`), proximity detection, graph optimizer, loop-closure rejection (`RGBD/OptimizeMaxError`) |
| dev.md §3 TF owner = Dev 2; architecture silent | — | Dev 2 owns `map->odom` (rtabmap) **and** `odom->base_link` (`odom_tf_bridge`). Dev 5 diff-drive must set `publish_odom_tf=false` (D6) |
| dev.md "TF publish rate ≥ 15 Hz, jitter < 50 ms" | `odom->base_link` rate = Dev 5's wheel-odom rate | `tf_delay: 0.05` gives 20 Hz `map->odom`; `tf_rate_check` measures both edges. Needs Dev 5 odom ≥ 15 Hz |
| dev.md task 6 "rosbag playback" | Profiles §4: `bag` is eval only | `bag_eval.launch.py` replays inputs only; TF regenerated |
| Nav2 global costmap may expect `/map` from SLAM | Mono → no depth → RTAB-Map produces no occupancy grid | Dev 2 publishes **no** `/map` occupancy. Dev 3 builds costmaps from semantic + optional voxel layers in the `map` frame |
| dev.md hours / difficulty | Soft aim ~30h (§17) | Not a design constraint |
