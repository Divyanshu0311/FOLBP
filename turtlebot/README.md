# turtlebot

Minimal ROS 2 Python scripts for a TurtleBot3 + OpenMANIPULATOR-X. No ROS workspace, no `colcon` — just `python3 <file>.py` after sourcing your ROS 2 install.

## Pipeline

```
/odom  ──▶  localization.py  ──▶  /robot_pose ─┐
                                                ├──▶  controller.py  ──▶  /cmd_vel
                      /goal_pose  ──────────────┘
                  (RViz / send_setpoint.py)
```

- `localization.py` — converts `nav_msgs/Odometry` from `/odom` into a flat `geometry_msgs/Pose2D` on `/robot_pose` (x, y, yaw). Treats the start position as origin so the published pose begins at (0, 0, 0); exposes `/reset_localization` (`std_srvs/Empty`) to re-zero on demand.
- `controller.py` — subscribes to `/robot_pose` and `/goal_pose`, runs a proportional go-to-goal controller, publishes `/cmd_vel`. Exposes `/controller/pause` and `/controller/resume` (`std_srvs/Empty`) so other nodes (e.g. `visual_servo.py`) can briefly take over `/cmd_vel`.
- `send_setpoint.py` — convenience CLI to publish a `PoseStamped` goal once. Supports relative (body- or world-frame) deltas via `-r`.
- `visual_servo.py` — vision-based final approach. Detects a red target, takes over `/cmd_vel`, and drives until the target fills the frame. Auto-pauses the controller on start and resumes it on finish.

## Run

Bring the robot up first (in its own terminals):
```bash
ros2 launch turtlebot3_manipulation_bringup hardware.launch.py
# or for sim
ros2 launch turtlebot3_manipulation_bringup gazebo.launch.py
```

Then in two terminals, each with ROS 2 sourced:
```bash
python3 localization.py
python3 controller.py
```

Send goals — three options:

1. **RViz** — click "2D Goal Pose" and drop a target on the map. RViz publishes to `/goal_pose` directly.
2. **CLI helper** — `x`, `y` in **centimetres**, `theta` in **degrees** (helper converts to m + rad before publishing):
   ```bash
   # absolute (world frame, same origin localization.py uses)
   python3 send_setpoint.py 100 50 0              # 1.0 m, 0.5 m, frame=odom
   python3 send_setpoint.py 100 50 90 --frame map # face +Y

   # relative — body frame (forward, left, yaw delta) added to current /robot_pose
   python3 send_setpoint.py 50 0 0 -r             # forward 50 cm
   python3 send_setpoint.py 0 0 90 -r             # rotate 90 deg in place
   python3 send_setpoint.py 30 20 -45 -r          # forward 30 cm, left 20 cm, then rotate -45 deg

   # relative in world frame (just adds to current x,y; ignores current heading)
   python3 send_setpoint.py 50 0 0 -r --world     # +50 cm along world x
   ```
3. **Plain `ros2 topic pub`:**
   ```bash
   ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped \
     '{header: {frame_id: "odom"}, pose: {position: {x: 1.0, y: 0.5, z: 0.0}, orientation: {w: 1.0}}}'
   ```

## Units

| where | distance | angle |
|---|---|---|
| `send_setpoint.py` CLI args | **centimetres** | **degrees** |
| ROS topics (`/odom`, `/goal_pose`, `/cmd_vel`, `/robot_pose`) | metres / m·s⁻¹ | radians |
| `controller.py` params (`position_tol`, `max_lin`, …) | metres / m·s⁻¹ | radians |

The CLI tool converts cm → m and deg → rad before publishing. RViz "2D Goal Pose" already publishes in metres + quaternions, so it works as-is.

## Re-zeroing localization

`/robot_pose` is reported in a local frame whose origin is wherever the robot was when `localization.py` started. To move the origin to the current position at any later moment, call the service:

```bash
ros2 service call /reset_localization std_srvs/srv/Empty
```

After a reset, the next `/robot_pose` reads (0, 0, 0) and goals on `/goal_pose` are interpreted relative to that new origin. The reset is logged by `localization.py` (so check that terminal to confirm). Disable auto-zero with `--ros-args -p auto_zero_on_start:=false` if you'd rather keep raw odometry coordinates.

## Frame note

`localization.py` reports pose in the **odom** frame (since it reads `/odom`). RViz's "2D Goal Pose" defaults to whatever fixed frame is set — usually `map` once AMCL is up. If frames don't match, the controller logs a warning and uses the coordinates as-is (no TF transform). Easiest fixes:

- Set RViz fixed frame to `odom` while testing without a map, **or**
- pass `--frame map` and switch `expected_frame` on the controller (`--ros-args -p expected_frame:=map`) once you're running AMCL with a real `map -> odom` transform.

## Parameters

`controller.py` exposes ROS 2 params (override with `--ros-args -p name:=value`):

| param | default | meaning |
|---|---|---|
| `pose_topic` | `/robot_pose` | where to read current pose |
| `goal_topic` | `/goal_pose` | where to read target |
| `cmd_topic` | `/cmd_vel` | velocity command output |
| `expected_frame` | `odom` | warn if incoming goal frame differs |
| `k_lin` | 0.5 | proportional gain on distance |
| `k_ang` | 1.5 | proportional gain on heading error |
| `max_lin` | 0.18 | linear speed cap (m/s) |
| `max_ang` | 1.2 | angular speed cap (rad/s) |
| `min_lin` | 0.03 | linear speed floor — below this the wheels stall (m/s) |
| `min_ang` | 0.15 | angular speed floor — below this the bot can't overcome stiction (rad/s) |
| `position_tol` | 0.02 | distance at which goal counts as reached (m, ~2 cm) |
| `heading_gate` | 0.3 | rotate-in-place if heading error exceeds this (rad, ~17°) |
| `align_final` | true | also align yaw to goal yaw after arrival (set false to ignore goal yaw) |
| `yaw_tol` | 0.02 | yaw tolerance for final alignment (rad, ~1.1°) |

## Rough → fine approach with visual servo

The flow when you want vision to refine the final approach:

```
send_setpoint.py 100 0 0   →  controller drives to (1.0, 0)
                                    ↓ target now visible
visual_servo.py             →  pauses controller, takes over /cmd_vel
                                    ↓ red target held for 1 s
                            →  resumes controller, exits
send_setpoint.py 0 0 90 -r  →  controller spins 90° (next setpoint)
```

Both nodes can be running the whole time — `visual_servo.py` calls `/controller/pause` on start and `/controller/resume` on exit, so they never fight on `/cmd_vel`.

## Topics + services used

| name | direction | type | source |
|---|---|---|---|
| `/odom` | read | `nav_msgs/Odometry` | TB3 bringup |
| `/cmd_vel` | write | `geometry_msgs/Twist` | TB3 bringup consumes it |
| `/robot_pose` | created | `geometry_msgs/Pose2D` | localization.py |
| `/goal_pose` | read | `geometry_msgs/PoseStamped` | RViz / send_setpoint.py |
| `/camera/image_raw` | read | `sensor_msgs/Image` | camera_ros |
| `/reset_localization` | service | `std_srvs/Empty` | `ros2 service call` |
| `/controller/pause` | service | `std_srvs/Empty` | visual_servo.py / `ros2 service call` |
| `/controller/resume` | service | `std_srvs/Empty` | visual_servo.py / `ros2 service call` |
