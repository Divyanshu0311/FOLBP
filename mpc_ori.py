#!/usr/bin/env python3
"""
Receding Horizon Control (MPC) for Crazyflie Obstacle Avoidance
================================================================

Two modes:
  GLOBAL_PLAN_ONLY = True
    → Solve ONE trajectory from start to goal before takeoff
    → Display the planned path for verification
    → Execute waypoints sequentially (open-loop)

  GLOBAL_PLAN_ONLY = False
    → Full receding horizon: re-solve MPC every timestep using
      live position feedback (closed-loop)

Obstacle constraints use SOFT formulation (slack variables) so the
solver never becomes infeasible, even if drone starts inside obstacle.
"""

import sys
import signal
import numpy as np
import casadi as ca
import time
import csv
import os
import rclpy
from rclpy.logging import get_logger
from crazyflie_py import Crazyswarm


# ==============================================================================
#                         USER CONFIGURATION
# ==============================================================================

# ---- Mode ----
GLOBAL_PLAN_ONLY = False     # True = plan once, execute open-loop
                             # False = receding horizon (replan every step)

# ---- Start & Goal [x, y, z] in meters ----
START_POSITION = [0.0, 0.0, 0.5]
GOAL_POSITION  = [-2.0, 0.0, 0.5]

# ---- Obstacles: Rectangular Parallelepipeds ----
# Each: [cx, cy, cz, half_x, half_y, half_z]
OBSTACLES = [
    [-1.0, 0.0, 0.5,  0.075, 0.25, 0.35],
]

# ---- MPC Tuning ----
HORIZON     = 15        # Prediction horizon for RHC mode (steps)
DT_MPC      = 0.3       # MPC discretization timestep (s) — also waypoint spacing
V_MAX       = 0.4       # Max velocity (m/s)
A_MAX       = 0.8       # Max acceleration (m/s^2)
OBS_MARGIN  = 0.15      # Safety margin around obstacles (m)

# For GLOBAL_PLAN_ONLY mode, the horizon is auto-computed to be long
# enough to reach the goal. You can also override it here:
GLOBAL_HORIZON_OVERRIDE = None   # Set to e.g. 40 to force; None = auto

# ---- Cost Weights ----
Q_POS       = 10.0      # Position error to goal
Q_VEL       = 1.0       # Velocity penalty
R_ACC       = 0.1       # Control effort
P_TERM      = 50.0      # Terminal cost multiplier
SLACK_PENALTY = 500.0   # Soft obstacle violation penalty

# ---- Flight Parameters ----
RATE            = 5.0    # Replan rate for RHC mode (Hz)
MAX_STEPS       = 600    # Max steps before forced landing
GOAL_TOLERANCE  = 0.10   # Goal reached distance (m)

# ---- Output ----
LOG_DIR     = "."
LOG_CSV     = "rhc_flight_log.csv"
PLOT_FILE   = "rhc_flight_results.png"

# ==============================================================================
#                       END OF USER CONFIGURATION
# ==============================================================================

logger = get_logger("mpc_flight")

FLIGHT_HEIGHT = START_POSITION[2]
START_2D = np.array(START_POSITION[0:2])
GOAL_2D  = np.array(GOAL_POSITION[0:2])
OBSTACLES_2D = [[o[0], o[1], o[3], o[4]] for o in OBSTACLES]


# ==============================
# MPC SOLVER
# ==============================
class DoubleIntegratorMPC:
    """
    2D Double integrator MPC with SOFT obstacle constraints.

    Soft constraint:  dx^2 + dy^2 + s >= 1,  s >= 0
    Cost penalty:     SLACK_PENALTY * s  (per obstacle, per step)

    This ensures the solver ALWAYS finds a feasible solution, even
    when starting inside an obstacle.
    """

    def __init__(self, horizon, dt, obstacles_2d, goal_2d,
                 v_max, a_max, margin, slack_penalty):
        self.H = horizon
        self.dt = dt
        self.goal = goal_2d
        self.v_max = v_max
        self.a_max = a_max
        self.obstacles = obstacles_2d
        self.margin = margin
        self.n_obs = len(obstacles_2d)
        self.slack_penalty = slack_penalty

        self.solve_count = 0
        self.fail_count = 0
        self._prev_X = None
        self._prev_U = None

        self._build()

    def _build(self):
        H, dt = self.H, self.dt
        opti = ca.Opti()

        X = opti.variable(4, H + 1)   # [px, py, vx, vy]
        U = opti.variable(2, H)       # [ax, ay]
        x0 = opti.parameter(4)
        xg = opti.parameter(2)

        obs_p = []
        for i in range(self.n_obs):
            obs_p.append(opti.parameter(4))

        # Dynamics: double integrator
        for k in range(H):
            opti.subject_to(X[0, k+1] == X[0, k] + X[2, k]*dt + 0.5*U[0, k]*dt**2)
            opti.subject_to(X[1, k+1] == X[1, k] + X[3, k]*dt + 0.5*U[1, k]*dt**2)
            opti.subject_to(X[2, k+1] == X[2, k] + U[0, k]*dt)
            opti.subject_to(X[3, k+1] == X[3, k] + U[1, k]*dt)

        opti.subject_to(X[:, 0] == x0)

        # Bounds
        for k in range(H + 1):
            opti.subject_to(opti.bounded(-self.v_max, X[2, k], self.v_max))
            opti.subject_to(opti.bounded(-self.v_max, X[3, k], self.v_max))
        for k in range(H):
            opti.subject_to(opti.bounded(-self.a_max, U[0, k], self.a_max))
            opti.subject_to(opti.bounded(-self.a_max, U[1, k], self.a_max))

        # Soft obstacle constraints
        cost = 0
        slack_vars = []
        for i in range(self.n_obs):
            obs = obs_p[i]
            obs_s = []
            for k in range(H + 1):
                s = opti.variable()
                opti.subject_to(s >= 0)
                obs_s.append(s)

                dx = (X[0, k] - obs[0]) / (obs[2] + self.margin)
                dy = (X[1, k] - obs[1]) / (obs[3] + self.margin)
                opti.subject_to(dx**2 + dy**2 + s >= 1.0)

                # Increasing penalty for later steps — escape quickly
                cost += self.slack_penalty * (1.0 + 0.5 * k) * s

            slack_vars.append(obs_s)

        # Tracking cost
        for k in range(H):
            cost += Q_POS * ca.sumsqr(X[0:2, k] - xg)
            cost += Q_VEL * ca.sumsqr(X[2:4, k])
            cost += R_ACC * ca.sumsqr(U[:, k])

        # Terminal
        cost += P_TERM * ca.sumsqr(X[0:2, H] - xg)
        cost += P_TERM * Q_VEL * ca.sumsqr(X[2:4, H])

        opti.minimize(cost)

        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.max_iter': 300,
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.tol': 1e-4,
        }
        opti.solver('ipopt', opts)

        # Store references
        self.opti = opti
        self.X = X
        self.U = U
        self.x0_param = x0
        self.xg_param = xg
        self.obs_params = obs_p
        self.slack_vars = slack_vars

    def solve(self, current_pos_2d, current_vel_2d, goal_2d=None):
        """
        Returns: (next_wp, full_path, planned_vel, solve_ms, solved)
        """
        self.solve_count += 1
        if goal_2d is None:
            goal_2d = self.goal

        x0 = np.array([current_pos_2d[0], current_pos_2d[1],
                        current_vel_2d[0], current_vel_2d[1]])
        self.opti.set_value(self.x0_param, x0)
        self.opti.set_value(self.xg_param, goal_2d)

        for i in range(self.n_obs):
            self.opti.set_value(self.obs_params[i], self.obstacles[i])

        # Warm start
        if self._prev_X is not None:
            try:
                X_init = np.hstack([self._prev_X[:, 1:], self._prev_X[:, -1:]])
                U_init = np.hstack([self._prev_U[:, 1:], self._prev_U[:, -1:]])
                self.opti.set_initial(self.X, X_init)
                self.opti.set_initial(self.U, U_init)
            except Exception:
                pass
        else:
            # Cold start: route around obstacles if straight line intersects
            for k in range(self.H + 1):
                a = k / self.H
                px = current_pos_2d[0] + a * (goal_2d[0] - current_pos_2d[0])
                py = current_pos_2d[1] + a * (goal_2d[1] - current_pos_2d[1])

                # Check if this point is inside any obstacle — if so, offset Y
                for obs in self.obstacles:
                    cx, cy, hw, hh = obs
                    if abs(px - cx) < (hw + self.margin) and abs(py - cy) < (hh + self.margin):
                        # Push sideways to avoid obstacle
                        py = cy + (hh + self.margin + 0.1) * (1 if py >= cy else -1)
                        if abs(py - cy) < 0.01:
                            py = cy + (hh + self.margin + 0.1)  # default: go positive Y

                self.opti.set_initial(self.X[0, k], px)
                self.opti.set_initial(self.X[1, k], py)
                self.opti.set_initial(self.X[2, k], 0)
                self.opti.set_initial(self.X[3, k], 0)

        try:
            t0 = time.time()
            sol = self.opti.solve()
            solve_ms = (time.time() - t0) * 1000

            X_sol = sol.value(self.X)
            U_sol = sol.value(self.U)
            self._prev_X = X_sol
            self._prev_U = U_sol

            next_wp = X_sol[0:2, 1]
            full_path = X_sol[0:2, :].T
            planned_vel = X_sol[2:4, 1]

            # Total slack
            total_slack = sum(
                sol.value(self.slack_vars[i][k])
                for i in range(self.n_obs)
                for k in range(self.H + 1)
            )

            if self.solve_count % 10 == 0 or total_slack > 0.01:
                logger.info(
                    f"[MPC] solve={solve_ms:.0f}ms H={self.H} "
                    f"wp=[{next_wp[0]:.3f},{next_wp[1]:.3f}] "
                    f"slack={total_slack:.3f}"
                )

            return next_wp, full_path, planned_vel, solve_ms, True

        except Exception as e:
            self.fail_count += 1
            self._prev_X = None
            self._prev_U = None

            # Better error reporting
            err_str = str(e)
            if 'Infeasible' in err_str or 'infeasible' in err_str:
                reason = "INFEASIBLE"
            elif 'Maximum_Iterations_Exceeded' in err_str or 'iter' in err_str.lower():
                reason = "MAX_ITER"
            else:
                reason = err_str[:80]

            if self.fail_count <= 5 or self.fail_count % 20 == 0:
                logger.warn(f"[MPC FAIL #{self.fail_count}] {reason}")
                logger.warn(f"  state=[{current_pos_2d[0]:.3f},{current_pos_2d[1]:.3f}] "
                            f"goal=[{goal_2d[0]:.3f},{goal_2d[1]:.3f}] H={self.H}")
                # Check if start is inside obstacle
                for i, obs in enumerate(self.obstacles):
                    cx, cy, hw, hh = obs
                    dx = abs(current_pos_2d[0] - cx) / (hw + self.margin)
                    dy = abs(current_pos_2d[1] - cy) / (hh + self.margin)
                    if dx**2 + dy**2 < 1.0:
                        logger.warn(f"  START IS INSIDE obstacle {i} exclusion zone!")
                # Check if horizon covers the distance
                max_reach = self.H * self.dt * self.v_max
                dist = np.linalg.norm(goal_2d - current_pos_2d)
                if dist > max_reach * 0.8:
                    logger.warn(f"  Horizon too short! dist={dist:.2f}m but "
                                f"max_reach={max_reach:.2f}m (H={self.H}, dt={self.dt}, v={self.v_max})")

            return current_pos_2d.copy(), None, np.zeros(2), 0, False


# ==============================
# CSV LOGGER
# ==============================
class FlightLogger:
    def __init__(self, filepath):
        self.filepath = filepath
        self.rows = []
        self.headers = [
            "step", "time_s",
            "pos_x", "pos_y", "pos_z",
            "goal_x", "goal_y",
            "wp_x", "wp_y",
            "dist_to_goal", "solve_ms", "solved",
        ]

    def log(self, step, t, pos_3d, goal_2d, wp_2d, dist, solve_ms, solved):
        self.rows.append([
            step, f"{t:.3f}",
            f"{pos_3d[0]:.4f}", f"{pos_3d[1]:.4f}", f"{pos_3d[2]:.4f}",
            f"{goal_2d[0]:.4f}", f"{goal_2d[1]:.4f}",
            f"{wp_2d[0]:.4f}", f"{wp_2d[1]:.4f}",
            f"{dist:.4f}", f"{solve_ms:.1f}", int(solved),
        ])

    def save(self):
        path = os.path.join(LOG_DIR, self.filepath)
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)
            writer.writerows(self.rows)
        logger.info(f"Flight log saved to {path} ({len(self.rows)} rows)")


# ==============================
# EMERGENCY LANDING (cmdPosition only)
# ==============================
def emergency_land(cf, timeHelper):
    logger.warn("EMERGENCY LANDING triggered")
    try:
        for i in range(60):
            pos = np.array(cf.position)
            if pos[2] < 0.06:
                break
            target = np.array([pos[0], pos[1], max(0.02, pos[2] - 0.01)])
            cf.cmdPosition(target, yaw=0.0)
            timeHelper.sleep(0.1)
        pos = np.array(cf.position)
        for i in range(10):
            cf.cmdPosition(np.array([pos[0], pos[1], 0.02]), yaw=0.0)
            timeHelper.sleep(0.1)
        logger.info("Emergency landing complete")
    except Exception as e:
        logger.error(f"Emergency landing error: {e}")


# ==============================
# LANDING (cmdPosition only — no mode switch)
# ==============================
def land_with_cmdposition(cf, timeHelper):
    logger.info("=== LANDING (cmdPosition) ===")
    for i in range(80):
        pos = np.array(cf.position)
        if pos[2] < 0.06:
            logger.info(f"  Near ground Z={pos[2]:.3f}")
            break
        target = np.array([pos[0], pos[1], max(0.02, pos[2] - 0.008)])
        cf.cmdPosition(target, yaw=0.0)
        timeHelper.sleep(0.1)
        if i % 15 == 0:
            logger.info(f"  Descending Z={pos[2]:.3f}")

    # Hold at ground
    pos = np.array(cf.position)
    for i in range(20):
        cf.cmdPosition(np.array([pos[0], pos[1], 0.02]), yaw=0.0)
        timeHelper.sleep(0.1)
    logger.info("Landed.")


# ==============================
# PLOTTING
# ==============================
def plot_results(traj_actual, errors, solve_times, planned_paths,
                 obstacles_3d, start_3d, goal_3d, global_plan=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Ellipse

    traj = np.array(traj_actual)
    errors = np.array(errors)
    solve_times = np.array(solve_times) if len(solve_times) > 0 else np.zeros(len(traj))
    steps = np.arange(len(traj))
    dt_ctrl = 1.0 / RATE
    time_s = steps * dt_ctrl

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # ---- 2D map ----
    ax = axes[0, 0]
    for obs in obstacles_3d:
        cx, cy, cz, hw, hh, hz = obs
        rect = Rectangle(
            (cx - hw, cy - hh), 2 * hw, 2 * hh,
            linewidth=2, edgecolor='red', facecolor='salmon', alpha=0.4,
        )
        ax.add_patch(rect)
        ax.text(cx, cy, f'{2*hw:.2f}x{2*hh:.2f}', ha='center', va='center',
                fontsize=7, color='darkred')
        ellipse = Ellipse(
            (cx, cy), 2 * (hw + OBS_MARGIN), 2 * (hh + OBS_MARGIN),
            linewidth=1, edgecolor='red', facecolor='none', linestyle=':', alpha=0.5
        )
        ax.add_patch(ellipse)

    # Global plan (if exists)
    if global_plan is not None:
        ax.plot(global_plan[:, 0], global_plan[:, 1], 'g--', linewidth=2,
                label='Planned path', zorder=3)
        ax.plot(global_plan[:, 0], global_plan[:, 1], 'g.', markersize=8, zorder=3)

    # RHC planned horizons
    if planned_paths is not None:
        for i, pp in enumerate(planned_paths):
            if pp is not None and i % 8 == 0:
                ax.plot(pp[:, 0], pp[:, 1], 'c-', alpha=0.12, linewidth=0.8)

    # Actual path
    ax.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2, label='Actual path', zorder=4)

    # Configured start (where you INTENDED to start)
    ax.plot(start_3d[0], start_3d[1], 's', color='lime', markersize=12,
            markeredgecolor='black', label='Configured start', zorder=5)
    # Actual start (where drone actually was)
    ax.plot(traj[0, 0], traj[0, 1], 'go', markersize=10,
            label='Actual start', zorder=5)
    # Goal
    ax.plot(goal_3d[0], goal_3d[1], 'r*', markersize=16, zorder=5, label='Goal')

    # Auto-scale
    all_x = [o[0] for o in obstacles_3d] + [start_3d[0], goal_3d[0], traj[0, 0]]
    all_y = [o[1] for o in obstacles_3d] + [start_3d[1], goal_3d[1], traj[0, 1]]
    pad = 0.5
    ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
    ax.set_ylim(min(all_y) - pad - 0.3, max(all_y) + pad + 0.3)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    mode_str = "Global Plan" if GLOBAL_PLAN_ONLY else "RHC"
    ax.set_title(f"2D Trajectory & Obstacles ({mode_str})")
    ax.legend(fontsize=7, loc='upper left')
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # ---- Position over time ----
    ax = axes[0, 1]
    ax.plot(time_s, traj[:, 0], 'r-', linewidth=1.2, label='X')
    ax.plot(time_s, traj[:, 1], 'g-', linewidth=1.2, label='Y')
    ax.plot(time_s, traj[:, 2], 'b-', linewidth=1.2, label='Z')
    ax.axhline(y=goal_3d[0], color='r', linestyle='--', alpha=0.3)
    ax.axhline(y=goal_3d[1], color='g', linestyle='--', alpha=0.3)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position [m]")
    ax.set_title("Position Over Time")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ---- Distance to goal ----
    ax = axes[1, 0]
    ax.plot(time_s, errors, 'k-', linewidth=1)
    ax.axhline(y=GOAL_TOLERANCE, color='g', linestyle='--', alpha=0.5,
               label=f'Tolerance ({GOAL_TOLERANCE}m)')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Distance to goal [m]")
    ax.set_title("Distance to Goal")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Solve time ----
    ax = axes[1, 1]
    if len(solve_times) > 0 and np.any(solve_times > 0):
        t_s = time_s[:len(solve_times)]
        ax.plot(t_s, solve_times, 'purple', linewidth=0.8)
        budget = 1000.0 / RATE
        ax.axhline(y=budget, color='r', linestyle='--', label=f'Budget ({budget:.0f}ms)')
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, 'Global plan mode\n(single solve)', ha='center',
                va='center', transform=ax.transAxes, fontsize=12, color='gray')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Solve time [ms]")
    ax.set_title("MPC Solve Time")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(LOG_DIR, PLOT_FILE)
    plt.savefig(plot_path, dpi=150)
    logger.info(f"Plot saved to {plot_path}")


# ==============================
# GLOBAL PLAN: solve once with auto-sized horizon
# ==============================
def compute_global_plan(start_2d, goal_2d, obstacles_2d):
    """
    Solve the MPC once from start to goal with a horizon long enough
    to cover the full distance (including detour around obstacles).
    """
    dist = np.linalg.norm(goal_2d - start_2d)

    # Auto-compute horizon: need enough steps to cover distance at V_MAX
    # with margin for obstacle detour (1.5x) and deceleration
    if GLOBAL_HORIZON_OVERRIDE is not None:
        H_global = GLOBAL_HORIZON_OVERRIDE
    else:
        # Estimate: time = dist * 1.5 / (0.7 * V_MAX), rounded up
        # The 1.5x accounts for obstacle detour, 0.7 for avg speed < max
        time_needed = dist * 1.5 / (0.7 * V_MAX)
        H_global = max(20, int(time_needed / DT_MPC) + 5)

    logger.info(f"Computing global plan: dist={dist:.2f}m, H={H_global}, "
                f"dt={DT_MPC}, reach={H_global * DT_MPC * V_MAX:.2f}m")

    mpc = DoubleIntegratorMPC(
        horizon=H_global, dt=DT_MPC,
        obstacles_2d=obstacles_2d, goal_2d=goal_2d,
        v_max=V_MAX, a_max=A_MAX, margin=OBS_MARGIN,
        slack_penalty=SLACK_PENALTY,
    )

    vel_2d = np.zeros(2)
    next_wp, full_path, vel, solve_ms, solved = mpc.solve(start_2d, vel_2d, goal_2d)

    if solved and full_path is not None:
        logger.info(f"Global plan computed in {solve_ms:.0f}ms — {len(full_path)} waypoints")
        for i, wp in enumerate(full_path):
            logger.info(f"  WP[{i:2d}] = [{wp[0]:.3f}, {wp[1]:.3f}]")
        return full_path, mpc
    else:
        logger.error("Global plan FAILED!")
        logger.error(f"  Try: increase V_MAX, reduce distance, check obstacle isn't blocking entirely")
        return None, None


# ==============================
# MAIN
# ==============================
def main():
    logger.info("=" * 60)
    logger.info("RECEDING HORIZON CONTROL — Double Integrator MPC")
    logger.info(f"  Mode:      {'GLOBAL PLAN (open-loop)' if GLOBAL_PLAN_ONLY else 'RHC (closed-loop)'}")
    logger.info(f"  Start:     {START_POSITION}")
    logger.info(f"  Goal:      {GOAL_POSITION}")
    logger.info(f"  Obstacles: {len(OBSTACLES)}")
    for i, obs in enumerate(OBSTACLES):
        logger.info(f"    [{i}] center=({obs[0]:.2f},{obs[1]:.2f},{obs[2]:.2f}) "
                     f"size=({2*obs[3]:.2f}x{2*obs[4]:.2f}x{2*obs[5]:.2f}m)")
    logger.info(f"  H={HORIZON} dt={DT_MPC} V_max={V_MAX} A_max={A_MAX} margin={OBS_MARGIN}")
    logger.info(f"  Q_pos={Q_POS} Q_vel={Q_VEL} R_acc={R_ACC} P_term={P_TERM} slack={SLACK_PENALTY}")
    logger.info("=" * 60)

    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    cf = swarm.allcfs.crazyflies[0]

    methods = [m for m in dir(cf) if m.startswith('cmd') and callable(getattr(cf, m))]
    logger.info(f"Available cmd* methods: {methods}")

    # Signal handler
    def signal_handler(sig, frame):
        logger.warn("Ctrl+C received!")
        emergency_land(cf, timeHelper)
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Build MPC
    mpc = DoubleIntegratorMPC(
        horizon=HORIZON, dt=DT_MPC,
        obstacles_2d=OBSTACLES_2D, goal_2d=GOAL_2D,
        v_max=V_MAX, a_max=A_MAX, margin=OBS_MARGIN,
        slack_penalty=SLACK_PENALTY,
    )

    # ============================================================
    # MODE 1: GLOBAL PLAN (solve before flight, execute open-loop)
    # ============================================================
    if GLOBAL_PLAN_ONLY:
        # Compute plan from configured start
        global_path, global_mpc = compute_global_plan(START_2D, GOAL_2D, OBSTACLES_2D)
        if global_path is None:
            logger.error("Cannot fly — no valid plan. Exiting.")
            return

        # Ground check
        logger.info("--- Ground check ---")
        for i in range(5):
            pos = np.array(cf.position)
            logger.info(f"  [{i}] pos=[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
            timeHelper.sleep(0.2)

        # Warn if actual position differs significantly from configured start
        actual_start = np.array(cf.position[0:2])
        start_offset = np.linalg.norm(actual_start - START_2D)
        if start_offset > 0.15:
            logger.warn(f"Actual position {actual_start} differs from configured "
                        f"start {START_2D} by {start_offset:.2f}m!")
            logger.warn("The global plan was computed from configured start. "
                        "Drone will fly to first waypoint.")

        # Takeoff
        logger.info(f"Taking off to {FLIGHT_HEIGHT}m...")
        cf.takeoff(targetHeight=FLIGHT_HEIGHT, duration=3.0)
        timeHelper.sleep(3.5)

        logger.info("--- Hover check ---")
        for i in range(5):
            pos = np.array(cf.position)
            logger.info(f"  [{i}] pos=[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
            timeHelper.sleep(0.2)

        # Execute waypoints
        flight_log = FlightLogger(LOG_CSV)
        traj_actual = []
        errors = []
        t_start = time.time()

        # Each waypoint is held for DT_MPC seconds (the time between
        # consecutive MPC steps). cmdPosition must be sent repeatedly
        # at ~10Hz to keep the drone in streaming mode.
        CMD_RATE = 10.0  # Hz — how often to re-send cmdPosition
        dt_cmd = 1.0 / CMD_RATE
        holds_per_wp = max(1, int(DT_MPC * CMD_RATE))  # send cmds per waypoint

        logger.info(f"Executing {len(global_path)} waypoints "
                    f"(hold {DT_MPC}s each = {holds_per_wp} cmds at {CMD_RATE}Hz)...")

        try:
            for wi, wp in enumerate(global_path):
                target_3d = np.array([wp[0], wp[1], FLIGHT_HEIGHT])

                # Hold this waypoint for DT_MPC seconds
                for hold in range(holds_per_wp):
                    cf.cmdPosition(target_3d, yaw=0.0)
                    timeHelper.sleep(dt_cmd)

                    # Log position on first and last hold of each waypoint
                    pos_3d = np.array(cf.position)
                    traj_actual.append(pos_3d.copy())
                    dist = np.linalg.norm(pos_3d[0:2] - GOAL_2D)
                    errors.append(dist)

                # Log once per waypoint
                pos_3d = np.array(cf.position)
                t_now = time.time() - t_start
                dist = np.linalg.norm(pos_3d[0:2] - GOAL_2D)

                flight_log.log(wi, t_now, pos_3d, GOAL_2D, wp, dist, 0, True)

                logger.info(
                    f"  WP[{wi:2d}/{len(global_path)-1}] t={t_now:5.1f}s "
                    f"target=[{wp[0]:+.3f},{wp[1]:+.3f}] "
                    f"actual=[{pos_3d[0]:+.3f},{pos_3d[1]:+.3f}] "
                    f"err={np.linalg.norm(pos_3d[0:2] - wp):.3f} dist_goal={dist:.3f}"
                )

            # Hover at goal
            logger.info("Hovering at goal for 3s...")
            goal_3d = np.array([GOAL_2D[0], GOAL_2D[1], FLIGHT_HEIGHT])
            for i in range(int(3.0 * CMD_RATE)):
                cf.cmdPosition(goal_3d, yaw=0.0)
                timeHelper.sleep(dt_cmd)
                pos_3d = np.array(cf.position)
                traj_actual.append(pos_3d.copy())
                errors.append(np.linalg.norm(pos_3d[0:2] - GOAL_2D))

            # Land
            land_with_cmdposition(cf, timeHelper)

        except Exception as e:
            logger.error(f"Exception: {e}")
            emergency_land(cf, timeHelper)

        # Save & plot
        flight_log.save()
        errors_arr = np.array(errors)
        logger.info("===== FLIGHT SUMMARY =====")
        logger.info(f"  Mode: Global Plan (open-loop)")
        logger.info(f"  Waypoints: {len(global_path)}")
        logger.info(f"  Final distance to goal: {errors_arr[-1]:.3f}m")

        try:
            plot_results(traj_actual, errors, [], None,
                         OBSTACLES, START_POSITION, GOAL_POSITION,
                         global_plan=global_path)
        except Exception as e:
            logger.warn(f"Plotting failed: {e}")

        return

    # ============================================================
    # MODE 2: RECEDING HORIZON (closed-loop, replan every step)
    # ============================================================
    dt = 1.0 / RATE

    # Ground check
    logger.info("--- Ground check ---")
    for i in range(5):
        pos = np.array(cf.position)
        logger.info(f"  [{i}] pos=[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
        timeHelper.sleep(0.2)

    # Takeoff
    logger.info(f"Taking off to {FLIGHT_HEIGHT}m...")
    cf.takeoff(targetHeight=FLIGHT_HEIGHT, duration=3.0)
    timeHelper.sleep(3.5)

    logger.info("--- Hover check ---")
    for i in range(5):
        pos = np.array(cf.position)
        logger.info(f"  [{i}] pos=[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
        timeHelper.sleep(0.2)

    flight_log = FlightLogger(LOG_CSV)
    traj_actual = []
    errors = []
    solve_times = []
    planned_paths = []
    prev_pos_2d = np.array(cf.position[0:2])
    t_start = time.time()

    logger.info("Starting RHC loop...")

    try:
        for step in range(MAX_STEPS):
            pos_3d = np.array(cf.position)
            pos_2d = pos_3d[0:2]
            t_now = time.time() - t_start

            vel_2d = (pos_2d - prev_pos_2d) / dt if step > 0 else np.zeros(2)
            dist = np.linalg.norm(pos_2d - GOAL_2D)

            errors.append(dist)
            traj_actual.append(pos_3d.copy())

            if dist < GOAL_TOLERANCE:
                logger.info(f"GOAL REACHED at step {step} (t={t_now:.1f}s)! dist={dist:.3f}m")
                flight_log.log(step, t_now, pos_3d, GOAL_2D, pos_2d, dist, 0, True)
                break

            next_wp, planned_path, planned_vel, solve_ms, solved = mpc.solve(
                pos_2d, vel_2d, GOAL_2D
            )
            solve_times.append(solve_ms)
            planned_paths.append(planned_path)

            flight_log.log(step, t_now, pos_3d, GOAL_2D, next_wp, dist, solve_ms, solved)

            if step % 10 == 0:
                logger.info(
                    f"[{step:4d}] t={t_now:5.1f}s "
                    f"pos=[{pos_2d[0]:+.3f},{pos_2d[1]:+.3f}] "
                    f"wp=[{next_wp[0]:+.3f},{next_wp[1]:+.3f}] "
                    f"dist={dist:.3f} solve={solve_ms:.0f}ms "
                    f"{'OK' if solved else 'FAIL'}"
                )

            target_3d = np.array([next_wp[0], next_wp[1], FLIGHT_HEIGHT])
            cf.cmdPosition(target_3d, yaw=0.0)

            prev_pos_2d = pos_2d.copy()
            timeHelper.sleepForRate(RATE)

        # Hover at goal
        logger.info("Hovering at goal for 2s...")
        goal_3d = np.array([GOAL_2D[0], GOAL_2D[1], FLIGHT_HEIGHT])
        for i in range(int(2.0 * RATE)):
            cf.cmdPosition(goal_3d, yaw=0.0)
            timeHelper.sleep(dt)
            pos_3d = np.array(cf.position)
            traj_actual.append(pos_3d.copy())
            errors.append(np.linalg.norm(pos_3d[0:2] - GOAL_2D))

        # Land
        land_with_cmdposition(cf, timeHelper)

    except Exception as e:
        logger.error(f"Exception: {e}")
        emergency_land(cf, timeHelper)

    # Save & plot
    flight_log.save()

    if len(errors) > 0:
        err = np.array(errors)
        logger.info("===== FLIGHT SUMMARY =====")
        logger.info(f"  Mode: RHC (closed-loop)")
        logger.info(f"  Steps: {len(err)}")
        logger.info(f"  Goal reached: {err[-1] < GOAL_TOLERANCE}")
        logger.info(f"  Final distance: {err[-1]:.3f}m")
        logger.info(f"  MPC — solves: {mpc.solve_count}, fails: {mpc.fail_count}")
        if len(solve_times) > 0:
            st = np.array(solve_times)
            logger.info(f"  Solve time — mean: {st.mean():.0f}ms, "
                        f"max: {st.max():.0f}ms, median: {np.median(st):.0f}ms")

        try:
            plot_results(traj_actual, errors, solve_times, planned_paths,
                         OBSTACLES, START_POSITION, GOAL_POSITION)
        except Exception as e:
            logger.warn(f"Plotting failed: {e}")


if __name__ == "__main__":
    main()