#!/usr/bin/env python3
"""
Week 3: Closed-Loop Receding Horizon Control (RHC / MPC)
=========================================================
Course  : SC 627 - Motion Planning & Coordination of Autonomous Vehicles

Fixes applied vs original code 2:
  1. DT_MPC = 0.3  (was 0.1) — X_pred[1] is now a real waypoint ~0.1-0.3m away
     instead of ~0.01m which the Crazyflie position controller couldn't track.
  2. Velocity estimate clamped — prevents Kalman filter divergence from
     noisy finite-difference spikes.
  3. Obstacle avoidance replaced: exponential soft penalty removed and replaced
     with ellipsoidal soft-slack constraints (same as working code 1).
     Exponential penalty was too weak — goal cost overwhelmed it causing the
     drone to try flying straight through the obstacle ("dash toward wall").
  4. Removed Mocap dependency — obstacle set statically, state from cf.position.
  5. RATE tied to DT_MPC so one cmdPosition is issued per MPC solve step.

Plots added:
  Fig 1 — XY plane: actual trajectory, MPC predicted paths, obstacle, start/goal
  Fig 2 — MPC solve time (ms) per step
  Fig 3 — Distance to goal (m) per step
  Fig 4 — Position in all 3 dimensions (x, y, z) vs step
"""

import sys
import signal
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless; change to "TkAgg" if you have a display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse

try:
    import casadi as ca
except ImportError:
    raise ImportError("CasADi is required:  pip install casadi")

try:
    import rclpy
    from rclpy.logging import get_logger
    from crazyflie_py import Crazyswarm
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False


# ==============================================================================
#                         USER CONFIGURATION
# ==============================================================================

START_POSITION  = [0.0,  0.0, 0.5]
GOAL_POSITION   = [-2.0, 0.0, 0.5]

# Obstacle [cx, cy, half_width_x, half_width_y]
OBSTACLE        = [-1.0, 0.0, 0.125, 0.45]
OBS_MARGIN      = 0.15      # safety margin added to obstacle half-widths (m)

# MPC
HORIZON         = 15
DT_MPC          = 0.3       # FIX 1: was 0.1 — 0.3 gives ~0.1-0.3m waypoint steps
V_MAX           = 0.4
A_MAX           = 0.8

# Cost weights
Q_DIAG          = [10.0, 10.0, 1.0, 1.0]
R_DIAG          = [0.1,  0.1]
P_SCALE         = 10.0
SLACK_PENALTY   = 500.0     # FIX 3: replaces weak exponential penalty

# Flight
RATE            = 1.0 / DT_MPC   # FIX: one cmdPosition per MPC step (~3.3 Hz)
MAX_STEPS       = 300
GOAL_TOLERANCE  = 0.10

# Plot output path
PLOT_OUTPUT_DIR = "."   # directory where PNG files are saved

# ==============================================================================

FLIGHT_HEIGHT   = START_POSITION[2]
GOAL_2D         = np.array(GOAL_POSITION[:2])
OBSTACLE_2D     = OBSTACLE   # [cx, cy, hw_x, hw_y]


# ═════════════════════════════════════════════════════════════════════════════
#  MPC CORE
# ═════════════════════════════════════════════════════════════════════════════

class MPCController:
    """
    2-D double integrator MPC.

    Obstacle avoidance via SOFT ELLIPSOIDAL SLACK CONSTRAINTS  (FIX 3):
        (dx/hw_x)^2 + (dy/hw_y)^2 + s >= 1,   s >= 0
        cost += SLACK_PENALTY * s
    This always keeps the problem feasible AND actually steers away from
    the obstacle, unlike the exponential penalty that gets overwhelmed.
    """

    def __init__(self, dt, H, obstacle_2d, margin,
                 v_max, a_max, slack_penalty,
                 Q=None, R=None, P_scale=10.0):

        self.dt   = dt
        self.H    = H
        self.v_max = v_max
        self.a_max = a_max

        # Discrete-time double integrator
        self.A = np.array([
            [1, 0, dt,  0],
            [0, 1,  0, dt],
            [0, 0,  1,  0],
            [0, 0,  0,  1],
        ], dtype=float)
        self.B = np.array([
            [0.5*dt**2, 0         ],
            [0,         0.5*dt**2 ],
            [dt,        0         ],
            [0,         dt        ],
        ], dtype=float)

        self.Q = Q if Q is not None else np.diag(Q_DIAG)
        self.R = R if R is not None else np.diag(R_DIAG)
        self.P = P_scale * self.Q

        self.obstacle = obstacle_2d   # [cx, cy, hw_x, hw_y]
        self.margin   = margin
        self.slack_penalty = slack_penalty

        self._U0      = None   # warm-start
        self._build_nlp()

    def _build_nlp(self):
        H, dt = self.H, self.dt
        nx, nu = 4, 2
        opti = ca.Opti()

        X  = opti.variable(nx, H + 1)   # [px, py, vx, vy]
        U  = opti.variable(nu, H)        # [ax, ay]
        x0 = opti.parameter(nx)
        xg = opti.parameter(nx)

        A_dm = ca.DM(self.A)
        B_dm = ca.DM(self.B)

        # Dynamics
        for k in range(H):
            opti.subject_to(X[:, k+1] == A_dm @ X[:, k] + B_dm @ U[:, k])
        opti.subject_to(X[:, 0] == x0)

        # Bounds
        for k in range(H + 1):
            opti.subject_to(opti.bounded(-self.v_max, X[2, k], self.v_max))
            opti.subject_to(opti.bounded(-self.v_max, X[3, k], self.v_max))
        for k in range(H):
            opti.subject_to(opti.bounded(-self.a_max, U[0, k], self.a_max))
            opti.subject_to(opti.bounded(-self.a_max, U[1, k], self.a_max))

        # ── FIX 3: Soft ellipsoidal obstacle constraint ────────────────────
        cx, cy, hw_x, hw_y = self.obstacle
        eff_hx = hw_x + self.margin
        eff_hy = hw_y + self.margin

        cost = ca.MX(0)
        for k in range(H + 1):
            s = opti.variable()
            opti.subject_to(s >= 0)
            dx = (X[0, k] - cx) / eff_hx
            dy = (X[1, k] - cy) / eff_hy
            opti.subject_to(dx**2 + dy**2 + s >= 1.0)
            # Increase penalty for later steps so drone escapes obstacle quickly
            cost += self.slack_penalty * (1.0 + 0.3 * k) * s

        # Stage + terminal cost
        Q_dm = ca.DM(self.Q)
        R_dm = ca.DM(self.R)
        P_dm = ca.DM(self.P)
        for k in range(H):
            ex = X[:, k] - xg
            cost += ex.T @ Q_dm @ ex + U[:, k].T @ R_dm @ U[:, k]
        ex_H = X[:, H] - xg
        cost += ex_H.T @ P_dm @ ex_H

        opti.minimize(cost)

        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.max_iter': 300,
            'ipopt.tol': 1e-4,
            'ipopt.warm_start_init_point': 'yes',
        }
        opti.solver('ipopt', opts)

        self.opti = opti
        self.X = X; self.U = U
        self.x0_p = x0; self.xg_p = xg
        self._nx = nx; self._nu = nu

    def solve(self, pos_2d, vel_2d, goal_2d):
        """Returns (next_wp_2d, full_path, solve_ms, success)."""
        x0_val = np.array([pos_2d[0], pos_2d[1], vel_2d[0], vel_2d[1]])
        xg_val = np.array([goal_2d[0], goal_2d[1], 0.0, 0.0])

        self.opti.set_value(self.x0_p, x0_val)
        self.opti.set_value(self.xg_p, xg_val)

        if self._U0 is not None:
            try:
                X_ws = np.hstack([self._U0['X'][:, 1:], self._U0['X'][:, -1:]])
                U_ws = np.hstack([self._U0['U'][:, 1:], self._U0['U'][:, -1:]])
                self.opti.set_initial(self.X, X_ws)
                self.opti.set_initial(self.U, U_ws)
            except Exception:
                pass
        else:
            for k in range(self.H + 1):
                a  = k / self.H
                px = pos_2d[0] + a * (goal_2d[0] - pos_2d[0])
                py = pos_2d[1] + a * (goal_2d[1] - pos_2d[1])
                cx, cy, hw_x, hw_y = self.obstacle
                if (abs(px - cx) < (hw_x + self.margin) and
                        abs(py - cy) < (hw_y + self.margin)):
                    py = cy + (hw_y + self.margin + 0.1)
                self.opti.set_initial(self.X[0, k], px)
                self.opti.set_initial(self.X[1, k], py)
                self.opti.set_initial(self.X[2, k], 0)
                self.opti.set_initial(self.X[3, k], 0)

        try:
            t0       = time.time()
            sol      = self.opti.solve()
            solve_ms = (time.time() - t0) * 1000

            X_sol = sol.value(self.X)
            U_sol = sol.value(self.U)
            self._U0 = {'X': X_sol, 'U': U_sol}

            next_wp = X_sol[0:2, 1]          # predicted pos after one DT step
            path    = X_sol[0:2, :].T
            return next_wp, path, solve_ms, True

        except Exception:
            self._U0 = None
            return pos_2d.copy(), None, 0.0, False


# ═════════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ═════════════════════════════════════════════════════════════════════════════

def _obs_patches(obstacle, margin, alpha_obs=0.55, alpha_margin=0.25):
    """Return (obstacle_patch, margin_patch) as matplotlib Ellipse objects."""
    cx, cy, hw_x, hw_y = obstacle
    obs = Ellipse(
        xy=(cx, cy), width=2*hw_x, height=2*hw_y,
        angle=0, facecolor="#e74c3c", edgecolor="#c0392b",
        linewidth=1.5, alpha=alpha_obs, zorder=3, label="Obstacle"
    )
    mrg = Ellipse(
        xy=(cx, cy), width=2*(hw_x+margin), height=2*(hw_y+margin),
        angle=0, facecolor="none", edgecolor="#e74c3c",
        linewidth=1.2, linestyle="--", alpha=0.8, zorder=3, label="Obs + margin"
    )
    return obs, mrg


def save_plots(log, obstacle, margin, start, goal, output_dir="."):
    """
    Produce and save four diagnostic figures.

    Parameters
    ----------
    log : dict with keys
        steps       – list[int]
        positions   – list of (x, y, z) tuples
        waypoints   – list of (x, y) tuples
        mpc_paths   – list of (N×2) arrays or None  (one per step)
        solve_ms    – list[float]
        dist        – list[float]
    obstacle : [cx, cy, hw_x, hw_y]
    margin   : float
    start    : [x, y, z]
    goal     : [x, y, z] or [x, y]
    output_dir : str  – directory for saved PNGs
    """

    steps      = np.array(log["steps"])
    positions  = np.array(log["positions"])   # (N, 3)
    waypoints  = np.array(log["waypoints"])   # (N, 2)
    solve_ms   = np.array(log["solve_ms"])    # (N,)
    dist       = np.array(log["dist"])        # (N,)
    mpc_paths  = log["mpc_paths"]             # list of arrays or None

    px, py, pz = positions[:, 0], positions[:, 1], positions[:, 2]

    STYLE = dict(fontsize=11)
    plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.35,
                         "figure.dpi": 130})

    # ── Figure 1: XY trajectory + obstacle ───────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(8, 6))

    # MPC predicted horizons (every 5th step, faint)
    for i, path in enumerate(mpc_paths):
        if path is not None and i % 5 == 0:
            ax1.plot(path[:, 0], path[:, 1],
                     color="#3498db", alpha=0.18, linewidth=0.9, zorder=2)

    # Actual trajectory
    ax1.plot(px, py, color="#2ecc71", linewidth=2.2,
             label="Actual trajectory", zorder=4)
    ax1.plot(waypoints[:, 0], waypoints[:, 1], ".",
             color="#27ae60", markersize=3, alpha=0.6, label="Waypoints", zorder=4)

    # Start / goal markers
    ax1.plot(*start[:2], "go", markersize=12, label="Start", zorder=5)
    ax1.plot(*goal[:2],  "r*", markersize=14, label="Goal",  zorder=5)

    # Obstacle patches
    obs_patch, mrg_patch = _obs_patches(obstacle, margin)
    ax1.add_patch(obs_patch)
    ax1.add_patch(mrg_patch)

    ax1.set_xlabel("x  (m)", **STYLE)
    ax1.set_ylabel("y  (m)", **STYLE)
    ax1.set_title("Fig 1 — XY Trajectory with Obstacle", **STYLE)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_aspect("equal")
    fig1.tight_layout()
    fig1.savefig(f"{output_dir}/fig1_xy_plan.png")
    plt.close(fig1)

    # ── Figure 2: MPC solve time ──────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(8, 4))

    ax2.plot(steps, solve_ms, color="#9b59b6", linewidth=1.5, label="Solve time")
    ax2.axhline(np.mean(solve_ms), color="#8e44ad", linestyle="--",
                linewidth=1.2, label=f"Mean = {np.mean(solve_ms):.1f} ms")
    ax2.fill_between(steps, solve_ms, alpha=0.15, color="#9b59b6")

    ax2.set_xlabel("Step", **STYLE)
    ax2.set_ylabel("Solve time  (ms)", **STYLE)
    ax2.set_title("Fig 2 — MPC Solve Time per Step", **STYLE)
    ax2.legend(fontsize=9)
    fig2.tight_layout()
    fig2.savefig(f"{output_dir}/fig2_solve_time.png")
    plt.close(fig2)

    # ── Figure 3: Distance to goal ────────────────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(8, 4))

    ax3.plot(steps, dist, color="#e67e22", linewidth=1.8, label="Distance to goal")
    ax3.axhline(GOAL_TOLERANCE, color="#c0392b", linestyle="--",
                linewidth=1.2, label=f"Tolerance = {GOAL_TOLERANCE} m")
    ax3.fill_between(steps, dist, GOAL_TOLERANCE,
                     where=(np.array(dist) > GOAL_TOLERANCE),
                     alpha=0.12, color="#e67e22")

    ax3.set_xlabel("Step", **STYLE)
    ax3.set_ylabel("Distance  (m)", **STYLE)
    ax3.set_title("Fig 3 — Distance to Goal", **STYLE)
    ax3.legend(fontsize=9)
    fig3.tight_layout()
    fig3.savefig(f"{output_dir}/fig3_dist_to_goal.png")
    plt.close(fig3)

    # ── Figure 4: All 3 dimensions vs step ───────────────────────────────────
    fig4, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    labels_colors = [("x", "#2980b9"), ("y", "#27ae60"), ("z", "#c0392b")]
    data_arr = [px, py, pz]

    for ax, (lbl, col), dat in zip(axes, labels_colors, data_arr):
        ax.plot(steps, dat, color=col, linewidth=1.8, label=lbl)
        ax.fill_between(steps, dat, alpha=0.12, color=col)
        ax.set_ylabel(f"{lbl}  (m)", **STYLE)
        ax.legend(loc="upper right", fontsize=9)

    axes[-1].set_xlabel("Step", **STYLE)
    axes[0].set_title("Fig 4 — Position in All 3 Dimensions", **STYLE)
    fig4.tight_layout()
    fig4.savefig(f"{output_dir}/fig4_positions_3d.png")
    plt.close(fig4)

    print(f"\n[Plots] Saved to {output_dir}/fig1_xy_plan.png  (and fig2, fig3, fig4)")


# ═════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def emergency_land(cf, timeHelper, logger):
    logger.warn("EMERGENCY LANDING")
    try:
        for _ in range(60):
            pos = np.array(cf.position)
            if pos[2] < 0.06:
                break
            cf.cmdPosition(np.array([pos[0], pos[1], max(0.02, pos[2] - 0.01)]), yaw=0.0)
            timeHelper.sleep(0.1)
        pos = np.array(cf.position)
        for _ in range(10):
            cf.cmdPosition(np.array([pos[0], pos[1], 0.02]), yaw=0.0)
            timeHelper.sleep(0.1)
    except Exception as e:
        logger.error(f"Emergency land error: {e}")


def land_with_cmdposition(cf, timeHelper, logger):
    logger.info("=== LANDING ===")
    for i in range(80):
        pos = np.array(cf.position)
        if pos[2] < 0.06:
            break
        cf.cmdPosition(np.array([pos[0], pos[1], max(0.02, pos[2] - 0.008)]), yaw=0.0)
        timeHelper.sleep(0.1)
        if i % 15 == 0:
            logger.info(f"  Z={pos[2]:.3f}")
    pos = np.array(cf.position)
    for _ in range(20):
        cf.cmdPosition(np.array([pos[0], pos[1], 0.02]), yaw=0.0)
        timeHelper.sleep(0.1)
    logger.info("Landed.")


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    logger = get_logger("mpc_rhc")

    logger.info("=" * 60)
    logger.info(f"  Start:    {START_POSITION}")
    logger.info(f"  Goal:     {GOAL_POSITION}")
    logger.info(f"  Obstacle: {OBSTACLE}  margin={OBS_MARGIN}")
    logger.info(f"  H={HORIZON}  DT={DT_MPC}  V_max={V_MAX}  A_max={A_MAX}")
    logger.info(f"  RATE={RATE:.2f} Hz  slack_penalty={SLACK_PENALTY}")
    logger.info("=" * 60)

    mpc = MPCController(
        dt=DT_MPC, H=HORIZON,
        obstacle_2d=OBSTACLE_2D, margin=OBS_MARGIN,
        v_max=V_MAX, a_max=A_MAX,
        slack_penalty=SLACK_PENALTY,
    )

    swarm      = Crazyswarm()
    timeHelper = swarm.timeHelper
    cf         = swarm.allcfs.crazyflies[0]

    # ── Telemetry log (collected for post-flight plots) ───────────────────────
    log = {
        "steps":      [],
        "positions":  [],   # (x, y, z)
        "waypoints":  [],   # (x, y)  — MPC next-step target
        "mpc_paths":  [],   # full horizon path (N×2) or None on failure
        "solve_ms":   [],
        "dist":       [],
    }

    def signal_handler(sig, frame):
        logger.warn("Ctrl+C — emergency landing")
        emergency_land(cf, timeHelper, logger)
        # Still save whatever data we collected
        if log["steps"]:
            save_plots(log, OBSTACLE_2D, OBS_MARGIN,
                       START_POSITION, GOAL_POSITION, PLOT_OUTPUT_DIR)
        sys.exit(0)

    signal.signal(signal.SIGINT,  signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Ground check
    logger.info("--- Ground check ---")
    for i in range(5):
        pos = np.array(cf.position)
        logger.info(f"  [{i}] {pos}")
        timeHelper.sleep(0.2)

    # Takeoff
    logger.info(f"Taking off to {FLIGHT_HEIGHT}m ...")
    cf.takeoff(targetHeight=FLIGHT_HEIGHT, duration=3.0)
    timeHelper.sleep(3.5)

    logger.info("--- Hover check ---")
    for i in range(5):
        pos = np.array(cf.position)
        logger.info(f"  [{i}] {pos}")
        timeHelper.sleep(0.2)

    # ── RHC loop ──────────────────────────────────────────────────────────────
    dt_loop  = 1.0 / RATE   # = DT_MPC, so one solve per step
    prev_pos = np.array(cf.position[:2])

    logger.info("Starting RHC loop ...")

    try:
        for step in range(MAX_STEPS):
            pos_3d = np.array(cf.position)
            pos_2d = pos_3d[:2]

            # FIX 2: clamp velocity estimate to avoid Kalman spikes
            raw_vel = (pos_2d - prev_pos) / dt_loop if step > 0 else np.zeros(2)
            vel_2d  = np.clip(raw_vel, -V_MAX, V_MAX)

            dist = np.linalg.norm(pos_2d - GOAL_2D)

            # ── Log telemetry ────────────────────────────────────────────────
            log["steps"].append(step)
            log["positions"].append(tuple(pos_3d))
            log["dist"].append(float(dist))

            if dist < GOAL_TOLERANCE:
                logger.info(f"GOAL REACHED at step {step}  dist={dist:.3f}m")
                # Fill waypoint/path/solve_ms with last known values so arrays
                # stay aligned
                log["waypoints"].append(tuple(pos_2d))
                log["mpc_paths"].append(None)
                log["solve_ms"].append(0.0)
                break

            result = mpc.solve(pos_2d, vel_2d, GOAL_2D)
            # solve() returns (next_wp, path, solve_ms, success)
            next_wp, path, solve_ms, success = result

            # ── Store MPC outputs ────────────────────────────────────────────
            log["waypoints"].append(tuple(next_wp))
            log["mpc_paths"].append(path)
            log["solve_ms"].append(float(solve_ms))

            if not success:
                logger.warn(f"[{step}] MPC failed — holding position")
                next_wp = pos_2d   # hold

            target_3d = np.array([next_wp[0], next_wp[1], FLIGHT_HEIGHT])
            cf.cmdPosition(target_3d, yaw=0.0)

            if step % 5 == 0:
                logger.info(
                    f"[{step:4d}] pos=[{pos_2d[0]:+.3f},{pos_2d[1]:+.3f}]  "
                    f"wp=[{next_wp[0]:+.3f},{next_wp[1]:+.3f}]  "
                    f"dist={dist:.3f}  {solve_ms:.0f}ms  "
                    f"{'OK' if success else 'FAIL'}"
                )

            prev_pos = pos_2d.copy()
            timeHelper.sleepForRate(RATE)

        # Hover at goal
        logger.info("Hovering at goal for 2s ...")
        goal_3d = np.array([GOAL_2D[0], GOAL_2D[1], FLIGHT_HEIGHT])
        for _ in range(int(2.0 * RATE)):
            cf.cmdPosition(goal_3d, yaw=0.0)
            timeHelper.sleep(dt_loop)

        land_with_cmdposition(cf, timeHelper, logger)

    except Exception as e:
        logger.error(f"Exception: {e}")
        emergency_land(cf, timeHelper, logger)

    finally:
        # ── Generate all four plots once the drone is safely on the ground ────
        if log["steps"]:
            logger.info("Generating diagnostic plots ...")
            save_plots(log, OBSTACLE_2D, OBS_MARGIN,
                       START_POSITION, GOAL_POSITION, PLOT_OUTPUT_DIR)
        else:
            logger.warn("No data logged — skipping plots.")


if __name__ == "__main__":
    main()