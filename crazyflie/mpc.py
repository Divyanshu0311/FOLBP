#!/usr/bin/env python3
"""ROS 2 node — Crazyflie + MPC, waits for service calls to take off / fly to a
setpoint / land. No obstacles (free space).

This is the drone-side counterpart of mission_controller.py. PEFA never talks
to it directly; drone_web_client.py is the HTTP bridge that translates
PEFA actions into the services exposed here.

Services (all std_srvs/Empty; block until done):
    /drone/takeoff             take off to flight_height
    /drone/fly_to              MPC-fly to the latest /drone/target (x, y) in m
    /drone/land                descend at current xy to ground

Topic for arguments:
    /drone/target (geometry_msgs/Pose2D, m)   — x, y in metres; theta ignored

Quick test from CLI:
    ros2 service call /drone/takeoff std_srvs/srv/Empty
    ros2 topic pub --once /drone/target geometry_msgs/Pose2D '{x: -1.0, y: 0.0, theta: 0.0}'
    ros2 service call /drone/fly_to std_srvs/srv/Empty
    ros2 service call /drone/land std_srvs/srv/Empty
"""

import math
import queue
import sys
import threading
import time

import numpy as np

try:
    import casadi as ca
except ImportError:
    raise ImportError("CasADi required: pip install casadi")

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Pose2D
from std_srvs.srv import Empty

from crazyflie_py import Crazyswarm  # initialises rclpy internally


# ============================================================================
#  MPC — 2-D double integrator, NO obstacle (free flight)
# ============================================================================

class MPCController:
    """Closed-loop MPC for a 2-D double integrator. Free space — no obstacle."""

    def __init__(self, dt=0.3, H=15, v_max=0.4, a_max=0.8,
                 Q_diag=(10.0, 10.0, 1.0, 1.0), R_diag=(0.1, 0.1), P_scale=10.0):
        self.dt = dt
        self.H = H
        self.v_max = v_max
        self.a_max = a_max

        self.A = np.array([
            [1, 0, dt,  0],
            [0, 1,  0, dt],
            [0, 0,  1,  0],
            [0, 0,  0,  1],
        ], dtype=float)
        self.B = np.array([
            [0.5*dt**2, 0],
            [0, 0.5*dt**2],
            [dt, 0],
            [0, dt],
        ], dtype=float)
        self.Q = np.diag(Q_diag)
        self.R = np.diag(R_diag)
        self.P = P_scale * self.Q
        self._U0 = None
        self._build_nlp()

    def _build_nlp(self):
        H, dt = self.H, self.dt
        nx, nu = 4, 2
        opti = ca.Opti()
        X = opti.variable(nx, H + 1)
        U = opti.variable(nu, H)
        x0 = opti.parameter(nx)
        xg = opti.parameter(nx)
        A_dm, B_dm = ca.DM(self.A), ca.DM(self.B)

        for k in range(H):
            opti.subject_to(X[:, k+1] == A_dm @ X[:, k] + B_dm @ U[:, k])
        opti.subject_to(X[:, 0] == x0)

        for k in range(H + 1):
            opti.subject_to(opti.bounded(-self.v_max, X[2, k], self.v_max))
            opti.subject_to(opti.bounded(-self.v_max, X[3, k], self.v_max))
        for k in range(H):
            opti.subject_to(opti.bounded(-self.a_max, U[0, k], self.a_max))
            opti.subject_to(opti.bounded(-self.a_max, U[1, k], self.a_max))

        cost = ca.MX(0)
        Q_dm, R_dm, P_dm = ca.DM(self.Q), ca.DM(self.R), ca.DM(self.P)
        for k in range(H):
            ex = X[:, k] - xg
            cost += ex.T @ Q_dm @ ex + U[:, k].T @ R_dm @ U[:, k]
        ex_H = X[:, H] - xg
        cost += ex_H.T @ P_dm @ ex_H

        opti.minimize(cost)
        opti.solver('ipopt', {
            'ipopt.print_level': 0, 'print_time': 0,
            'ipopt.max_iter': 200, 'ipopt.tol': 1e-4,
            'ipopt.warm_start_init_point': 'yes',
        })
        self.opti = opti
        self.X, self.U = X, U
        self.x0_p, self.xg_p = x0, xg

    def solve(self, pos_2d, vel_2d, goal_2d):
        x0 = np.array([pos_2d[0], pos_2d[1], vel_2d[0], vel_2d[1]])
        xg = np.array([goal_2d[0], goal_2d[1], 0.0, 0.0])
        self.opti.set_value(self.x0_p, x0)
        self.opti.set_value(self.xg_p, xg)

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
                a = k / self.H
                self.opti.set_initial(self.X[0, k], pos_2d[0] + a * (goal_2d[0] - pos_2d[0]))
                self.opti.set_initial(self.X[1, k], pos_2d[1] + a * (goal_2d[1] - pos_2d[1]))
                self.opti.set_initial(self.X[2, k], 0)
                self.opti.set_initial(self.X[3, k], 0)

        try:
            t0 = time.time()
            sol = self.opti.solve()
            ms = (time.time() - t0) * 1000
            X_sol = sol.value(self.X)
            U_sol = sol.value(self.U)
            self._U0 = {'X': X_sol, 'U': U_sol}
            return X_sol[0:2, 1], ms, True
        except Exception:
            self._U0 = None
            return pos_2d.copy(), 0.0, False


# ============================================================================
#  Crazyflie ROS node
# ============================================================================

class _Cmd:
    """Internal command object placed on the executor queue from a service handler.

    The flight ops (takeoff/fly_to/land) call Crazyswarm functions which need
    to run on the *main* thread — that's the one that holds the rclpy executor
    Crazyswarm relies on. So service handlers enqueue a _Cmd and wait on its
    Event, while main() runs a worker loop that pops commands and executes.
    """
    __slots__ = ('kind', 'kwargs', 'done', 'success', 'info')

    def __init__(self, kind, **kwargs):
        self.kind = kind
        self.kwargs = kwargs
        self.done = threading.Event()
        self.success = False
        self.info = ''


class CrazyflieNode(Node):
    def __init__(self, cmd_queue, get_state):
        super().__init__('crazyflie_mpc')
        self.cmd_queue = cmd_queue
        self.get_state = get_state  # callable -> dict {'is_flying': bool}
        self.target = None

        cb = ReentrantCallbackGroup()
        self.create_subscription(Pose2D, '/drone/target', self.on_target, 10, callback_group=cb)
        self.create_service(Empty, '/drone/takeoff', self.svc_takeoff, callback_group=cb)
        self.create_service(Empty, '/drone/fly_to', self.svc_fly_to, callback_group=cb)
        self.create_service(Empty, '/drone/land', self.svc_land, callback_group=cb)

        self.declare_parameter('cmd_timeout_s', 120.0)
        self.get_logger().info(
            'crazyflie_mpc ready — services: /drone/takeoff /drone/fly_to /drone/land; '
            'args via /drone/target (Pose2D, m)')

    # ----- callbacks -----

    def on_target(self, msg: Pose2D):
        self.target = (msg.x, msg.y)
        self.get_logger().info(f'/drone/target set: x={msg.x:.3f} y={msg.y:.3f}')

    def _dispatch(self, cmd: _Cmd, timeout):
        self.cmd_queue.put(cmd)
        if not cmd.done.wait(timeout=timeout):
            cmd.info = f'timeout after {timeout}s'
        self.get_logger().info(
            f"[{cmd.kind}] success={cmd.success} info={cmd.info}")

    def svc_takeoff(self, _req, response):
        cmd = _Cmd('takeoff')
        self._dispatch(cmd, float(self.get_parameter('cmd_timeout_s').value))
        return response

    def svc_fly_to(self, _req, response):
        if self.target is None:
            self.get_logger().error('fly_to: no /drone/target seen')
            return response
        x, y = self.target
        cmd = _Cmd('fly_to', x=x, y=y)
        self._dispatch(cmd, float(self.get_parameter('cmd_timeout_s').value))
        return response

    def svc_land(self, _req, response):
        cmd = _Cmd('land')
        self._dispatch(cmd, float(self.get_parameter('cmd_timeout_s').value))
        return response


# ============================================================================
#  Flight worker (runs on main thread, owns Crazyswarm + MPC)
# ============================================================================

class FlightWorker:
    DEFAULT_HEIGHT = 0.5
    GOAL_TOL = 0.10
    HOVER_S = 1.5
    MAX_STEPS = 300

    def __init__(self, swarm, mpc, flight_height=None):
        self.swarm = swarm
        self.timeHelper = swarm.timeHelper
        self.cf = swarm.allcfs.crazyflies[0]
        self.mpc = mpc
        self.flight_height = flight_height or self.DEFAULT_HEIGHT
        self.is_flying = False

    def state(self):
        return {'is_flying': self.is_flying, 'flight_height': self.flight_height}

    def _ensure_airborne(self):
        if not self.is_flying:
            print(f'[worker] auto-takeoff to {self.flight_height}m before flight op')
            self.cf.takeoff(targetHeight=self.flight_height, duration=3.0)
            self.timeHelper.sleep(3.5)
            self.is_flying = True

    def takeoff(self):
        if self.is_flying:
            return True, f'already airborne at {self.flight_height}m'
        print(f'[worker] takeoff -> {self.flight_height}m')
        self.cf.takeoff(targetHeight=self.flight_height, duration=3.0)
        self.timeHelper.sleep(3.5)
        self.is_flying = True
        return True, f'airborne at {self.flight_height}m'

    def fly_to(self, x, y):
        target = np.array([float(x), float(y)])
        self._ensure_airborne()
        prev = np.array(self.cf.position[:2])
        rate = 1.0 / self.mpc.dt
        dist = float('inf')
        for step in range(self.MAX_STEPS):
            pos_3d = np.array(self.cf.position)
            pos_2d = pos_3d[:2]
            raw_v = (pos_2d - prev) / self.mpc.dt if step > 0 else np.zeros(2)
            v = np.clip(raw_v, -self.mpc.v_max, self.mpc.v_max)
            dist = float(np.linalg.norm(pos_2d - target))
            if dist < self.GOAL_TOL:
                break
            wp, ms, ok = self.mpc.solve(pos_2d, v, target)
            if not ok:
                wp = pos_2d
            self.cf.cmdPosition(np.array([wp[0], wp[1], self.flight_height]), yaw=0.0)
            if step % 10 == 0:
                print(f'[worker] fly_to[{step}] pos=({pos_2d[0]:+.2f},{pos_2d[1]:+.2f}) '
                      f'tgt=({target[0]:+.2f},{target[1]:+.2f}) dist={dist:.3f} '
                      f'mpc={ms:.0f}ms')
            prev = pos_2d.copy()
            self.timeHelper.sleepForRate(rate)

        # Hover at goal so PEFA "sees" arrival.
        goal_3d = np.array([target[0], target[1], self.flight_height])
        for _ in range(int(self.HOVER_S * rate)):
            self.cf.cmdPosition(goal_3d, yaw=0.0)
            self.timeHelper.sleep(self.mpc.dt)
        return True, f'arrived ({target[0]:+.2f},{target[1]:+.2f}) dist={dist:.3f}'

    def land(self):
        if not self.is_flying:
            return True, 'already on ground'
        pos = np.array(self.cf.position)
        tx, ty = float(pos[0]), float(pos[1])
        print(f'[worker] land at ({tx:+.2f},{ty:+.2f})')
        for _ in range(80):
            p = np.array(self.cf.position)
            if p[2] < 0.06:
                break
            self.cf.cmdPosition(np.array([tx, ty, max(0.02, p[2] - 0.008)]), yaw=0.0)
            self.timeHelper.sleep(0.1)
        for _ in range(20):
            self.cf.cmdPosition(np.array([tx, ty, 0.02]), yaw=0.0)
            self.timeHelper.sleep(0.1)
        self.is_flying = False
        return True, f'landed at ({tx:+.2f},{ty:+.2f})'

    def emergency_land(self):
        try:
            if not self.is_flying:
                return
            print('[worker] EMERGENCY LAND')
            for _ in range(60):
                p = np.array(self.cf.position)
                if p[2] < 0.06:
                    break
                self.cf.cmdPosition(np.array([p[0], p[1], max(0.02, p[2] - 0.01)]), yaw=0.0)
                self.timeHelper.sleep(0.1)
            self.is_flying = False
        except Exception as e:
            print(f'[worker] emergency_land error: {e}')

    def run_loop(self, cmd_queue: 'queue.Queue[_Cmd]', stop_flag):
        """Pop commands from the queue and execute on this thread."""
        while not stop_flag.is_set():
            try:
                cmd = cmd_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                if cmd.kind == 'takeoff':
                    cmd.success, cmd.info = self.takeoff()
                elif cmd.kind == 'fly_to':
                    cmd.success, cmd.info = self.fly_to(cmd.kwargs['x'], cmd.kwargs['y'])
                elif cmd.kind == 'land':
                    cmd.success, cmd.info = self.land()
                else:
                    cmd.success, cmd.info = False, f'unknown kind {cmd.kind}'
            except Exception as e:
                cmd.success, cmd.info = False, f'exception: {e}'
            finally:
                cmd.done.set()


# ============================================================================
#  main
# ============================================================================

def main():
    # Crazyswarm() initialises rclpy + a connection to crazyflie_server.
    swarm = Crazyswarm()
    mpc = MPCController()
    worker = FlightWorker(swarm, mpc)

    cmd_queue: 'queue.Queue[_Cmd]' = queue.Queue()
    stop = threading.Event()

    node = CrazyflieNode(cmd_queue, worker.state)

    # Spin the rclpy executor on a background thread; main thread runs the
    # flight worker so all Crazyswarm calls happen on a single (non-rclpy) thread.
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()

    try:
        worker.run_loop(cmd_queue, stop)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        worker.emergency_land()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
