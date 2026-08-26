import numpy as np
import math
import argparse
import os
from tkinter import _flatten
import omnigibson as og
from omnigibson.macros import gm

from omni.isaac.core.utils.prims import get_prim_at_path
from scipy.spatial.transform import Rotation as R
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.quadruped.utils.rot_utils import get_xyz_euler_from_quaternion, get_quaternion_from_euler

from omni.isaac.core.objects import DynamicCuboid, FixedCuboid
from omni.isaac.core.utils.stage import get_current_stage
from omnigibson.sensors.vision_sensor import VisionSensor



from haskillset import SkillSet, LogSet
from recorder import NullRecorder, ViewportRecorder
import worldmodel
from constants import *
from tasks import TASKSET, TASK_OCCUPANCY_MAP_CONFIG
from agents import *
import time
from tqdm import tqdm
duration = 5
# Don't use GPU dynamics and use flatcache for performance boost
gm.USE_GPU_DYNAMICS = True
# gm.ENABLE_FLATCACHE = False
gm.ENABLE_HQ_RENDERING = True


from ros_hademo_ws.src.hademo.src.action_subscriber import ResultPublishandActionSubscribe


from multiprocessing import Process  
import carb
import omni
from omni.isaac.occupancy_map import _occupancy_map


# def get_occupancy_map(task_name):
#     physx = omni.physx.acquire_physx_interface()
#     stage_id = omni.usd.get_context().get_stage_id()

#     generator = _occupancy_map.Generator(physx, stage_id)
#     # 0.05m cell size, output buffer will have 4 for occupied cells, 5 for unoccupied, and 6 for cells that cannot be seen
#     # this assumes your usd stage units are in m, and not cm
#     generator.update_settings(1, 0, 255, 255)
#     print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#     # Set location to map from and the min and max bounds to map to
#     if "Merom_1_int" in task_name:
#         generator.set_transform((0, 0, 0), (-2, -2, 1), (5, 9, 2))
#         '''
#             Top Left: (4.925000286102295, -1.975)		 Top Right: (-1.975, -1.975)
#             Bottom Left: (4.925000286102295, 8.924999809265136)		 Bottom Right: (-1.975, 8.924999809265136)
#             Coordinates of top left of image (pixel 0,0) as origin, + X down, + Y right:
#             (1.975, 4.925000286102295)
#             Image size in pixels: 139, 219
#         '''
#     generator.generate2d()
#     # Get locations of the occupied cells in the stage
#     points = generator.get_occupied_positions()
#     # Get dimensions for 2d buffer
#     dims = generator.get_dimensions()
#     print(dims)
#     # Get computed 2d occupancy buffer
#     buffer = generator.get_buffer()
#     print(np.array(buffer).shape)
#     print("***********************************************************")




# def create_cube(agent, cube_name, color=np.array([0, 0, 1.0])):
#     # _quat_cube2world_back = R.from_euler('XYZ', [-180, 0, 180], degrees=True)
    
    
#     if agent.name == "franka":
#         base_pos = agent._init_position
#         _quat_cube2world_front = euler_angles_to_quat(np.array([-np.pi, 0, np.pi]))
#     if agent.name == "aliengo":
#         base_pos, base_ori = agent.get_arm_base_position_orientation()
#         _quat_cube2world_front = base_ori

#     return FixedCuboid(
#             prim_path="/World/" + cube_name,
#             name=cube_name,
#             position=base_pos + np.array([0.5, 0.3, 0.5]),
#             orientation=_quat_cube2world_front,
#             scale=np.array([0.02, 0.02, 0.02]),
#             color=color,
#         )



# ---------------------------------------------------------------------------
# Viewer-camera chase cam
#
# main() parks the viewer camera at a fixed pose per task, which is fine while
# everything of interest sits still but loses the payload the moment a robot
# carries it across the scene.  The classes below re-aim the camera every
# simulation step instead.
#
# A single "point the camera at the payload" rule does not survive this task,
# for two reasons:
#
#   1. The kebab spends the opening minutes inside a *closed fridge*.  A camera
#      trained on it sits inside the fridge geometry and the viewport is black.
#      So the first shot watches the dog that is going to fetch it.
#
#   2. Once the quadrotor has the kebab it flies out to the terrace.  A camera
#      that keeps a fixed offset from the drone ends up ploughing through walls
#      and roof on the way.  So the second shot flies the camera along the path
#      the drone itself just flew -- provably open space -- trailing a few
#      metres behind at the drone's own cruising height.
# ---------------------------------------------------------------------------

FOLLOW_CAM_SMOOTHING = 0.08


def look_at_quaternion(position, target, up=(0.0, 0.0, 1.0)):
    """Orientation as (x, y, z, w) that aims a USD camera at `target`.

    A USD camera looks down its own -Z axis with +Y up, so the rotation we want
    is just the basis [right, up, -forward] written in world coordinates.
    """
    forward = np.asarray(target, dtype=float) - np.asarray(position, dtype=float)
    distance = np.linalg.norm(forward)
    if distance < 1e-6:
        return np.array([0.0, 0.0, 0.0, 1.0])
    forward = forward / distance

    z_axis = -forward
    x_axis = np.cross(np.asarray(up, dtype=float), z_axis)
    if np.linalg.norm(x_axis) < 1e-6:
        # Looking straight up or down; the roll is arbitrary, so pick one.
        x_axis = np.cross(np.array([0.0, 1.0, 0.0]), z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    return R.from_matrix(np.column_stack((x_axis, y_axis, z_axis))).as_quat()


def prim_position(prim_path):
    """World translate of `prim_path`, or None if it is not on the stage.

    Read straight off the stage, the same way update_worldmodel does, and that
    is deliberate: a carried object is never reparented under the robot.
    Agent.attach() writes the object's world translate every frame, so
    /World/<name> stays live whether the object is in the fridge, in the
    aliengo's gripper, or slung under the quadrotor.
    """
    prim = get_current_stage().GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return None
    translate = prim.GetAttribute("xformOp:translate").Get()
    if translate is None:
        return None
    return np.array(translate, dtype=float)


class PathTrail:
    """Breadcrumb trail of where a subject has been.

    Used to place the camera on ground the subject has already covered, which
    for a flying robot is the one path guaranteed to be clear of walls.
    """

    def __init__(self, min_spacing=0.02, max_points=4000):
        self.min_spacing = min_spacing
        self.max_points = max_points
        self._points = []

    def record(self, position):
        if self._points and np.linalg.norm(position - self._points[-1]) < self.min_spacing:
            return
        self._points.append(np.array(position, dtype=float))
        if len(self._points) > self.max_points:
            del self._points[:len(self._points) - self.max_points]

    def point_behind(self, distance):
        """Point `distance` metres back along the trail.

        Falls back to the oldest breadcrumb we have while the trail is still
        shorter than that, so the camera starts close and drifts back to its
        proper distance as the flight gets going.
        """
        if not self._points:
            return None
        travelled = 0.0
        for i in range(len(self._points) - 1, 0, -1):
            travelled += np.linalg.norm(self._points[i] - self._points[i - 1])
            if travelled >= distance:
                return self._points[i - 1]
        return self._points[0]


class FollowShot:
    """One camera setup: a subject to watch, and where to watch it from.

    Three ways to place the camera, and they suit different phases:

    `pivot` -- the camera does not move at all; it stands at a fixed world
        point and only turns to track the subject.  Unglamorous, but it is the
        only placement that cannot walk into a wall, which is what a moving
        camera keeps doing in a room this cluttered.

    `offset` + `lock_height` -- a fixed compass offset from the subject at a
        fixed world height.  Good for a robot working in one spot, where the
        camera wants to stay put on the room side of the action.  Locking the
        height keeps the camera from burrowing into the floor or the ceiling
        when the subject changes level.

    `trail` + `rise` -- `trail` metres back along the path the subject itself
        has flown, `rise` metres above it.  Good for long flights: the shot
        stays behind the drone through both legs of the route (+Y out of the
        living room, then +X across to the terrace) without a fixed offset
        dragging the camera through the building on the corner.
    """

    def __init__(self, subject, offset=(0.0, 0.0, 0.0), lock_height=None,
                 pivot=None, trail=None, rise=0.4, min_standoff=2.5,
                 smoothing=FOLLOW_CAM_SMOOTHING):
        self.subject = subject
        self.prim_path = "/World/" + subject
        self.offset = np.asarray(offset, dtype=float)
        self.lock_height = lock_height
        self.pivot = None if pivot is None else np.asarray(pivot, dtype=float)
        self.trail = trail
        self.rise = rise
        self.min_standoff = min_standoff
        self.smoothing = smoothing
        self._path = PathTrail() if trail is not None else None

    def target(self):
        return prim_position(self.prim_path)

    def camera_position(self, target, current=None):
        if self.pivot is not None:
            return self.pivot.copy()

        if self.trail is not None:
            self._path.record(target)
            behind = self._path.point_behind(self.trail)
            if behind is None:
                behind = target
            # `rise` is relative to the flown path, and immediately after a
            # takeoff that path is still on the floor -- so the camera is placed
            # near floor level looking UP at a drone already at cruise height,
            # and the frame fills with ceiling.  Raising `rise` to fix that
            # overshoots later, once the trail point is itself at cruise: in a
            # room with a 2.4 m ceiling the camera then sits outside the
            # building.  `lock_height` is the way out -- a fixed world height
            # that is above the cruise altitude and below the ceiling for the
            # whole flight, which is exactly what the docstring above promises
            # it does when the subject changes level.
            z = self.lock_height if self.lock_height is not None else behind[2] + self.rise
            position = np.array([behind[0], behind[1], z])
            return self._enforce_standoff(position, target, current)

        position = target + self.offset
        if self.lock_height is not None:
            position[2] = self.lock_height
        return position

    def _enforce_standoff(self, position, target, current):
        """Keep the camera from ending up on top of the subject.

        A flight shot starts with an empty trail, so point_behind hands back
        the subject's own position -- and while the drone climbs out of the
        handoff the camera would sit inside it, looking at nothing.  Push the
        camera out to min_standoff along the best horizontal direction we have:
        away from wherever the camera already is (which at the cut is the spot
        the previous shot left it, known-good ground), else back along the
        subject's own heading, else an arbitrary but stable axis.
        """
        if self.min_standoff is None:
            return position
        if np.linalg.norm(position[:2] - target[:2]) >= self.min_standoff:
            return position

        candidates = []
        if current is not None:
            candidates.append(np.asarray(current, dtype=float)[:2] - target[:2])
        candidates.append(position[:2] - target[:2])
        candidates.append(-self._heading_xy())
        candidates.append(np.array([1.0, 0.0]))

        for direction in candidates:
            norm = np.linalg.norm(direction)
            if norm > 1e-3:
                position[:2] = target[:2] + (direction / norm) * self.min_standoff
                break
        return position

    def _heading_xy(self):
        """Horizontal direction the subject is travelling, from the trail."""
        points = self._path._points if self._path is not None else []
        if len(points) < 2:
            return np.zeros(2)
        return points[-1][:2] - points[-2][:2]


class ChaseCamera:
    """Runs a task's shot list on og.sim's viewer camera.

    The cut from one shot to the next happens when `carrier` picks up
    `payload` -- for demo2, the moment the dog drops the kebab into the
    quadrotor's basket.  It latches: the drone hands the kebab over again at
    the far end, and cutting back to the dog there would be nonsense.
    """

    def __init__(self, shots, carrier=None, payload=None):
        assert shots, "a ChaseCamera needs at least one shot"
        self.shots = shots
        self.carrier = carrier
        self.payload = payload
        self._agents = None
        self._shot_index = 0
        self._camera_position = None
        self._look_at = None
        self._warned = set()

    def bind(self, agents_dict):
        """Hand over the agents, once SkillSet has built them.

        Until this is called the camera just runs its first shot, which is
        correct: nothing can have been picked up yet.
        """
        self._agents = agents_dict

    @property
    def shot(self):
        return self.shots[self._shot_index]

    def _payload_collected(self):
        if self._agents is None or self.carrier is None or self.payload is None:
            return False
        agent = self._agents.get(self.carrier)
        attached = getattr(agent, "attached_prim_list", None)
        return bool(attached) and ("/World/" + self.payload) in attached

    def _advance_shot(self):
        if self._shot_index + 1 < len(self.shots) and self._payload_collected():
            self._shot_index += 1
            print("Follow camera: %s has %s -- switching to shot %d (%s)."
                  % (self.carrier, self.payload, self._shot_index + 1, self.shot.subject))

    def step(self):
        """Re-aim the camera.  Call once per og.sim.step()."""
        self._advance_shot()
        shot = self.shot

        target = shot.target()
        if target is None:
            if shot.prim_path not in self._warned:
                print("Follow camera: no prim at %s; leaving the viewer camera "
                      "where main() put it." % shot.prim_path)
                self._warned.add(shot.prim_path)
            return

        desired = shot.camera_position(target, self._camera_position)
        if self._camera_position is None:
            # First frame: snap, so the shot opens already framed on the subject.
            self._camera_position = desired
            self._look_at = target
        else:
            self._camera_position += shot.smoothing * (desired - self._camera_position)
            self._look_at += shot.smoothing * (target - self._look_at)

        og.sim.viewer_camera.set_position_orientation(
            position=self._camera_position,
            orientation=look_at_quaternion(self._camera_position, self._look_at),
        )


# Shot list per task; keys are matched as substrings of the task name.
#
# demo2 numbers, for reference when retuning: the dog works the fridge around
# (4.2, 0.6) and meets the drone at (4.6, 1.95); the drone then flies to
# (5, 9) and on to the terrace at (15.3, 12), cruising at quadrotor_defaults
# ["hover_height"] = 1.5 m the whole way.
FOLLOW_CAM_SHOTS = {
    # demo1 numbers, for reference when retuning: everything happens along one
    # corridor of the living room at y ~ 7.5, running -X from the coffee table
    # (3.6, 7.37) where the apple starts, past the handover, to the breakfast
    # table (-0.7, 7.45) where the franka is waiting at (-1.15, 7.55).
    "Merom_1_int": dict(
        carrier="quadrotor_0",
        payload="apple_agveuv_0",
        shots=[
            # The fetch.  A pivot, for the reason spelled out in the demo2 entry
            # below: a camera holding an offset from the dog cannot cross this
            # room without ending up inside something.
            #
            # It stands at (1.775, 6.05), which is not a guess -- that is
            # "targetpoint-living_room_0-p1" from the task defaults, a navigation
            # waypoint the dog itself is sent to, so it is known-open floor.  It
            # sits ~1.5 m south of the action line and looks north, which puts
            # both ends of the corridor at about 2.6 m: close enough to read the
            # apple, wide enough to hold the handover.
            dict(subject="aliengo_0", pivot=(1.775, 6.05, 1.6), smoothing=0.10),
            # The flight.  Trail the apple, height-locked.
            #
            # A pivot cannot do this leg: the drone leaves the living room for the
            # dining alcove, and the two are not mutually visible -- a pivot that
            # frames the flight ends the run pointed at the wall between them,
            # which is why main() ships separate viewpoints for the two ends.
            #
            # So the camera has to travel, and `trail` is what follows the route
            # without a fixed offset dragging it through the wall on the corner.
            # lock_height is what makes it usable: the drone cruises at
            # quadrotor_defaults["hover_height"] = 1.5 m under a ~2.4 m ceiling,
            # so 1.95 m sits just above it looking slightly down, all the way from
            # the takeoff (where the flown path is still at land_height = 0.05)
            # to the landing on the breakfast table.
            dict(subject="apple_agveuv_0", trail=1.8, lock_height=1.95,
                 min_standoff=1.5, smoothing=0.12),
        ],
    ),
    "house_double_floor_lower": dict(
        carrier="quadrotor_0",
        payload="kebab_cewhbv_0",
        shots=[
            # The fetch.  A camera that keeps an offset from the dog does not
            # survive this kitchen: the dog spawns at (7.4, 1.9), in the corner
            # by the window, and any offset that frames it nicely at the fridge
            # puts the lens through the wall at x = 8.4 at spawn.
            #
            # So this one does not move.  It stands one third of the way along
            # the segment between the two viewpoints main() ships for this task
            # -- (5, 2.235, 1.46) "in front of the fridge" and (8.25, 3.35,
            # 1.05) "far from fridge" -- and only turns.  Both endpoints are
            # authored and known to render, so the segment between them is open
            # kitchen floor, which no amount of arithmetic on root_link
            # positions can establish on its own (that authored (8.25, 3.35)
            # sits 0.47 m from a door origin and is fine, while 0.79 m from a
            # wall origin is inside the wall).
            #
            # From here the dog stays 1.2 - 2.7 m away across the whole fetch:
            # spawn, the walk over, the fridge, and the handover spot.
            dict(subject="aliengo_0", pivot=(5.98, 2.57, 1.45), smoothing=0.10),
            # The flight.  Follow where the drone has been, but stay CLOSE.
            # The distance here is not about framing, it is about occlusion:
            # the route leaves the building through a doorway, and a camera
            # several metres back is still in the dark room looking at a door
            # frame while the drone is already outside in the garden.  Under
            # 2 m the camera goes through the door with it.
            dict(subject="kebab_cewhbv_0", trail=1.8, rise=0.5,
                 min_standoff=1.5, smoothing=0.12),
        ],
    ),
}


class HASimulationSystem:
    def __init__(self, task_name, env, follow_camera=None, recorder=None):
        self.task_name = task_name
        self.env = env
        # NullRecorder when --record is off, so the step path needs no conditional.
        self.recorder = recorder if recorder is not None else NullRecorder()


        if len(env.agents) == 0:
            while True:
                og.sim.step()
        
        self.world_entity_name = self.getWorldEntityName(env.config)

        # None when this task has no follow target and none was requested,
        # in which case the viewer camera keeps the fixed pose main() gave it.
        self.follow_camera = follow_camera
        # print(self.world_entity_name)
        # print("+++++++++++++++++++++++++++++++++++++++++++++")
        for _ in range(50):
            # step simulation
            og.sim.step()

        with tqdm(total=duration, desc="Wait 5 sec before the sim start.", unit="s") as pbar:
            start_time = time.time()
            while True:

                self.update_follow_camera()
                og.sim.step()
                self.update_worldmodel()
                elapsed_time = time.time() - start_time
                pbar.update(elapsed_time - pbar.n)
                
                # Check if 5 seconds have passed
                if elapsed_time > duration:
                    print("5 seconds have passed, exiting loop.")
                    break
        self.update_worldmodel()

        self.skillset = SkillSet(self.task_name, env.agents)
        if self.follow_camera is not None:
            # The camera needs the agents to see when the drone takes the
            # payload; until now it has been running its opening shot.
            self.follow_camera.bind(self.skillset.agents_dict)
        print(env.agents)
        # input('stop')
        self.logger = LogSet()


        # ROS 2: the node (or, under Isaac Sim's python3.7, the sidecar that
        # owns it) is created and spun by ResultPublishandActionSubscribe --
        # there is no separate init_node step as there was with rospy.
        self.simnode = ResultPublishandActionSubscribe(task_name)


        tmp = 0
        _tmp = 0
        flag = False
        while og.app.is_running():

            next_step_action = self.simnode._next_step_action
            if next_step_action:
                print(next_step_action)
                print("----------------------------------------------------------------")
                feedback_result = self.running_next_step_action(next_step_action)
                self.logger.action_round_plus1()

                self.update_worldmodel()
                self.simnode.publish_feedback_result(feedback_result)
                
            else:
                for aliengo_name in self.skillset.aliengo_name_list:
                    self.skillset.aliengo_dog_stand_still(aliengo_name)

                for quadrotor_name in self.skillset.quadrotor_name_list:
                    self.skillset.quadrotor_hover(quadrotor_name)
            if flag:
                ocm_p = Process(target=get_occupancy_map, args=(task_name, ))
                ocm_p.start()
            
            self.skillset.attach()

            # step simulation
            self._step_sim()

        # og.app has stopped, so nothing will consume further actions; tear the
        # ROS 2 endpoint down instead of blocking in a spin() that can never
        # make progress (the old rospy.spin() here just hung on shutdown).
        self.simnode.shutdown()
        self.recorder.close()

    def _step_sim(self):
        """Advance the simulator one step: re-aim the camera, step, capture.

        Both the idle loop and the action loop go through here, which is what makes
        the recording continuous -- the sim keeps stepping (and so keeps rendering)
        while the planner is off waiting on an LLM, so the video never freezes.
        """
        self.update_follow_camera()
        og.sim.step()
        self.recorder.capture()

    def update_follow_camera(self):
        """No-op unless a follow target was configured for this run."""
        if self.follow_camera is not None:
            self.follow_camera.step()

    def getWorldEntityName(self, env_config):
        robot_config = env_config['robots']
        agent_config = env_config['agents']
        object_config = env_config['objects']


        world_entity_name = []

        for i in range(len(robot_config)):
            entity = robot_config[i]
            world_entity_name.append(entity['name'])
            
        for i in range(len(agent_config)):
            entity = agent_config[i]
            world_entity_name.append(entity['name'])
            
        for i in range(len(object_config)):
            entity = object_config[i]
            world_entity_name.append(entity['name'])
            
        return world_entity_name

    def update_worldmodel(self):
        
        state = dict()

        # print('******Current World Model***********')
        for entity_name in self.world_entity_name:
            prim_path = '/World/' + entity_name
            xform = get_current_stage().GetPrimAtPath(prim_path)
            translate = xform.GetAttribute('xformOp:translate').Get()
            orient = xform.GetAttribute('xformOp:orient').Get()  # scalar-first
            quat_imaginery = np.array(orient.GetImaginary())
            # print(orient.GetReal())
            # print(orient.GetImaginary())
            # print(np.array(orient))
            # print(np.array(orient)[0])
            # import time
            # time.sleep(3)
            state[entity_name] = [np.array(translate)[0], np.array(translate)[1], np.array(translate)[2], 
                                  quat_imaginery[0], quat_imaginery[1], quat_imaginery[2], orient.GetReal()]
            
            # print(xform.GetAuthoredAttributes())
            # print('translate:', xform.GetAttribute('xformOp:translate').Get())
            # print('orient:', xform.GetAttribute('xformOp:orient').Get())
            
            # print('entity name', entity_name)
            # print('translate:', translate)
            # print('orient (scalar-first):', orient)
            # print(state[entity_name])

            if "aliengo" in entity_name:
                arm_base_pos, arm_base_ori = self.env.agents[1].get_arm_base_position_orientation()
                arm_base_pos = arm_base_pos.tolist()
                arm_base_ori = arm_base_ori.tolist()
                state[entity_name].extend([arm_base_pos[0], arm_base_pos[1], arm_base_pos[2], 
                                           arm_base_ori[1], arm_base_ori[2], arm_base_ori[3], arm_base_ori[0]])

              
        worldmodel.set_state(state, self.task_name)

    

    def running_next_step_action(self, next_step_action):
        action_num = 0
        finish_flag = {}
        feedback_result = {}
        # print(next_step_action)
        for agent_name in self.skillset.agents_dict.keys():
            if len(next_step_action[agent_name]):
                action_num += 1
            finish_flag[agent_name] = False if len(next_step_action[agent_name]) else True
            feedback_result[agent_name] = {}
                
        assert action_num, "current recerived next_step_action is empty"
        # if action_num == 0:
            
        #     # Run simulation step
        #     og.sim.step()
        ########################################
        ## 
        print("============== Action Round %d ==============" % self.logger.action_round)
        self.recorder.mark("action_round", round=self.logger.action_round)
        print(finish_flag)
        while action_num:
            

            for agent_name, agent_finish_flag in finish_flag.items():
                if not agent_finish_flag:
                    func_name, func_args = next_step_action[agent_name]
                    print('next_step_action:', next_step_action)
                    done, success, info = eval(f"self.skillset.{func_name}")(**func_args)
                    print('skillset:', func_name, 'done:', done, 'success:', success, 'info:', info,'\n')

                    if "waypoint_ind" in func_args.keys():
                        func_args["waypoint_ind"] = info["waypoint_ind"]
                    if done:
                        finish_flag[agent_name] = True
                        feedback_result[agent_name] = {"has_result": True, "success": success, "info": self.logger.get_single_agent_result_info(success, info)}
                        print("%s: %s%s" % (func_name, "success" if success else "failed, ", "" if success else self.logger.get_single_agent_result_info(success, info)))
                        
                        if "franka" in agent_name:
                            self.skillset.reset_franka_waypoint_ind(agent_name)
                        if "aliengo" in agent_name:
                            self.skillset.reset_aliengo_waypoint_ind(agent_name)
                        if "quadrotor" in agent_name:
                            self.skillset.reset_quadrotor_waypoint_ind(agent_name)
                            
                        action_num -= 1
                else:
                    # If there is no action for the aliengo/quadrotor, then remain stationary.
                    if "aliengo" in agent_name:
                        self.skillset.aliengo_dog_stand_still(agent_name)
                    if "quadrotor" in agent_name:
                        self.skillset.quadrotor_hover(agent_name)


            # Run simulation step
            self._step_sim()

        return feedback_result

    # def test_with_defined_next_step_action(self):
    #     ########################################
    #     ## Franka
    #     if self.skillset.franka:
    #         franka_target_cube = create_cube(self.skillset.franka, cube_name="franka_target_cube", color=np.array([0, 0, 1.0]))
    #         franka_cube_position, franka_cube_orientation = franka_target_cube.get_world_pose()

    #         franka_waypoint_pos = np.array([
    #             franka_cube_position
    #         ])
    #         franka_waypoint_ori = np.array([
    #             franka_cube_orientation
    #         ])

    #     ########################################
    #     ## Aliengo
    #     if self.skillset.aliengo:
    #         aliengo_dog_waypoint_pos = np.array([
    #             self.skillset.aliengo._init_position + np.array([0.5, 0, 0]),
    #             self.skillset.aliengo._init_position + np.array([1, 0.5, 0])
    #         ])

    #         aliengo_dog_waypoint_ori = np.array([
    #             get_quaternion_from_euler(np.array([0.0, 0.0, 0.0])),
    #             get_quaternion_from_euler(np.array([0.0, 0.0, 45.0/180*math.pi])),
    #         ])

    #         aliengo_target_cube = create_cube(self.skillset.aliengo, cube_name="aliengo_target_cube", color=np.array([0, 1.0, 0.0]))
    #         aliengo_cube_position, aliengo_cube_orientation = aliengo_target_cube.get_world_pose()

    #         aliengo_arm_waypoint_pos = np.array([
    #             aliengo_cube_position
    #         ])
    #         aliengo_arm_waypoint_ori = np.array([
    #             aliengo_cube_orientation
    #         ])


    #     ########################################
    #     ## Quadrotor
    #     if self.skillset.quadrotor:
    #         quadrotor_waypoint_pos = np.array([
    #             self.skillset.quadrotor._init_position+np.array([0.5, 0, self.skillset.quadrotor.default_hover_height]),
    #             self.skillset.quadrotor._init_position+np.array([1.0, 0.5, self.skillset.quadrotor.default_hover_height]),
    #         ])

    #         quadrotor_waypoint_ori = np.array([
    #             self.skillset.quadrotor._init_orientation,
    #             (R.from_quat(R.from_euler('z', 45, degrees=True).as_quat()) * R.from_quat(self.skillset.quadrotor._init_orientation[[1, 2, 3, 0]])).as_quat()[[3, 0, 1, 2]]
    #         ])



    #     ########################################
    #     ## Define action_list
    #     next_step_action = {
    #         # "franka": [],
    #         # ["franka_open_gripper", {}],
    #         # ["franka_close_gripper", {}],
    #         "franka": ["franka_pick", {"waypoint_pos": franka_waypoint_pos, "waypoint_ori": franka_waypoint_ori, "waypoint_ind": 0}],
    #         # ["franka_place", {"waypoint_pos": franka_waypoint_pos, "waypoint_ori": franka_waypoint_ori, "waypoint_ind": 0}],
            
    #         "aliengo": ["aliengo_dog_move_with_waypoints", {"waypoint_pos": aliengo_dog_waypoint_pos, "waypoint_ori": aliengo_dog_waypoint_ori, "waypoint_ind": 0}],
    #         # ["aliengo_arm_open_gripper", {}],
    #         # ["aliengo_arm_close_gripper", {}],
    #         # ["aliengo_arm_pick", {"waypoint_pos": aliengo_arm_waypoint_pos, "waypoint_ori": aliengo_arm_waypoint_ori, "waypoint_ind": 0}],
    #         # ["aliengo_arm_place", {"waypoint_pos": aliengo_arm_waypoint_pos, "waypoint_ori": aliengo_arm_waypoint_ori, "waypoint_ind": 0}],
            
    #         # "quadrotor": []
    #         # ["quadrotor_takeoff", {}]
    #         # ["quadrotor_land", {}]
    #         "quadrotor": ["quadrotor_move_with_waypoints", {"waypoint_pos": quadrotor_waypoint_pos, "waypoint_ori": quadrotor_waypoint_ori, "waypoint_ind": 0}]
    #     }

    #     finish_flag = {
    #         "franka": False if len(next_step_action["franka"]) else True,
    #         "aliengo": False if len(next_step_action["aliengo"]) else True,
    #         "quadrotor": False if len(next_step_action["quadrotor"]) else True
    #     }

    #     # has_result: bool, success: bool, errorInfo: string
    #     feedback_result = {
    #         "franka": {},
    #         "aliengo": {},
    #         "quadrotor": {}
    #     }

    #     ########################################
    #     ## 
    #     print("Simulation Begin!")
    #     # Step!


    #     for i in range(100000):
            
    #         # 没有针对 aliengo / quadrotor 的动作，则保持不动
    #         if finish_flag["aliengo"]:
    #             self.skillset.aliengo_dog_stand_still()
    #         if finish_flag["quadrotor"]:
    #             self.skillset.quadrotor_hover()

    #         for agent_name, agent_finish_flag in finish_flag.items():
    #             if not agent_finish_flag:
    #                 func_name, func_args = next_step_action[agent_name]
    #                 done, success, info = eval(f"self.skillset.{func_name}")(**func_args)
    #                 if "waypoint_ind" in func_args.keys():
    #                     func_args["waypoint_ind"] = info["waypoint_ind"]
    #                 if done:
    #                     finish_flag[agent_name] = True
    #                     feedback_result[agent_name] = {"has_result": True, "success": success, "info": self.skillset.get_single_agent_result_info(success, info)}
    #                     print("%s: %s%s" % (agent_name, "success" if success else "failed, ", "" if success else self.skillset.get_single_agent_result_info(success, info)))
                    
    #                     self.skillset.reset_all_waypoint_ind()
    #                     eval("self.skillset.reset_%s_waypoint_ind" % agent_name)()
    #                     eval("self.skillset.reset_%s_cache" % agent_name)()
                    
                    

    #         # Run simulation step
    #         og.sim.step()



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, default='Merom_1_int_Task1')

    # Chase cam.  Defaults come from FOLLOW_CAM_SHOTS; these flags are here so
    # the framing can be retuned without editing the source.
    # Performance.  See apply_render_performance_settings().
    parser.add_argument('--indirect_diffuse', action='store_true',
                        help='Keep indirect diffuse lighting on.  Prettier '
                             'ambient light, roughly 5 fps slower.')
    parser.add_argument('--show_agent_cameras', action='store_true',
                        help="Keep the agents' onboard camera viewports "
                             'rendering.  Nothing reads them, so they cost a '
                             'render pass per step for display only.')
    parser.add_argument('--viewer_size', type=int, nargs=2, default=(960, 540),
                        metavar=('WIDTH', 'HEIGHT'),
                        help='Main viewport resolution.  The task configs ask '
                             'for 1280x720; 960x540 is 44%% fewer pixels to '
                             'ray-trace.  Pass 1280 720 to restore.')

    # Recording.  See recorder.py; the compositor is tools/compose_video.py.
    parser.add_argument('--record', action='store_true',
                        help='Capture the viewer camera to sim.mp4 + frames.jsonl '
                             'in --record_dir, for the demo videos.')
    parser.add_argument('--record_dir', type=str, default=None,
                        help='Where the capture goes.  Defaults to '
                             '$COHERENT_RECORD_DIR, else results/videos/<task>-<ts>.')
    parser.add_argument('--record_fps', type=int, default=30,
                        help='Playback frame rate written into the mp4 header.')
    parser.add_argument('--record_every', type=int, default=1,
                        help='Capture one frame per N sim steps.  The action '
                             'timestep is 16/400 s (25 Hz), so 1 is right for a '
                             '30 fps video; raise it to shrink long runs.')

    parser.add_argument('--no_follow_camera', action='store_true',
                        help='Leave the viewer camera at the fixed pose set in '
                             'main() instead of chasing anything.')
    parser.add_argument('--follow_object', type=str, default=None,
                        help='Ignore the task shot list and chase this one '
                             'object for the whole run, e.g. aliengo_0.')
    parser.add_argument('--follow_pivot', type=float, nargs=3, default=None,
                        metavar=('X', 'Y', 'Z'),
                        help='Where the fetch-shot camera stands while it '
                             'tracks the dog.  Keep it between the two '
                             'viewpoints main() ships for the task -- those are '
                             'known to be inside the room.')
    parser.add_argument('--follow_trail', type=float, default=None,
                        help='Metres the flight shot trails behind the drone '
                             'along its own flown path.')
    parser.add_argument('--follow_rise', type=float, default=None,
                        help='Metres the flight shot sits above the drone\'s '
                             'cruising height.  Raise it for a wider view of '
                             'the terrace, lower it if the camera clips the roof.')
    parser.add_argument('--follow_smoothing', type=float, default=None,
                        help='Fraction of the tracking error the camera closes '
                             'per step; lower is smoother and laggier.')

    return parser.parse_args()


def build_chase_camera(args):
    """The ChaseCamera for this run, or None to leave the viewer camera alone."""
    if args.no_follow_camera:
        return None

    if args.follow_object:
        config = dict(shots=[dict(subject=args.follow_object, pivot=(5.98, 2.57, 1.45))])
    else:
        config = None
        for task_key, task_config in FOLLOW_CAM_SHOTS.items():
            if task_key in args.task_name:
                config = task_config
                break
        if config is None:
            return None

    shots = []
    for shot_config in config["shots"]:
        shot_config = dict(shot_config)
        # Only override a knob where the shot actually uses it, so tuning the
        # flight does not quietly reshape the fetch.
        is_trail_shot = shot_config.get("trail") is not None
        if args.follow_trail is not None and is_trail_shot:
            shot_config["trail"] = args.follow_trail
        if args.follow_rise is not None and is_trail_shot:
            shot_config["rise"] = args.follow_rise
        if args.follow_pivot is not None and shot_config.get("pivot") is not None:
            shot_config["pivot"] = tuple(args.follow_pivot)
        if args.follow_smoothing is not None:
            shot_config["smoothing"] = args.follow_smoothing
        shots.append(FollowShot(**shot_config))

    camera = ChaseCamera(shots, carrier=config.get("carrier"),
                         payload=config.get("payload"))
    print("Viewer camera shot list: %s"
          % " -> ".join(shot.subject for shot in shots))
    return camera


def build_recorder(args):
    """The ViewportRecorder for this run, or a NullRecorder when --record is off."""
    if not args.record:
        return NullRecorder()
    out_dir = args.record_dir or os.environ.get("COHERENT_RECORD_DIR")
    if not out_dir:
        out_dir = os.path.join(
            os.environ.get("COHERENT_PATH", os.getcwd()),
            "results", "videos",
            "%s-%s" % (args.task_name, time.strftime("%Y%m%d-%H%M%S")),
        )
    return ViewportRecorder(out_dir, fps=args.record_fps, every_n=args.record_every)


def apply_render_performance_settings(args):
    """Trade some lighting quality for frame rate.

    Simulator._set_renderer_settings() turns indirect diffuse ON and notes in
    the same breath that it "Can be False with a 5fps gain".  Those settings
    are applied while the simulator boots, so this has to run afterwards to
    win.  The effect is purely cosmetic -- ambient bounce light goes flat --
    and nothing in the benchmark reads pixels, so nothing downstream cares.
    """
    if args.indirect_diffuse:
        print("Render: indirect diffuse left ON.")
    else:
        carb.settings.get_settings().set_bool("/rtx/indirectDiffuse/enabled", False)
        print("Render: indirect diffuse OFF (--indirect_diffuse to restore it).")

    hide_agent_camera_viewports(args)


def hide_agent_camera_viewports(args):
    """Stop the agents' onboard cameras from rendering into docked viewports.

    Every Camera prim found under an agent's links gets a VisionSensor, and
    each of those opens its own viewport window -- that is what the
    franka_first_view / franka_third_view panels are.  Each one is a full RTX
    render pass every single step.

    Nothing reads them.  This benchmark drives the simulator directly with
    og.sim.step(); it never calls env.step() or any agent's get_obs(), so the
    pixels are produced and thrown away.  The sensors are left in place and
    still work if something asks them for an image later -- only the window is
    hidden, which is what stops the per-frame render.

    og.sim.viewer_camera is a VisionSensor too, and hiding it would blank the
    main viewport, so it is explicitly skipped.
    """
    if args.show_agent_cameras:
        print("Render: agent camera viewports left visible.")
        return

    viewer_camera = getattr(og.sim, "viewer_camera", None)
    hidden = []
    for prim_path, sensor in VisionSensor.SENSORS.items():
        if sensor is viewer_camera:
            continue
        try:
            sensor.viewer_visibility = False
        except Exception as exc:  # a sensor with no viewport of its own
            print("Render: could not hide %s (%s)" % (prim_path, exc))
            continue
        hidden.append(sensor.name)

    print("Render: hid %d agent camera viewport(s)%s "
          "(--show_agent_cameras to keep them)."
          % (len(hidden), (": " + ", ".join(hidden)) if hidden else ""))


def main(args):

    cfg = TASKSET[args.task_name]

    # env_base applies cfg["render"] to og.sim while it loads, so the override
    # has to land before the Environment is built.
    if args.viewer_size is not None:
        width, height = args.viewer_size
        requested = cfg.get("render", {})
        if (width, height) != (requested.get("viewer_width"), requested.get("viewer_height")):
            print("Render: viewport %dx%d instead of the task's %sx%s."
                  % (width, height, requested.get("viewer_width"),
                     requested.get("viewer_height")))
        cfg["render"] = dict(requested, viewer_width=width, viewer_height=height)

    env = og.Environment(configs=cfg, physics_timestep=1/400., action_timestep=16/400.)

    apply_render_performance_settings(args)

    og.sim.enable_viewer_camera_teleoperation()
    # Update the simulator's viewer camera's pose so it points towards the robot


    if 'Merom_1_int' in args.task_name:
        # original Merom_1_int
        # og.sim.viewer_camera.set_position_orientation(
        #     position=np.array([4.86, 7.85, 1.05]),
        #     orientation=np.array([0.4576,0.46,0.54,0.536]),  # (x,y,z,w)

        # )

        # original Merom_1_int (franka back)
        og.sim.viewer_camera.set_position_orientation(
            position=np.array([-2.10719494,  6.81360779,  1.55880719]),
            orientation=np.array([0.47629345, -0.38766603, -0.49748654,  0.61267181]),  # (x,y,z,w)

        )

    
    if 'house_double_floor_lower' in args.task_name:
        # original house_double_floor (garden)
        # og.sim.viewer_camera.set_position_orientation(
        #     position=np.array([20.53, 14.6, 3.1]),
        #     orientation=np.array([0.31,0.537,0.68,0.4]),  # (x,y,z,w)

        # )

        # original house_double_floor (far from fridge)
        og.sim.viewer_camera.set_position_orientation(
            position=np.array([8.25, 3.35, 1.05]),
            orientation=np.array([0.19,0.6,0.737,0.23]),  # (x,y,z,w)
    
        )

        # original house_double_floor (in front of the fridge)
        # og.sim.viewer_camera.set_position_orientation(
        #     position=np.array([5, 2.235, 1.46]),
        #     orientation=np.array([0.058,0.535,0.84,0.075]),  # (x,y,z,w)
          
        # )


        # original house_double_floor (outside)
        # og.sim.viewer_camera.set_position_orientation(
        #     position=np.array([13.43361232, 10.73460434,  1.65867802]),
        #     orientation=np.array([0.57089634, -0.34936513, -0.39396453, 0.62993121]),  # (x,y,z,w)
        # #    
        # )
    ss = HASimulationSystem(task_name=args.task_name, env=env,
                            follow_camera=build_chase_camera(args),
                            recorder=build_recorder(args))





if __name__ == '__main__':
    args = parse_args()
    main(args)


