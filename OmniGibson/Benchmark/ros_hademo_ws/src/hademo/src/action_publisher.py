#!/usr/bin/env python3
# ## path planning

# import omni
# from omni.isaac.occupancy_map import _occupancy_map


## msg
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import (
    QoSProfile,
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSReliabilityPolicy,
)
from std_msgs.msg import Float64MultiArray
from hademo.msg import Action, FuncAndArgs, Args
from hademo.msg import Result

# A ROS 1 latched publisher becomes TRANSIENT_LOCAL durability in ROS 2.  Both
# ends must declare it: the LLM/sim handshake depends on a late-joining node
# still receiving the last message published before it subscribed.
LATCHED_QOS = QoSProfile(
    depth=10,
    history=QoSHistoryPolicy.KEEP_LAST,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)


import json
import math
import yaml
import numpy as np
import argparse
from scipy.spatial.transform import Rotation as R
import os
import sys
print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))
# print(sys.path)
from tasks import *
import worldmodel
from constants import *







# Vertical clearances for the franka's pick and place, in metres.
#
# The arm was originally given a single waypoint -- the grasp point itself -- so
# it swung diagonally from wherever it was straight into the target, sweeping
# through the basket rim on the way in and dragging the apple sideways on the way
# out. Approaching from directly above and descending makes the motion a normal
# top-down pick-and-place, and it keeps the IK in a comfortable part of the
# workspace instead of at the edge of reach, which is where it used to stall.
# How long the bridge waits for the simulator to report one action finished.
#
# Was 60 s, which is comfortable for a walk or a short flight but not for the
# house demo's articulated work -- opening the fridge is a multi-stage door
# manipulation, and the terrace flight is roughly ten metres. A timeout here is
# reported to the planner as a failed action, so an over-tight value invents
# failures that never happened. Override with $COHERENT_SIM_TIMEOUT_S.
SIM_RESULT_TIMEOUT_S = float(os.environ.get("COHERENT_SIM_TIMEOUT_S", "120"))

FRANKA_APPROACH_H = 0.15   # hover here before descending onto the target
FRANKA_LIFT_H = 0.20       # lift straight up to here before travelling


def _stacked(pos, height):
    """`pos` raised by `height` on Z."""
    return [pos[0], pos[1], pos[2] + height]


def _get_quat_from_euler(euler):
    quat_back = R.from_euler("XYZ", euler).as_quat()

    quat_first = quat_back[[3, 0, 1, 2]]

    return quat_first

def _get_quat_with_z_up(start_pos, end_pos):
    diff_pos = np.array(end_pos) - np.array(start_pos)
    norm_dir_2D = diff_pos / np.linalg.norm(diff_pos)

    x_axis = np.array([norm_dir_2D[0], norm_dir_2D[1], 0], dtype=np.float32)
    z_axis = np.array([0, 0, 1], dtype=np.float32)
    y_axis = np.cross(z_axis, x_axis)

    rot_mat = np.zeros((3, 3))
    rot_mat[:, 0] = x_axis
    rot_mat[:, 1] = y_axis
    rot_mat[:, 2] = z_axis

    euler = R.from_matrix(rot_mat).as_euler('XYZ')

    quat_first = _get_quat_from_euler(euler)

    return quat_first

Merom_1_int_Task1_TextPlanList = [
    "Task beginning",

    ["aliengo_0", "movetowards", "apple_agveuv_0"],
    ["aliengo_0", "grab", "apple_agveuv_0"],
    ["aliengo_0", "movetowards", "quadrotor_0"],
    ["aliengo_0", "putintobasket", ["apple_agveuv_0", "quadrotor_0"]],

    ["quadrotor_0", "takeoff_from", ""],
    ["quadrotor_0", "movetowards", "breakfast_table_skczfi_0"],
    ["quadrotor_0", "land_on", "breakfast_table_skczfi_0"],
    ["franka_0", "grabfrombasket", ["apple_agveuv_0", "quadrotor_0"]],
    ["franka_0", "puton", "breakfast_table_wicker_basket_dgkhyn_0-p1"],
    
]    
house_double_floor_lower_Task1_TextPlanList = [
    "Task beginning",

    ["aliengo_0", "movetowards", "fridge_xyejdx_0_open_door-p1"],
    ["aliengo_0", "movetowards", "fridge_xyejdx_0_open_door-p2"],
    ["aliengo_0", "open", "fridge_xyejdx_0-j_link_1"],
    ["aliengo_0", "movetowards", "kebab_cewhbv_0"],
    ["aliengo_0", "grab", "kebab_cewhbv_0"],
    ["aliengo_0", "movetowards", "quadrotor_0"],
    ["aliengo_0", "putintobasket", ["kebab_cewhbv_0", "quadrotor_0"]],
    ["quadrotor_0", "takeoff_from", ""],
    ["quadrotor_0", "movetowards", "flying_waypoint_out_livingroom_p1"],
    ["quadrotor_0", "movetowards", "flying_waypoint_over_pedestal_table_dmghrm_0"],
    ["quadrotor_0", "land_on", "pedestal_table_dmghrm_0"],
    ["franka_0", "grabfrombasket", ["kebab_cewhbv_0", "quadrotor_0"]],
    ["franka_0", "puton", "grill_sxfjac_0"],

]  


class Translator:
    def __init__(self, task_name):
        self.task_name = task_name
        self.task = eval("defaults_%s" % self.task_name)

    def translate(self, text_action):
        agent_name = text_action[0]
        operation = text_action[1]
        other = text_action[2]
        if '-' in other:
            parts = other.split("-")
            other1 = parts[0]
            other2 = parts[1]
        self.worldstate = worldmodel.get_state(self.task_name)
        

        # e.g. aliengo [movetowards] <apple> (17)
        if operation == "movetowards":
            sim_action_list = self.movetowards2Simaction(agent_name, operation, other)        
        if operation == "grab":
            sim_action_list = self.grab2Simaction(agent_name, operation, other)         
        if operation == "grabfrombasket":
            sim_action_list = self.grabfrombasket2Simaction(agent_name, operation, other)       
        if operation == "puton":
            # A planner emits "[puton] <kebab>(41) on <grill>(43)", which the
            # handler splits into a two-element list -- but puton2Simaction keys
            # straight into task.putpose, so a list raises TypeError: unhashable.
            # The hardcoded demo plans pass a bare surface name, which is why this
            # only bites once a planner drives the scene. Same normalisation the
            # putinto branch below already does.
            sim_action_list = self.puton2Simaction(
                agent_name, operation, self._resolve_putpose_key(other))
        if operation in ("putintobasket", "putinto"):
            # "putinto" a quadrotor basket → putintobasket logic (uses quadrotor pose)
            # "putinto" a regular container (e.g. wicker basket) → puton logic (uses task.putpose)
            if isinstance(other, list) and any("quadrotor" in o for o in other):
                sim_action_list = self.putintobasket2Simaction(agent_name, operation, other)
            elif isinstance(other, str) and "quadrotor" in other:
                sim_action_list = self.putintobasket2Simaction(agent_name, operation, other)
            else:
                # Regular container — treat like "puton". other is [obj, container],
                # puton expects just the container name with a putpose key.
                sim_action_list = self.puton2Simaction(
                    agent_name, operation, self._resolve_putpose_key(other))
        
        if operation == "open":
            sim_action_list = self.opendoor2Simaction(agent_name, operation, other)   
        if operation == "takeoff_from":
            sim_action_list = self.takeoff2Simaction(agent_name, operation, other)      
        if operation == "land_on":
            sim_action_list = self.land2Simaction(agent_name, operation, other)      
        
        return sim_action_list

    def _resolve_putpose_key(self, other):
        """The task.putpose key for a place target.

        Two shapes arrive here. The hardcoded demo plans pass a bare surface name
        ("grill_sxfjac_0"). A planner passes the pair from "[puton] X on Y", which
        the handler splits into [object, surface] -- so take the surface.

        putpose keys are also sometimes qualified beyond the object name
        ("breakfast_table_wicker_basket_dgkhyn_0-p1" for "wicker_basket_dgkhyn_0"),
        so fall back to a substring match before giving up.
        """
        target = other[1] if isinstance(other, (list, tuple)) and len(other) > 1 else other
        if isinstance(target, (list, tuple)):
            target = target[0]
        if target not in self.task.putpose:
            matched = [k for k in self.task.putpose if target in k]
            if matched:
                target = matched[0]
        return target

    def getactionTemplate(self):
        t = dict()
        for name in self.task.agent_name_list:
            t[name] = []
        return t
        
    def movetowards2Simaction(self, agent_name, operation, other):
        sim_action_list = []

        if "aliengo" in agent_name:
            simactionTemplate = self.getactionTemplate()
            simactionTemplate[agent_name] = ["aliengo_dog_stand_begin_to_move", {}]
            sim_action_list.append(simactionTemplate)

        
        currentpose = self.worldstate[agent_name]
        
        if 'putpoint-' in other:

            putname = other[len('putpoint-'):]
            putpose = self.task.putpose[other[len('putpoint-'):]]
            if putname in self.task.nearbypose.keys():
                _goalposes = self.task.nearbypose[putname]
                if len(_goalposes) == 1:
                    goalpose = _goalposes[0]
                else:
                    closeInd = np.argmin(np.linalg.norm(np.array(_goalposes)[:, :2] - np.array(putpose)[:2], axis=-1))
                    goalpose = _goalposes[closeInd]
            else:
                _goalposes = np.concatenate(list(self.task.nearbypose.values()), axis=0)
                if len(_goalposes) == 1:
                    goalpose = _goalposes[0]
                else:
                    closeInd = np.argmin(np.linalg.norm(np.array(_goalposes)[:, :2] - np.array(putpose)[:2], axis=-1))
                    goalpose = list(_goalposes[closeInd])
            
        elif other in self.task.room_name_list:

            _goalposes = self.task.roompose[other]
            if len(_goalposes) == 1:
                goalpose = _goalposes[0]
            else:
                closeInd = np.argmin(np.linalg.norm(np.array(_goalposes)[:, :2] - np.array(currentpose)[:2], axis=-1))
                goalpose = _goalposes[closeInd]
        elif other in self.task.nearbypose.keys():
            _goalposes = self.task.nearbypose[other]
            if len(_goalposes) == 1:
                goalpose = _goalposes[0]
            else:
                closeInd = np.argmin(np.linalg.norm(np.array(_goalposes)[:, :2] - np.array(currentpose)[:2], axis=-1))
                goalpose = _goalposes[closeInd]
        elif other in self.task.agent_name_list and "quadrotor" in other:
            landpointName = self.get_landpoint_name(other)
            _goalposes = self.task.nearbypose[landpointName]
            if len(_goalposes) == 1:
                goalpose = _goalposes[0]
            else:
                closeInd = np.argmin(np.linalg.norm(np.array(_goalposes)[:, :2] - np.array(currentpose)[:2], axis=-1))
                goalpose = list(_goalposes[closeInd])
        else:
            # movetowards to object
            objectpose =  self.worldstate[other]
            _goalposes = np.concatenate(list(self.task.nearbypose.values()), axis=0)
            # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", other)
            # print(_goalposes)
            if len(_goalposes) == 1:
                goalpose = _goalposes[0]
            else:
                closeInd = np.argmin(np.linalg.norm(np.array(_goalposes)[:, :2] - np.array(objectpose)[:2], axis=-1))
                goalpose = list(_goalposes[closeInd])
            # print("(", closeInd, ")")
            # print(goalpose)
            # input('sss')
        waypoints = self.pathplanning(currentpose, goalpose)

        simactionTemplate = self.getactionTemplate()
        if "aliengo" in agent_name:
            simactionTemplate[agent_name] = ["aliengo_dog_move_with_waypoints", waypoints]
        elif "franka" in agent_name:
            simactionTemplate[agent_name] = ["franka_move_with_waypoints", waypoints]
        else:
            simactionTemplate[agent_name] = ["quadrotor_move_with_waypoints", waypoints]
        
        sim_action_list.append(simactionTemplate)

        print("))))))))))))))))))))))))))))))))))))))")
        print(sim_action_list)
        return sim_action_list

    
    def grab2Simaction(self, agent_name, operation, other):


        sim_action_list = []

        attached_prim_path = '/World/' + other  #  + '/base_link'
        objectpose =  self.worldstate[other]
        agentpose = self.worldstate[agent_name]
        print('objectpose', objectpose)
        print('agentpose', agentpose)
             
        grabpos = [
            objectpose[0],
            objectpose[1],
            objectpose[2],        
        ]

        print('grabpos', grabpos)
        
        if "aliengo" in agent_name:
            # print("###########################################")
            # print(self.worldstate[agent_name])
            if self.task_name == "Merom_1_int_Task1":
                #grab apple
                grabori = self.get_oripose2world(gripper2base=aliengo_defaults["topdowngrasp"], base2world=self.worldstate[agent_name][10:])
            elif self.task_name == "house_double_floor_lower_Task1":
                #grab kebab
                grabori = self.get_oripose2world(gripper2base=aliengo_defaults["frontreargrasp"], base2world=self.worldstate[agent_name][10:])
            else:   
                grabori = self.get_oripose2world(gripper2base=aliengo_defaults["topdowngrasp"], base2world=self.worldstate[agent_name][10:])

            print('grabori', grabori)
            # input('stop')
            # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            # print(grabori)
            init_pos, init_quat = self.get_init_EE_pose2world(initEEpose2base_list=aliengo_defaults["initEEpose2base"], base2world_list=self.worldstate[agent_name][7:])
            
            # print(init_pos)
            # print(init_quat)
            # print("*********************************************!!!!!!!!!!!!!!!!!!!!!")
            simactionTemplate = self.getactionTemplate()
            simactionTemplate[agent_name] = ["aliengo_dog_stand_begin_to_manipulate", {}]
            sim_action_list.append(simactionTemplate)

            simactionTemplate = self.getactionTemplate()
            simactionTemplate[agent_name] = ["aliengo_arm_open_gripper", {}]
            sim_action_list.append(simactionTemplate)

            simactionTemplate = self.getactionTemplate()
            simactionTemplate[agent_name] = ["aliengo_arm_pick", {
                'attached_prim_path': attached_prim_path,
                'waypoint_pos': [
                                    grabpos,
                                ],
                'waypoint_ori': [
                                    grabori,
                                ],
                'waypoint_ind': 1
            }]
            sim_action_list.append(simactionTemplate)


            simactionTemplate = self.getactionTemplate()
            simactionTemplate[agent_name] = ["aliengo_arm_move_with_waypoints", {
                'waypoint_pos': [
                                    init_pos
                                ],
                'waypoint_ori': [
                                    init_quat
                                ],
                'waypoint_ind': 1
            }]
            # TODO
            sim_action_list.append(simactionTemplate)
        
        if "franka" in agent_name:
            grabori = self.get_oripose2world(gripper2base=franka_defaults["topdowngrasp"], base2world=self.worldstate[agent_name][3:])
            init_pos, init_quat = self.get_init_EE_pose2world(initEEpose2base_list=franka_defaults["initEEpose2base"], base2world_list=self.worldstate[agent_name])
            simactionTemplate = self.getactionTemplate()
            simactionTemplate[agent_name] = ["franka_open_gripper", {}]
            sim_action_list.append(simactionTemplate)
            
            simactionTemplate = self.getactionTemplate()
            simactionTemplate[agent_name] = ["franka_pick", {
                'attached_prim_path': attached_prim_path,
                'waypoint_pos': [objectpose[:3]],
                'waypoint_ori': [grabori],
                'waypoint_ind': 1
            }]
            sim_action_list.append(simactionTemplate)


            simactionTemplate = self.getactionTemplate()
            simactionTemplate[agent_name] = ["franka_move_with_waypoints", {
                'waypoint_pos': [
                                    init_pos
                                ],
                'waypoint_ori': [
                                    init_quat
                                ],
                'waypoint_ind': 1
            }]
            sim_action_list.append(simactionTemplate)


        return sim_action_list
    
    def grabfrombasket2Simaction(self, agent_name, operation, other):


        attached_prim_path = '/World/' + other[0]  #  + '/base_link'
        sim_action_list = []


        objectpose =  self.worldstate[other[0]]
        agentpose = self.worldstate[agent_name]
        signflag = (np.array(agentpose) - np.array(objectpose)) / np.linalg.norm(np.array(agentpose) - np.array(objectpose), 2)
        
        grabpos = [
            objectpose[0],
            objectpose[1],
            objectpose[2],
        ]

        

        if "aliengo" in agent_name:

            grabori = self.get_oripose2world(gripper2base=aliengo_defaults["topdowngrasp"], base2world=self.worldstate[agent_name][3:])

            # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            # print(grabori)
            init_pos, init_quat = self.get_init_EE_pose2world(initEEpose2base_list=aliengo_defaults["initEEpose2base"], base2world_list=self.worldstate[agent_name][7:])
            
            # print(init_pos)
            # print(init_quat)
            # print("*********************************************!!!!!!!!!!!!!!!!!!!!!")
            simactionTemplate = self.getactionTemplate()
            simactionTemplate[agent_name] = ["aliengo_dog_stand_begin_to_manipulate", {}]
            sim_action_list.append(simactionTemplate)

            simactionTemplate = self.getactionTemplate()
            simactionTemplate[agent_name] = ["aliengo_arm_open_gripper", {}]
            sim_action_list.append(simactionTemplate)

            simactionTemplate = self.getactionTemplate()
            simactionTemplate["quadrotor"] = ["quadrotor_detach_prim_path", {}]
            sim_action_list.append(simactionTemplate)

            simactionTemplate = self.getactionTemplate()
            simactionTemplate[agent_name] = ["aliengo_arm_pick", {
                'attached_prim_path': attached_prim_path,
                'waypoint_pos': [
                                    grabpos,
                                ],
                'waypoint_ori': [
                                    grabori,
                                ],
                'waypoint_ind': 1
            }]
            sim_action_list.append(simactionTemplate)


            simactionTemplate = self.getactionTemplate()
            simactionTemplate[agent_name] = ["aliengo_arm_move_with_waypoints", {
                'waypoint_pos': [
                                    init_pos
                                ],
                'waypoint_ori': [
                                    init_quat
                                ],
                'waypoint_ind': 1
            }]
            sim_action_list.append(simactionTemplate)
        
        if "franka" in agent_name:
            grabori = self.get_oripose2world(gripper2base=franka_defaults["topdowngrasp"], base2world=self.worldstate[agent_name][3:])
            init_pos, init_quat = self.get_init_EE_pose2world(initEEpose2base_list=franka_defaults["initEEpose2base"], base2world_list=self.worldstate[agent_name])
            simactionTemplate = self.getactionTemplate()
            simactionTemplate[agent_name] = ["franka_open_gripper", {}]
            sim_action_list.append(simactionTemplate)
            
            # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", objectpose[:3])
            # Hover above the apple first, then come straight down onto it.
            # franka_pick closes the gripper only once the LAST waypoint is
            # reached, so the grasp still happens at the apple.
            simactionTemplate = self.getactionTemplate()
            simactionTemplate[agent_name] = ["franka_pick", {
                'agent_name': agent_name,
                'attached_prim_path': attached_prim_path,
                "waypoint_pos": [_stacked(grabpos, FRANKA_APPROACH_H), grabpos],
                'waypoint_ori': [grabori, grabori],
                'waypoint_ind': 1
            }]
            sim_action_list.append(simactionTemplate)

            simactionTemplate = self.getactionTemplate()
            simactionTemplate[other[1]] = ["quadrotor_detach_prim_path", {}]
            sim_action_list.append(simactionTemplate)


            # Lift vertically clear of the basket before travelling home, so the
            # apple is not dragged across the rim.
            simactionTemplate = self.getactionTemplate()
            simactionTemplate[agent_name] = ["franka_move_with_waypoints", {
                'waypoint_pos': [
                                    _stacked(grabpos, FRANKA_LIFT_H),
                                    init_pos
                                ],
                'waypoint_ori': [
                                    grabori,
                                    init_quat
                                ],
                'waypoint_ind': 1
            }]
            print('init_pos', init_pos, 'init_quat', init_quat)
            sim_action_list.append(simactionTemplate)


        return sim_action_list
    
    

    def putintobasket2Simaction(self, agent_name, operation, other):


        attached_prim_path = '/World/' + other[0]  #  + '/base_link'
        sim_action_list = []

        quadrotorpose =  self.worldstate[other[1]]
        putpos = [
                    quadrotorpose[0], 
                    quadrotorpose[1], 
                    quadrotorpose[2]+0.3
                ]
        
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # print(putpos)

        if "aliengo" in agent_name:
            putori = self.get_oripose2world(gripper2base=aliengo_defaults["topdowngrasp"], base2world=self.worldstate[agent_name][10:])
            # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            # print(putpos)
            # print(putori)
            init_pos, init_quat = self.get_init_EE_pose2world(initEEpose2base_list=aliengo_defaults["initEEpose2base"], base2world_list=self.worldstate[agent_name][7:])
            
            # print(init_pos)
            # print(init_quat)
            # print("*********************************************!!!!!!!!!!!!!!!!!!!!!")
            simactionTemplate = self.getactionTemplate()
            simactionTemplate[agent_name] = ["aliengo_dog_stand_begin_to_manipulate", {}]
            sim_action_list.append(simactionTemplate)

            simactionTemplate = self.getactionTemplate()
            simactionTemplate[agent_name] = ["aliengo_arm_place", {
                'waypoint_pos': [
                                    putpos,
                                ],
                'waypoint_ori': [
                                    putori,
                                ],
                'waypoint_ind': 1
            }]
            sim_action_list.append(simactionTemplate)


            simactionTemplate = self.getactionTemplate()
            simactionTemplate[other[1]] = ["quadrotor_set_attached_prim_path", {
                                                "agent_name": other[1],
                                                "attached_prim_path": attached_prim_path
                                            }]
            sim_action_list.append(simactionTemplate)

            simactionTemplate = self.getactionTemplate()
            simactionTemplate[agent_name] = ["aliengo_arm_move_with_waypoints", {
                'waypoint_pos': [
                                    init_pos
                                ],
                'waypoint_ori': [
                                    init_quat
                                ],
                'waypoint_ind': 1
            }]
            sim_action_list.append(simactionTemplate)
        
        if "franka" in agent_name:
            putori = self.get_oripose2world(gripper2base=franka_defaults["topdowngrasp"], base2world=self.worldstate[agent_name][3:])
            init_pos, init_quat = self.get_init_EE_pose2world(initEEpose2base_list=franka_defaults["initEEpose2base"], base2world_list=self.worldstate[agent_name])
            simactionTemplate = self.getactionTemplate()
            simactionTemplate[agent_name] = ["franka_place", {
                'waypoint_pos': [
                                    putpos
                                ],
                'waypoint_ori': [
                                    putori
                                ],
                'waypoint_ind': 1
            }]
            sim_action_list.append(simactionTemplate)
            simactionTemplate = self.getactionTemplate()
            simactionTemplate["quadrotor"] = ["quadrotor_set_attached_prim_path", {
                                                "attached_prim_path": attached_prim_path
                                            }]
            sim_action_list.append(simactionTemplate)

            simactionTemplate = self.getactionTemplate()
            simactionTemplate[agent_name] = ["franka_move_with_waypoints", {
                'waypoint_pos': [
                                    init_pos
                                ],
                'waypoint_ori': [
                                    init_quat
                                ],
                'waypoint_ind': 1
            }]
            sim_action_list.append(simactionTemplate)


        print("(((((((((((((())))))))))))))")
        print(sim_action_list)
        return sim_action_list


    def puton2Simaction(self, agent_name, operation, other):
        sim_action_list = []


        putpos = [
            self.task.putpose[other][0],
            self.task.putpose[other][1],
            self.task.putpose[other][2]+0.1,
        ]

        if "aliengo" in agent_name:

            putori = self.get_oripose2world(gripper2base=aliengo_defaults["topdowngrasp"], base2world=self.worldstate[agent_name][10:])
            init_pos, init_quat = self.get_init_EE_pose2world(initEEpose2base_list=aliengo_defaults["initEEpose2base"], base2world_list=self.worldstate[agent_name][7:])
            
            # print(init_pos)
            # print(init_quat)
            # print("*********************************************!!!!!!!!!!!!!!!!!!!!!")
            simactionTemplate = self.getactionTemplate()
            simactionTemplate[agent_name] = ["aliengo_dog_stand_begin_to_manipulate", {}]
            sim_action_list.append(simactionTemplate)

            simactionTemplate = self.getactionTemplate()
            simactionTemplate[agent_name] = ["aliengo_arm_place", {
                'waypoint_pos': [
                                    putpos,
                                ],
                'waypoint_ori': [
                                    putori,
                                ],
                'waypoint_ind': 1
            }]
            sim_action_list.append(simactionTemplate)



            simactionTemplate = self.getactionTemplate()
            simactionTemplate[agent_name] = ["aliengo_arm_move_with_waypoints", {
                'waypoint_pos': [
                                    init_pos
                                ],
                'waypoint_ori': [
                                    init_quat
                                ],
                'waypoint_ind': 1
            }]
            sim_action_list.append(simactionTemplate)

        if "franka" in agent_name:
            putori = self.get_oripose2world(gripper2base=franka_defaults["topdowngrasp"], base2world=self.worldstate[agent_name][3:])
            init_pos, init_quat = self.get_init_EE_pose2world(initEEpose2base_list=franka_defaults["initEEpose2base"], base2world_list=self.worldstate[agent_name])
            # Travel to a hover above the destination, then lower the apple onto
            # it. franka_place opens the gripper once the LAST waypoint is reached,
            # so the release still happens at the target -- but now from directly
            # above it, rather than at the end of a diagonal swing that used to
            # clip the basket and drop the apple short.
            simactionTemplate = self.getactionTemplate()
            simactionTemplate[agent_name] = ["franka_place", {
                'waypoint_pos': [
                                    _stacked(putpos, FRANKA_APPROACH_H),
                                    putpos
                                ],
                'waypoint_ori': [
                                    putori,
                                    putori
                                ],
                'waypoint_ind': 1
            }]
            sim_action_list.append(simactionTemplate)


            # Retreat straight up before going home, so the gripper does not sweep
            # the apple back off the surface it was just placed on.
            simactionTemplate = self.getactionTemplate()
            simactionTemplate[agent_name] = ["franka_move_with_waypoints", {
                'waypoint_pos': [
                                    _stacked(putpos, FRANKA_LIFT_H),
                                    init_pos
                                ],
                'waypoint_ori': [
                                    putori,
                                    init_quat
                                ],
                'waypoint_ind': 1
            }]
            sim_action_list.append(simactionTemplate)

        return sim_action_list

    def opendoor2Simaction(self, agent_name, operation, other):
        sim_action_list = []


        openpos_list, openori_list = self.opendoorTrajectory(self.task.opendoorpose[other])
        
        grabori = self.get_oripose2world(gripper2base=aliengo_defaults["frontreargrasp"], base2world=self.worldstate[agent_name][10:])
        init_pos, init_quat = self.get_init_EE_pose2world(initEEpose2base_list=aliengo_defaults["initEEpose2base"], base2world_list=self.worldstate[agent_name][7:])
        
        # print(init_pos)
        # print(init_quat)
        # print("*********************************************!!!!!!!!!!!!!!!!!!!!!")
        simactionTemplate = self.getactionTemplate()
        simactionTemplate[agent_name] = ["aliengo_dog_stand_begin_to_manipulate", {}]
        sim_action_list.append(simactionTemplate)

        simactionTemplate = self.getactionTemplate()
        simactionTemplate[agent_name] = ["aliengo_arm_open_gripper", {}]
        sim_action_list.append(simactionTemplate)
       
        simactionTemplate = self.getactionTemplate()
        simactionTemplate[agent_name] = ["aliengo_arm_move_with_waypoints", {
            'waypoint_pos': openpos_list,
            'waypoint_ori': grabori,
            'waypoint_ind': len(openpos_list)
        }]
        sim_action_list.append(simactionTemplate)

        simactionTemplate = self.getactionTemplate()
        simactionTemplate[agent_name] = ["aliengo_arm_move_with_waypoints", {
            'waypoint_pos': [
                                init_pos
                            ],
            'waypoint_ori': [
                                init_quat
                            ],
            'waypoint_ind': 1
        }]
        sim_action_list.append(simactionTemplate)

        ############################################
        
        simactionTemplate = self.getactionTemplate()
        new_other = other.replace('-', '/')
        simactionTemplate[agent_name] = ["aliengo_arm_door_open", {
            'articulation_path': '/World/' + new_other,
            # 'link_name': other2,
        }]
        sim_action_list.append(simactionTemplate)
        ##########################################################
        return sim_action_list
    
    
    def takeoff2Simaction(self, agent_name, operation, other):
        sim_action_list = []

        simactionTemplate = self.getactionTemplate()
        simactionTemplate[agent_name] = ["quadrotor_takeoff", {}]
        sim_action_list.append(simactionTemplate)

        return sim_action_list

    def land2Simaction(self, agent_name, operation, other):
        self.landmark = other
        sim_action_list = []

        # landpose = self.task.landpose[other]
        landpos = self.task.landpose[other][:3]
        landori = self.quatrear2quatfirst(self.task.landpose[other][3:])
        print("landpos:", landpos)
        print("landori:", landori)
        
        simactionTemplate = self.getactionTemplate()
        

        simactionTemplate = self.getactionTemplate()
        simactionTemplate[agent_name] = ["quadrotor_land", {
                'waypoint_pos': [
                                    landpos,
                                ],
                'waypoint_ori': [
                                    landori,
                                ],
                'waypoint_ind': 1
            }]
        sim_action_list.append(simactionTemplate)

        return sim_action_list


    def opendoorTrajectory(self, waypoints, deltastep=0.05):
        
        openpos_list = []
        openori_list = []
        cnt = len(waypoints)
        if cnt == 1:
            openpos_list.append(waypoints[0][:3])
            openori_list.append(waypoints[0][3:])
            return openpos_list, openori_list
        
        for i in range(cnt-1):
            posA = np.array(waypoints[i][:3])
            posB = np.array(waypoints[i+1][:3])
            
            deltapos = posB - posA
            deltasteps = np.ceil(np.abs(posB-posA) / deltastep).astype(int)
            max_step = np.max(deltasteps)
            for k in range(max_step):
                _pos = []
                for dim in range(3):
                    x = posA[dim] + np.sign(deltapos[dim]) * deltastep * k
                    if np.sign(deltapos[dim]) > 0 and x > posB[dim]:
                        x = posB[dim]
                    if np.sign(deltapos[dim]) < 0 and x < posB[dim]:
                        x = posB[dim]
                    _pos.append(x)

                openpos_list.append(_pos)
                openori_list.append(self.quatrear2quatfirst(waypoints[i+1][3:]))
        return openpos_list, openori_list

    def pathplanning(self, currentpose, goalpose):
        return  {
                    "waypoint_pos": [
                        goalpose[:3]
                    ],
                    "waypoint_ori": [
                        [goalpose[6], goalpose[3], goalpose[4], goalpose[5]]
                    ],
                    "waypoint_ind": 1
                }
    
    def quatrear2quatfirst(self, quatrear):
        return [quatrear[3], quatrear[0], quatrear[1], quatrear[2]]

    def quatfirst2quatrear(self, quatfirst):
        return [quatfirst[1], quatfirst[2], quatfirst[3], quatfirst[0]]

    def get_oripose2world(self, gripper2base, base2world):
        # print("gripper2base:", gripper2base)
        # print("base2world:", base2world)
        gripper2world = R.from_quat(base2world) * R.from_quat(gripper2base)
        quat_rear = gripper2world.as_quat()
        return self.quatrear2quatfirst(quat_rear)

    def get_init_EE_pose2world(self, initEEpose2base_list, base2world_list):
        
        # print("EE2basePos: ", initEEpose2base_list)
        EE2base = np.eye(4)
        EE2base[:3, 3] = np.array(initEEpose2base_list[:3])
        EE2base[:3, :3] = R.from_quat(np.array(initEEpose2base_list[3:])).as_matrix()



        # print("base2worldPos: ", base2world_list)
        base2world = np.eye(4)
        base2world[:3, 3] = np.array(base2world_list[:3])
        base2world[:3, :3] = R.from_quat(np.array(base2world_list[3:])).as_matrix()
        EE2world = np.matmul(base2world, EE2base)
        init_pos = EE2world[:3, 3]
        init_quat = R.from_matrix(EE2world[:3, :3]).as_quat()[[3, 0, 1, 2]]

        return init_pos, init_quat

    def get_landpoint_name(self, agent_name):
        alllandname = list(self.task.landpose.keys())
        alllandpos = np.array(list(self.task.landpose.values()))[:, :2]
        quadrotorpos = self.worldstate[agent_name][:2]

        closeInd = np.argmin(np.linalg.norm(alllandpos - quadrotorpos, axis=-1))
    
        return "landpoint-" + alllandname[closeInd]

class ActionPublishandResultSubscribe(Node):
    def __init__(self, task_name):
        super().__init__('LLM')
        self.task_name = task_name

        self.task = eval("defaults_%s" % self.task_name)
        self.TextPlanList = eval(self.task_name+"_TextPlanList")
        # print("## Task Name: ", self.task_name)
        # for i in range(len(self.TextPlanList)):
        #     print("Step %d: [agent] %s [op] %s %s" % (i+1, self.TextPlanList[i][0], self.TextPlanList[i][1], self.TextPlanList[i][2]))
        # print("#############################")

        self.translator = Translator(task_name)
        self.sim_action_list = []
        self.text_action_cnt = 0

        self._sub = self.create_subscription(
            Result, 'resultTopic', self.callback, LATCHED_QOS)
        # Define the publisher's name, message type, and QoS profile.
        self._pub = self.create_publisher(Action, 'actionTopic', LATCHED_QOS)

        self.warmup = True
        self.pub_flag = False
    


    # action_list, pub action
    def callback(self, result_msg):
        assert isinstance(result_msg, Result)
        if self.warmup:
            self.warmup = False
        else:
            self.sim_action_list.pop(0)
            
            print("*******************************")
            print("Receive result:")
            print(result_msg)
            print("*******************************")
            
            # print(worldmodel.get_state(self.task_name))
            # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        # check sub_data
        # if not self.check_success_from_result_msg(result_msg):
        #     # LLM needs to re-plan
        #     return
        
        if len(self.sim_action_list)==0:
            self.TextPlanList.pop(0)
            if len(self.TextPlanList):
                print("************************************************************************************")
                next_text_action = self.TextPlanList[0]
                self.text_action_cnt += 1
                print('### Text-based Action %d: [agent] %s [op] %s %s' % (self.text_action_cnt, next_text_action[0], next_text_action[1], next_text_action[2]))
                self.sim_action_list = self.translator.translate(next_text_action)
            print(self.sim_action_list)
            # input("Press Enter to continue...")
        if len(self.sim_action_list):
            next_sim_action = self.sim_action_list[0]
            print("next_sim_action: ", next_sim_action)
            action_msg = self.encode_to_action_msg(next_sim_action)
            # print(action_msg)
            # input("Press Enter to continue...")
            # rospy.loginfo("Publish the next action") # Output log information to the screen and write it to the rosout node.
            # print(action_msg)
            self._pub.publish(action_msg) # Publish messages to the topic.
            self.sub_flag = False
            self.pub_flag = True
        


    


    def encode_to_action_msg(self, next_step_action):

        def _parse(next_step_action):
            func_args_msg = FuncAndArgs()
            if len(next_step_action):
                func_args_msg.has_func = True
                func_args_msg.func_name = next_step_action[0]
                
                args_msg = Args()
                # no args
                if next_step_action[1] == dict():
                    args_msg.has_args = False
                else:
                    args_msg.has_args = True
                    
                    if 'attached_prim' in next_step_action[0]:
                        args_msg.attached_prim_path = next_step_action[1]["attached_prim_path"]
                    elif 'door_open' in next_step_action[0]:
                        args_msg.attached_prim_path = next_step_action[1]["articulation_path"]

                    else:
                        waypoint_pos_msg = Float64MultiArray()
                        waypoint_ori_msg = Float64MultiArray()
                        # print(list(np.array(next_step_action[1]["waypoint_pos"], dtype=np.float64).flatten()))
                        waypoint_pos_msg.data = np.array(next_step_action[1]["waypoint_pos"], dtype=np.float64).flatten().tolist()
                        waypoint_ori_msg.data = np.array(next_step_action[1]["waypoint_ori"], dtype=np.float64).flatten().tolist()

                        args_msg.waypoint_pos = waypoint_pos_msg  # list(np.array(next_step_action[1]["waypoint_pos"]).flatten())
                        args_msg.waypoint_ori = waypoint_ori_msg  # list(np.array(next_step_action[1]["waypoint_ori"]).flatten())
                        args_msg.waypoint_ind = 0
                        if 'pick' in next_step_action[0]:
                            args_msg.attached_prim_path = next_step_action[1]["attached_prim_path"]
                    # print(args_msg)
                    # print("####################################")

                func_args_msg.args = args_msg
            else:
                func_args_msg.has_func = False
            return func_args_msg

        action_msg = Action()
        # print(next_step_action)
        # potential_agent_name_list = ['franka_0', 'franka_1', 'franka_2', 'aliengo_0', 'aliengo_1', 'aliengo_2']

        for agent in self.task.agent_name_list:
            exec("action_msg.%s = _parse(next_step_action[agent])" % agent)
                
        # print(action_msg)

        # action_msg.franka = franka_func_args_msg
        # action_msg.aliengo = aliengo_func_args_msg
        # action_msg.quadrotor = quadrotor_func_args_msg

        return action_msg

    def check_success_from_result_msg(self, result_msg):

        return True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, default='Merom_1_int_Task1')
    parser.add_argument('--mode', type=str, default='hardcoded', choices=['hardcoded', 'ws', 'ws_test'],
                        help='hardcoded = original TextPlanList, ws = receive via WebSocket + ROS, ws_test = WebSocket only (no ROS, echoes success)')
    parser.add_argument('--ws_port', type=int, default=8765,
                        help='WebSocket server port (only used in ws mode)')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode in ('ws', 'ws_test'):
        main_ws(args)
    else:
        rclpy.init()
        node = ActionPublishandResultSubscribe(task_name=args.task_name)
        try:
            rclpy.spin(node)
        except (KeyboardInterrupt, ExternalShutdownException):
            # Ctrl-C or SIGTERM: rclpy has already torn the context down.
            pass
        finally:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


def load_action_macros(task_name):
    """One planner action -> the sim text-actions that carry it out.

    See Benchmark/action_macros.json. An absent file or task means "no macros",
    which is correct for Merom: every action it plans maps 1:1 already.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "..", "..", "..", "action_macros.json")
    path = os.path.normpath(path)
    if not os.path.exists(path):
        print("[WS] no action_macros.json at %s; passing actions through" % path,
              flush=True)
        return {}
    with open(path) as f:
        table = json.load(f)
    macros = {k: v for k, v in table.get(task_name, {}).items()
              if not k.startswith("_")}
    print("[WS] %d action macro(s) for %s" % (len(macros), task_name), flush=True)
    return macros


def expand_action(action_str, agent, op, target, macros):
    """Return the list of [agent, op, target] the bridge should actually run.

    `action_str` is the planner's action after name mapping, which is the form the
    macro keys are written in. No entry -> the action passes through unchanged.
    """
    key = " ".join(str(action_str).split())
    # Agent-scoped entries win over bare ones. Both the dog and the arm grab the
    # kebab in the house task -- the dog out of the fridge, the arm out of the
    # drone's basket -- so the same action string has to expand differently
    # depending on who is doing it.
    steps = macros.get("%s|%s" % (agent, key)) or macros.get(key)
    if not steps:
        return [[agent, op, target]]
    out = []
    for step in steps:
        if isinstance(step, (list, tuple)):
            out.append([agent, step[0], step[1]])
        else:
            s_op, _, s_target = str(step).partition(" ")
            out.append([agent, s_op, s_target.strip()])
    print("[WS] macro: %r -> %d sim action(s)" % (key, len(out)), flush=True)
    return out


def main_ws(args):
    """
    WebSocket mode: receive one text action at a time from PEFA,
    translate it, execute via ROS (or echo in test mode), send back result.
    """
    import asyncio
    import websockets
    import json
    import threading

    test_mode = (args.mode == 'ws_test')

    node = None
    if not test_mode:
        rclpy.init()
        node = Node('LLM_WS')

    task_name = args.task_name
    task = eval("defaults_%s" % task_name)
    translator = Translator(task_name)
    macros = load_action_macros(task_name)

    # ROS setup (only if not test mode)
    pub = None
    result_event = threading.Event()
    # last_result is intentionally pessimistic by default — the callback must fill it.
    last_result = {
        "success": False,
        "info": "no_result_yet",
        "per_agent": {},         # agent_name -> {"has_result", "success", "info"}
        "any_reported": False,   # True if at least one agent reported has_result=True
    }

    if not test_mode:
        pub = node.create_publisher(Action, 'actionTopic', LATCHED_QOS)

        def result_callback(result_msg):
            assert isinstance(result_msg, Result)
            per_agent = {}
            failed_agents = []
            reported_agents = []
            any_reported = False
            for agent_name in task.agent_name_list:
                ri = getattr(result_msg, agent_name)
                entry = {
                    "has_result": bool(ri.has_result),
                    "success": bool(ri.success),
                    "info": ri.info or "",
                }
                per_agent[agent_name] = entry
                if entry["has_result"]:
                    any_reported = True
                    reported_agents.append(agent_name)
                    if not entry["success"]:
                        failed_agents.append(agent_name)

            # overall success rule: at least one agent reported AND none of the
            # reporting agents reported failure.
            overall_success = any_reported and (len(failed_agents) == 0)

            if failed_agents:
                info_str = "; ".join(
                    f"{a}: {per_agent[a]['info']}" for a in failed_agents
                )
            elif any_reported:
                info_str = "; ".join(
                    f"{a}: {per_agent[a]['info']}" for a in reported_agents
                )
            else:
                # No agent reported — treat as an anomaly, not silent success.
                info_str = "no_agent_reported_has_result"

            last_result["success"] = overall_success
            last_result["info"] = info_str
            last_result["per_agent"] = per_agent
            last_result["any_reported"] = any_reported
            print(
                f"[WS][result_cb] any_reported={any_reported} "
                f"ok={overall_success} info={info_str}",
                flush=True,
            )
            result_event.set()

        node.create_subscription(Result, 'resultTopic', result_callback, LATCHED_QOS)

        # Unlike rospy, rclpy has no implicit background spin thread.  The
        # asyncio WebSocket server owns the main thread, so spin the node on its
        # own thread or result_callback would never fire and every action would
        # hit the 60 s timeout.
        def _spin():
            try:
                rclpy.spin(node)
            except (KeyboardInterrupt, ExternalShutdownException):
                pass

        _spin_thread = threading.Thread(target=_spin, daemon=True)
        _spin_thread.start()

    def encode_to_action_msg(next_step_action):
        def _parse(nsa):
            func_args_msg = FuncAndArgs()
            if len(nsa):
                func_args_msg.has_func = True
                func_args_msg.func_name = nsa[0]
                args_msg = Args()
                if nsa[1] == dict():
                    args_msg.has_args = False
                else:
                    args_msg.has_args = True
                    if 'attached_prim' in nsa[0]:
                        args_msg.attached_prim_path = nsa[1]["attached_prim_path"]
                    elif 'door_open' in nsa[0]:
                        args_msg.attached_prim_path = nsa[1]["articulation_path"]
                    else:
                        waypoint_pos_msg = Float64MultiArray()
                        waypoint_ori_msg = Float64MultiArray()
                        waypoint_pos_msg.data = np.array(nsa[1]["waypoint_pos"], dtype=np.float64).flatten().tolist()
                        waypoint_ori_msg.data = np.array(nsa[1]["waypoint_ori"], dtype=np.float64).flatten().tolist()
                        args_msg.waypoint_pos = waypoint_pos_msg
                        args_msg.waypoint_ori = waypoint_ori_msg
                        args_msg.waypoint_ind = 0
                        if 'pick' in nsa[0]:
                            args_msg.attached_prim_path = nsa[1]["attached_prim_path"]
                func_args_msg.args = args_msg
            else:
                func_args_msg.has_func = False
            return func_args_msg

        action_msg = Action()
        for agent in task.agent_name_list:
            exec("action_msg.%s = _parse(next_step_action[agent])" % agent)
        return action_msg

    # Version-agnostic handler: works with websockets>=10 (single arg) AND <10 (websocket, path).
    async def handle_pefa(websocket, path=None):
        import traceback
        peer = getattr(websocket, "remote_address", "?")
        print(f"[WS] PEFA connected from {peer} (path={path})", flush=True)

        try:
            if not test_mode:
                # Warmup: the sim side (action_subscriber.initial_pub) publishes ONE
                # empty Result() as soon as it sees /actionTopic. We just wait for
                # that handshake — do NOT treat the empty Result as failure.
                # If the event has already fired before this handler was attached
                # (e.g. the sim started first and published immediately), the wait
                # returns right away.
                print("[WS] Waiting for sim warmup Result() handshake...", flush=True)
                ready = result_event.wait(timeout=120)
                result_event.clear()
                if not ready:
                    err = ("Sim warmup timed out (no Result msg in 120s). "
                           "Is OmniGibson/action_subscriber running and has it seen /actionTopic?")
                    print(f"[WS] {err}", flush=True)
                    await websocket.send(json.dumps({"success": False, "info": err}))
                    return
                print(f"[WS] Sim ready! (warmup info={last_result.get('info','')})", flush=True)

            async for message in websocket:
                try:
                    data = json.loads(message)
                except Exception as e:
                    await websocket.send(json.dumps({"success": False, "info": f"bad json: {e}"}))
                    continue
                print(f"[WS RECV] {data}", flush=True)

                import re
                action_str = data.get("action", "")
                og_agent = data.get("agent", "")

                action_match = re.match(r'\[(\w+)\]', action_str)
                action_name = action_match.group(1) if action_match else ""

                remaining = re.sub(r'\[\w+\]\s*', '', action_str).strip()
                parts = re.split(r'\s+(?:into|on)\s+', remaining)
                if len(parts) == 1:
                    target_str = parts[0]
                else:
                    target_str = [p.strip() for p in parts]

                # Normalize object names that the LLM may produce but the
                # translator does not recognise.  E.g. "quadrotor_0_basket"
                # should become "quadrotor_0" for the sim-action lookup,
                # because the basket is attached to the quadrotor agent and
                # is not a separate worldstate entity.
                _basket_alias = {
                    "quadrotor_0_basket": "quadrotor_0",
                    "quadrotor_1_basket": "quadrotor_1",
                    "quadrotor_2_basket": "quadrotor_2",
                }
                if isinstance(target_str, list):
                    target_str = [_basket_alias.get(t, t) for t in target_str]
                else:
                    target_str = _basket_alias.get(target_str, target_str)

                # One planner action can need several sim commands -- see
                # action_macros.json. Merom expands 1:1; the house demo's fridge
                # and terrace flight do not.
                text_actions = expand_action(action_str, og_agent, action_name,
                                             target_str, macros)
                print(f"[WS] Text action(s) for translator: {text_actions}", flush=True)

                if test_mode:
                    # No ROS — just echo success
                    print(f"[WS TEST] Would translate and execute: {text_actions}", flush=True)
                    response = json.dumps({"success": True, "info": "test_mode"})
                    print(f"[WS SEND] {response}", flush=True)
                    await websocket.send(response)
                    continue

                # Translate every expanded step up front, so a macro with a bad
                # target fails before any of it has moved a robot.
                sim_action_list = []
                translate_error = None
                for text_action in text_actions:
                    try:
                        sim_action_list.extend(translator.translate(text_action))
                    except Exception as e:
                        tb = traceback.format_exc()
                        print(f"[WS] Translation error: {e}\n{tb}", flush=True)
                        translate_error = f"translation_error: {e}"
                        break
                if translate_error:
                    await websocket.send(json.dumps({"success": False, "info": translate_error}))
                    continue

                if not sim_action_list:
                    await websocket.send(json.dumps({"success": False, "info": "empty_sim_action_list"}))
                    continue

                # Execute each sim command sequentially
                overall_success = True
                fail_info = ""
                timed_out = False
                for sim_action in sim_action_list:
                    print(f"[WS] Executing sim action: {list(sim_action.keys())}", flush=True)
                    try:
                        action_msg = encode_to_action_msg(sim_action)
                    except Exception as e:
                        overall_success = False
                        fail_info = f"encode_error: {e}"
                        break
                    result_event.clear()
                    # stale-result guard: reset so a timeout or an empty Result
                    # (all has_result=False) is detected honestly.
                    last_result["success"] = False
                    last_result["info"] = "awaiting_sim_result"
                    last_result["per_agent"] = {}
                    last_result["any_reported"] = False
                    pub.publish(action_msg)
                    got_result = result_event.wait(timeout=SIM_RESULT_TIMEOUT_S)
                    if not got_result:
                        overall_success = False
                        fail_info = "sim_result_timeout_%gs" % SIM_RESULT_TIMEOUT_S
                        timed_out = True
                        break
                    if not last_result.get("any_reported", False):
                        # sim published a Result, but NO agent flagged has_result=True.
                        # That almost certainly means the sim step finished without
                        # the expected agent reporting — treat as failure so the
                        # oracle can re-plan instead of silently looping.
                        overall_success = False
                        fail_info = "sim_result_had_no_reporting_agent"
                        break
                    if not last_result["success"]:
                        overall_success = False
                        fail_info = last_result.get("info", "sim_reported_failure")
                        break

                response = json.dumps({
                    "success": overall_success,
                    "info": fail_info if not overall_success else last_result.get("info", ""),
                    "timed_out": timed_out,
                    "per_agent": last_result.get("per_agent", {}),
                })
                print(f"[WS SEND] {response}", flush=True)
                await websocket.send(response)
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[WS] Handler crashed: {e}\n{tb}", flush=True)
            try:
                await websocket.send(json.dumps({"success": False, "info": f"handler_crash: {e}"}))
            except Exception:
                pass
        finally:
            print(f"[WS] PEFA disconnected ({peer})", flush=True)

    async def run_server():
        print(f"[WS] Starting WebSocket server on 0.0.0.0:{args.ws_port} (mode={args.mode})", flush=True)
        # Keepalive is DISABLED, not merely lengthened. The handler does its sim
        # work synchronously, so a long action (the house land_on runs ~190s)
        # blocks this event loop outright: the library cannot service a pong in
        # time and tears the connection down with 1011 "keepalive ping timeout",
        # orphaning the planner mid-run. ping_interval=20/ping_timeout=120 was
        # not enough under websockets>=14, which enforces the deadline strictly.
        # Both ends are local processes, so liveness detection buys nothing here.
        async with websockets.serve(
            handle_pefa, "0.0.0.0", args.ws_port,
            ping_interval=None, ping_timeout=None, max_size=None,
        ):
            print("[WS] Server ready. Awaiting PEFA client...", flush=True)
            await asyncio.Future()  # run forever

    asyncio.run(run_server())


if __name__ == '__main__':
    main()


