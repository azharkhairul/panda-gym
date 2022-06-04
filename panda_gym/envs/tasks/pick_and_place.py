
from hashlib import new
from cv2 import threshold
import numpy as np
from gym import utils
from PIL import Image
import os, sys
from io import StringIO
import math
import cv2
from tempfile import TemporaryFile
from scipy.sparse import csr_matrix

from panda_gym.envs.core import Task
from panda_gym.utils import distance, get_view, thresh_callback, mask2binary, eef_binary, compute_M, get_indices_sparse
import pybullet as p
import pybullet_utils.bullet_client as bc
import pybullet_data

class PickAndPlace(Task):
    def __init__(
        self,
        sim,
        reward_type="sparse",
        distance_threshold=0.05,
        goal_xy_range=0.2,
        goal_z_range=0.08,
        obj_xy_range=0.2,
    ):

        self.background_color = [val / 255 for val in (116, 160, 216)]

        options = "--background_color_red={} \
                    --background_color_green={} \
                    --background_color_blue={}".format(
            *self.background_color
        )
        self.connection_mode = p.DIRECT     #"""p.GUI if render else """
        self.physics_client = bc.BulletClient(connection_mode=self.connection_mode, options=options)
        self.physics_client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        self.physics_client.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)

        self.width=45
        self.height=30

        self.sim = sim
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.object_size = 0.04
        self.goal_range_low = np.array([-goal_xy_range / 2, -goal_xy_range / 2, 0])
        self.goal_range_high = np.array([goal_xy_range / 2, goal_xy_range / 2, goal_z_range])
        self.obj_range_low = np.array([-obj_xy_range / 2, -obj_xy_range / 2, 0])
        self.obj_range_high = np.array([obj_xy_range / 2, obj_xy_range / 2, 0])
        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=[0, 0, 0], distance=0.9, yaw=45, pitch=-30)

    def _create_scene(self):
        self.sim.create_plane(z_offset=-0.4)
        
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.sim.create_box(
            body_name="object",
            half_extents=[
                self.object_size / 2,
                self.object_size / 2,
                self.object_size / 2,
            ],
            mass=2,
            position=[0.0, 0.0, self.object_size / 2],
            rgba_color=[0.9, 0.1, 0.1, 1],
            friction=5,  # increase friction. For some reason, it helps a lot learning
        )
        self.sim.create_box(
            body_name="goal",
            half_extents=[
                self.object_size / 2,
                self.object_size / 2,
                self.object_size / 2,
            ],
            mass=0.0,
            ghost=True,
            position=[0.0, 0.0, 0.05],
            rgba_color=[0.9, 0.4, 0.1, 0.3], 
            
        )
        self.sim.create_slanted(z_offset = 0)
        
    def get_goal(self):

        while True:
            if sum(self.goal1) == 0 or sum(self.goal2) == 0 or sum(self.goal3): #make sure the picture is not empty
                self.goal1 = []
                self.goal2 = []
                self.goal3 = []
                # self.goal1, self.goal2 , self.goal3 = mask2binary(self, a = "goal")
                self.goal1, self.goal2 = mask2binary(self, a = "goal")
                # eef = eef_binary(self, a = "goal")
                # self.eef_goal = eef
            else:
                self.rew_thresh = (sum(self.goal1) + sum(self.goal2) + sum(self.goal3)) * 0.05
                self.des_goal = np.concatenate([self.goal1.copy(),self.goal2.copy(),self.goal3.copy(), self.eef_goal.copy()])
                self.goal1 = np.array(self.goal1)
                self.goal2 = np.array(self.goal2) 
                self.goal3 = np.array(self.goal3)
                self.eef_goal = np.array(self.eef_goal)
                # return np.concatenate([self.goal1.copy(),self.goal2.copy(), self.goal3.copy(), self.eef_goal.copy()])
                return np.concatenate([self.goal1.copy(),self.goal2.copy(), self.eef_goal.copy()])
    
    
    
    def get_obs(self):
        # position, rotation of the object
        # object_position = np.array(self.sim.get_base_position("object"))
        # object_rotation = np.array(self.sim.get_base_rotation("object"))
        # object_velocity = np.array(self.sim.get_base_velocity("object"))
        # object_angular_velocity = np.array(self.sim.get_base_angular_velocity("object"))
      
        # new_image1, new_image2 , new_image3= mask2binary(self, a = "obj")   #retrieve the binary img of the object
        new_image1, new_image2 , = mask2binary(self, a = "obj")
        eef = eef_binary(self, a = "obj")    #retrieve the binary img from the eef
        
        observation = np.concatenate(
            [
                # object_rotation,
                # object_position,
                # object_velocity,
                # object_angular_velocity,

                new_image1,
                new_image2,
                # new_image3,
                eef,
            ]
        )
        return observation

    def get_achieved_goal(self):

        # self.obj1, self.obj2, self.obj3 = mask2binary(self, a = "obj")
        self.obj1, self.obj2 = mask2binary(self, a = "obj")
        self.eef_obj = eef_binary(self, a = "obj")
        self.obj1 = np.array(self.obj1)
        self.obj2 = np.array(self.obj2) 
        self.obj3 = np.array(self.obj3)
        self.eef_obj = np.array(self.eef_obj)

        # return np.concatenate([self.obj1,self.obj2,self.obj3, self.eef_obj])
        return np.concatenate([self.obj1,self.obj2, self.eef_obj])


    def reset(self):
        self.goal = self._sample_goal()
        object_position = self._sample_object()
        self.sim.set_base_pose("goal", self.goal, [0, 0, 0, 1])
        self.sim.set_base_pose("object", object_position, [0, 0, 0, 1])
        self.goal1 = []
        self.goal2 = []
        self.goal3 = []
        self.obj3 = []
        self.eef_goal = []
        # self.goal1, self.goal2, self.goal3 = mask2binary(self, a = "goal") #picture for the goal
        self.goal1, self.goal2 = mask2binary(self, a = "goal") #picture for the goal
        self.eef_goal = eef_binary(self, a = "goal")
        
    def _sample_goal(self):
        """Randomize goal."""
        goal = [0.0, 0.0, self.object_size *2]  #make sure the goal is always in the air
        noise = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        if self.np_random.random() < 0.1:
            noise[2] = 0.0
        goal += noise
        return goal

    def _sample_object(self):
        """Randomize start position of object."""
        object_position = [0.0,0.0, self.object_size /2]
        noise = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        object_position += noise
        return object_position

    def is_success(self, achieved_goal, desired_goal):
        d = distance(achieved_goal, desired_goal)

        return (d >= self.rew_thresh).astype(np.float32)


    def compute_reward(self, achieved_goal, desired_goal, info):
        d = distance(achieved_goal, desired_goal)
        
        if sum(self.eef_obj) < 8:
            penalty = -0.5
        else: 
            penalty = 0

        return -(d < self.rew_thresh).astype(np.float32) + penalty
