
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
        goal_z_range=0.1,
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

        # self.width=68
        # self.height=45
        self.width=45
        self.height=30
        # self.width=23
        # self.height=15

        self.goal_sum = 0
        self.obs_sum = 0
        # self.rew_thresh = 20
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
        # px1,px2,mask1,mask2 = get_view(self)

        # mask_obj1 = Image.fromarray(mask1) 
        # new_image1 = []
        # for item in mask_obj1.getdata():
        #     if (
        #         item == 4.0 #object              
        #     ):
        #         new_image1.append((1))
        #     else:
        #         new_image1.append((0))

        # BW_obj1 = Image.new('1', (self.width, self.height))
        # BW_obj1.putdata(new_image1)
        # BW_obj1.save("goal1.jpg")

        # mask_obj2 = Image.fromarray(mask2) 
        # new_image2 = []
        # for item in mask_obj2.getdata():

        #     if (item == 4.0 ):#object        
        #         new_image2.append((1))
        #     else:
        #         new_image2.append((0))

        # BW_obj2 = Image.new('1', (self.width, self.height))
        # BW_obj2.putdata(new_image2)
        # BW_obj2.save("goal2.jpg")
        # new_image1, new_image2 = mask2binary(self, a = "goal")
        # return np.concatenate([new_image1,new_image2])
        while True:
            if sum(self.goal1) == 0 or sum(self.goal2) == 0: 
                self.goal1 = []
                self.goal2 = []
                self.goal3 = []
                self.eef_goal = []
                self.goal1, self.goal2 , self.goal3 = mask2binary(self, a = "goal")
                eef = eef_binary(self, a = "goal")
                self.eef_goal = eef
            else:
                self.rew_thresh = (sum(self.goal1) + sum(self.goal2) + sum(self.goal3)) * 0.05
                self.des_goal = np.concatenate([self.goal1.copy(),self.goal2.copy(),self.goal3.copy(), self.eef_goal.copy()])
                self.rew_goal1 = get_indices_sparse(self.goal1.copy())
                self.rew_goal2 = get_indices_sparse(self.goal2.copy())
                self.rew_goal3 = get_indices_sparse(self.goal3.copy())
                # print(sum(self.goal1.copy()))
                self.goal1 = np.array(self.goal1)
                self.goal2 = np.array(self.goal2) 
                self.goal3 = np.array(self.goal3)
                self.eef_goal = np.array(self.eef_goal)
                return np.concatenate([self.goal1.copy(),self.goal2.copy(), self.goal3.copy(), self.eef_goal.copy()])
    
    
    
    def get_obs(self):
        # position, rotation of the object
        # object_position = np.array(self.sim.get_base_position("object"))
        # object_rotation = np.array(self.sim.get_base_rotation("object"))
        # object_velocity = np.array(self.sim.get_base_velocity("object"))
        # object_angular_velocity = np.array(self.sim.get_base_angular_velocity("object"))
        # px1,px2,mask1,mask2 = get_view(self)

        # mask_obj1 = Image.fromarray(mask1) 
        # new_image1 = []
        # for item in mask_obj1.getdata():
        #     if (
        #         # item == -1.0 #the wall
        #         # item == 0.0 #robot
        #         # item == 1.0 #floor
        #         # item == 2.0 #table 
        #         item == 3.0 #object 
        #         # item == 4.0 #goal               
        #     ):
        #         new_image1.append((1))
        #     else:
        #         new_image1.append((0))

        # # BW_obj1 = Image.new('1', (self.width, self.height))
        # # BW_obj1.putdata(new_image1)
        # # BW_obj1.save("obs1.jpg")

        # mask_obj2 = Image.fromarray(mask2) 
        # new_image2 = []
        # for item in mask_obj2.getdata():

        #     if (item == 3.0 ):#object        
        #         new_image2.append((1))
        #     else:
        #         new_image2.append((0))

        
        # BW_obj2 = Image.new('1', (self.width, self.height))
        # BW_obj2.putdata(new_image2)
        # BW_obj2.save("obs2.jpg")
        
        # print(np.shape(BW_obj1))
        
        # something = cv.fromarray(BW_obj1) 
        # BW_obj2.open("came2.jpg")
        # src = cv2.imread("came2.jpg")
        # shi = np.array(new_image2,dtype='uint8')
        # new_1 = np.reshape(shi, (self.height, self.width))
        # fff = thresh_callback(shi)
        # rgb1 = Image.fromarray(px1) 
        # rgb1.save('RGB image for object (camera1).png')
        # rgb2 = Image.fromarray(px2) 
        # rgb2.save('RGB image for object (camera2).png')


        new_image1, new_image2 , new_image3= mask2binary(self, a = "obj")
        eef = eef_binary(self, a = "obj")
        
        
        
        # points = []
        # for obj in np.unique(new_image1):
        #     if obj == 0:
        #         continue
        #     points.append(np.mean(np.where(new_image1 == obj), axis=1))
        # points_np = np.array(points)
        
        # for _ in range(100):
        #     points = self.get_indices_sparse(new_image1,mask_s)
        #     print(points)
        # for _ in range(100):
        # points = get_indices_sparse(new_image1)
        
        # if points != []:
        #     print( points[0])
            
                # points = points[0]+21212
            # print('asdasda',points)

        observation = np.concatenate(
            [
                # object_rotation,
                # object_position,
                # object_velocity,
                # object_angular_velocity,

                new_image1,
                new_image2,
                new_image3,
                eef,
            ]
        )
        return observation

    def get_achieved_goal(self):
        # new_image1 = []
        # new_image2 = []
        self.obj1, self.obj2, self.obj3 = mask2binary(self, a = "obj")
        self.eef_obj = eef_binary(self, a = "obj")
        self.obj1 = np.array(self.obj1)
        self.obj2 = np.array(self.obj2) 
        self.obj3 = np.array(self.obj3)
        self.eef_obj = np.array(self.eef_obj)
        # px1,px2,px3,mask1,mask2,mask3 = get_view(self)
    
        # rgb1 = Image.fromarray(px1) 
        # rgb1.save('CAMMMEERRAAA11111.png')
        return np.concatenate([self.obj1,self.obj2,self.obj3, self.eef_obj])
        # return np.concatenate([self.obj1,self.obj2, self.obj3])

    def reset(self):
        self.goal = self._sample_goal()
        object_position = self._sample_object()
        self.sim.set_base_pose("goal", self.goal, [0, 0, 0, 1])
        self.sim.set_base_pose("object", object_position, [0, 0, 0, 1])
        self.goal1 = []
        self.goal2 = []
        self.goal3 = []
        self.eef_goal = []
        self.goal1, self.goal2, self.goal3 = mask2binary(self, a = "goal") #picture for the goal
        self.eef_goal = eef_binary(self, a = "goal")
        
    def _sample_goal(self):
        """Randomize goal."""
        # goal = [-0.13, 0.0, self.object_size *2.5]  # z offset for the cube center
        goal = [0.0, 0.0, self.object_size *2]
        noise = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        # if self.np_random.random() < 0.1:
        #     noise[2] = 0.0
        goal += noise
        return goal

    def _sample_object(self):
        """Randomize start position of object."""
        object_position = [0.0,0.0, self.object_size /2]
        noise = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        object_position += noise
        return object_position

    def is_success(self, achieved_goal, desired_goal):
        # goal1 = []
        # goal2 = []
        # obj1 = []
        # obj2 = []
        # goal1 = thresh_callback(cv2.imread("goal1.jpg"))
        # goal2 = thresh_callback(cv2.imread("goal2.jpg"))
        # obj1 = thresh_callback(cv2.imread("obs1.jpg"))
        # obj2 = thresh_callback(cv2.imread("obs2.jpg"))
        # cam1, cam2 = mask2binary(self, a = "obj")

        # goal1 = thresh_callback(self.goal1.copy())
        # goal2 = thresh_callback(self.goal2.copy())
        # obj1 = thresh_callback(cam1)
        # obj2 = thresh_callback(cam2)

        # d1 = distance(goal1,obj1) 
        # d2 = distance(goal2, obj2)
        # d = d1+d2
        # d = 1
        # print('Le reward  d1',d1)
        # print('Le reward  d2',d2)
        # return (d < 3)#.astype(np.float32)
        d = distance(achieved_goal, desired_goal)
        # obj1 = get_indices_sparse(self.obj1)
        # obj2 = get_indices_sparse(self.obj2)
        # obj3 = get_indices_sparse(self.obj3)
        # goal1 = self.rew_goal1.copy()
        # goal2 = self.rew_goal2.copy()
        # goal3 = self.rew_goal3.copy()
        # d1 = 0
        # d2 = 0
        # d3 = 0
        # if obj1 !=[] and goal1 != []:
        #     d1 = abs(goal1[0]-obj1[0])
        # if obj2 !=[] and goal2 != []:
        #     d2 = abs(goal2[0]-obj2[0])
        # if obj3 !=[] and goal3 != []:
        #     d3 = abs(goal3[0]-obj3[0]) 

        # d = np.linalg.norm(d1+d2+d3, axis=-1)
        # img_threshold = sum(self.des_goal) * 0.15
        return (d >= self.rew_thresh).astype(np.float32)
        # return (d < 30).astype(np.float32)

    def compute_reward(self, achieved_goal, desired_goal, info):
        # goal = sum(desired_goal)
        # current = sum(achieved_goal)
        # # threshold = ((goal * 0.7)/np.size(achieved_goal)) * 100
        # threshold = (goal * 0.7)
        d = distance(achieved_goal, desired_goal)
        # print(np.shape(achieved_goal))
        # print('goal = ',goal)
        # print('current = ',current)
        # print('threshold = ',threshold)
        # d = sum(achieved_goal*desired_goal)
        # print(d)
        # if d > threshold :
        #     return 0
        # else: 
        """min 9f 35"""
        #     return -1
        # cam1, cam2 = mask2binary(self, a = "obj")
        # goal1 = thresh_callback(cv2.imread("goal1.jpg"))
        # goal2 = thresh_callback(cv2.imread("goal2.jpg"))
        # obj1 = thresh_callback(cv2.imread("obs1.jpg"))
        # obj2 = thresh_callback(cv2.imread("obs2.jpg"))
        # goal1 = thresh_callback(self.goal1.copy())
        # goal2 = thresh_callback(self.goal2.copy())
        # obj1 = thresh_callback(cam1)
        # obj2 = thresh_callback(cam1)
        
        # d = np.linalg.norm(goal1 - obj1, axis=-1) + np.linalg.norm(goal2 - obj2, axis=-1)
        # d1 = distance(self.goal1, cam1) 
        # d2 = distance(self.goal2, cam2)
        
        # print('Le reward  d1',d1)
        # print('sum achieved goal = ', sum(achieved_goal))
        # print('sum desired goal = ', sum(desired_goal))
        # print('Le reward  d2',(d))
        # return -(d1+d2)
        # return 0
        # if self.reward_type == "sparse":
        #     return -(d > self.distance_threshold).astype(np.float32)
        # else:
        #     return -d
        # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!',d)
        # eef = eef_binary(self, a = "obj")
        # obj1 = get_indices_sparse(self.obj1)
        # obj2 = get_indices_sparse(self.obj2)
        # obj3 = get_indices_sparse(self.obj3)
        # goal1 = self.rew_goal1.copy()
        # goal2 = self.rew_goal2.copy()
        # goal3 = self.rew_goal3.copy()
        # d1 = 0
        # d2 = 0
        # d3 = 0
        # if obj1 !=[] and goal1 != []:
        #     d1 = abs(goal1[0]-obj1[0])
        # if obj2 !=[] and goal2 != []:
        #     d2 = abs(goal2[0]-obj2[0])
        # if obj3 !=[] and goal3 != []:
        #     d3 = abs(goal3[0]-obj3[0]) 
        
        # d = np.linalg.norm(d1+d2+d3, axis=-1)
        
        if sum(self.eef_obj) < 8:
            penalty = -0.5
        else: 
            penalty = 0
        # img_threshold = sum(self.des_goal) * 0.15
        # print('img',img_threshold)
        # print(penalty)
        # print('rew thresh',self.rew_thresh)
        # print('dist ',d)
        return -(d < self.rew_thresh).astype(np.float32) + penalty
        # print(penalty+d)
        # return -((d > 20).astype(np.float32))+ penalty
        # return -d  