
from hashlib import new
from cv2 import threshold
import numpy as np
from gym import utils
from PIL import Image
import os, sys
from io import StringIO
import cv2
from tempfile import TemporaryFile


from panda_gym.envs.core import Task
from panda_gym.utils import distance, get_view, thresh_callback, mask2binary
import pybullet as p
import pybullet_utils.bullet_client as bc
import pybullet_data
# p.connect(p.GUI)
# p.setAdditionalSearchPath(pybullet_data.getDataPath())
class PickAndPlace(Task):
    def __init__(
        self,
        sim,
        reward_type="sparse",
        distance_threshold=0.05,
        goal_xy_range=0.2,
        goal_z_range=0.2,
        obj_xy_range=0.3,
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
                self.goal1, self.goal2 = mask2binary(self, a = "goal")
            else:
                self.des_goal = np.concatenate([self.goal1.copy(),self.goal2.copy()])
                return np.concatenate([self.goal1.copy(),self.goal2.copy()])

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

        # BW_obj1 = np.ravel(BW_obj1)
        # BW_obj2 = np.ravel(BW_obj2)
        # new_image1 = []
        # new_image2 = []
        new_image1, new_image2 = mask2binary(self, a = "obj")
        observation = np.concatenate(
            [
                # object_rotation,
                # object_position,
                # object_velocity,
                # object_angular_velocity,

                new_image1,
                new_image2,
            ]
        )
        return observation

    def get_achieved_goal(self):
        # new_image1 = []
        # new_image2 = []
        new_image1, new_image2 = mask2binary(self, a = "obj")

        return np.concatenate([new_image1,new_image2])

    def reset(self):
        self.goal = self._sample_goal()
        object_position = self._sample_object()
        self.sim.set_base_pose("goal", self.goal, [0, 0, 0, 1])
        self.sim.set_base_pose("object", object_position, [0, 0, 0, 1])
        self.goal1 = []
        self.goal2 = []
        self.goal1, self.goal2 = mask2binary(self, a = "goal") #picture for the goal

    def _sample_goal(self):
        """Randomize goal."""
        goal = [0.1, 0.2, self.object_size *2.5]  # z offset for the cube center
        noise = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        if self.np_random.random() < 0.3:
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
        img_threshold = sum(desired_goal) * 0.15
        return (d >= img_threshold).astype(np.float32)

    def compute_reward(self, achieved_goal, desired_goal, info):
        # goal = sum(desired_goal)
        # current = sum(achieved_goal)
        # # threshold = ((goal * 0.7)/np.size(achieved_goal)) * 100
        # threshold = (goal * 0.7)
        d = distance(achieved_goal, desired_goal)
        # print('goal = ',goal)
        # print('current = ',current)
        # print('threshold = ',threshold)
        # d = sum(achieved_goal*desired_goal)
        # print(d)
        # if d > threshold :
        #     return 0
        # else: 
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

        img_threshold = (sum(self.des_goal) * 0.15)
        # print(img_threshold)
        return -(d < img_threshold).astype(np.float32)
        # return -(d < 2).astype(np.float32)
        # return -((d < self.distance_threshold).astype(np.float32))