from turtle import pen
import numpy as np
from gym import utils

from panda_gym.envs.core import Task
from panda_gym.utils import distance, comparee
from pyrsistent import b

class PickAndPlaceCluttered(Task):

    def __init__(
        self,
        sim,
        reward_type="sparse",
        distance_threshold=0.05,
        goal_xy_range=0.3,
        goal_z_range=0.2,
        obj_xy_range=0.3,
        
    ):
        self.sim = sim
        self.ep_counter = 0.0
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.object_size = 0.04

        self.previous_object2 = []
        self.previous_object3 = []

        self.goal_range_low = np.array([-goal_xy_range / 2, -goal_xy_range / 2, 0.0])
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
            body_name="object1",
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
            body_name="object2",
            half_extents=[
                self.object_size / 2,
                self.object_size / 2,
                self.object_size / 2,
            ],
            mass=2,
            position=[0.0, 0.0, self.object_size / 2],
            rgba_color=[0.1, 0.9, 0.1, 1],
            friction=5,  # increase friction. For some reason, it helps a lot learning
        )
        self.sim.create_box(
            body_name="object3",
            half_extents=[
                self.object_size / 2,
                self.object_size / 2,
                self.object_size / 2,
            ],
            mass=2,
            position=[0.0, 0.0, self.object_size / 2],
            rgba_color=[0.1, 0.1, 0.9, 1],
            friction=5,  # increase friction. For some reason, it helps a lot learning
        )
        self.sim.create_box(
            body_name="target",
            half_extents=[
                self.object_size / 2,
                self.object_size / 2,
                self.object_size / 2,
            ],
            mass=0.0,
            ghost=True,
            position=[0.0, 0.0, 0.05],
            rgba_color=[1, 1, 0, 0.3],
        )

    def get_goal(self):
        return self.goal.copy()

    def get_obs(self):
        # position, rotation of the object
        
        object_position = np.array(self.sim.get_base_position("object1"))
        object_rotation = np.array(self.sim.get_base_rotation("object1"))
        object_velocity = np.array(self.sim.get_base_velocity("object1"))
        object_angular_velocity = np.array(self.sim.get_base_angular_velocity("object1"))
          
        # object_cam = np.array(self.sim.get_visual_data("object2"))

        object_position2 = np.array(self.sim.get_base_position("object2"))
        object_rotation2 = np.array(self.sim.get_base_rotation("object2"))
        object_velocity2 = np.array(self.sim.get_base_velocity("object2"))
        object_angular_velocity2 = np.array(self.sim.get_base_angular_velocity("object2"))

        object_position3 = np.array(self.sim.get_base_position("object3"))
        object_rotation3 = np.array(self.sim.get_base_rotation("object3"))
        object_velocity3 = np.array(self.sim.get_base_velocity("object3"))
        object_angular_velocity3 = np.array(self.sim.get_base_angular_velocity("object3"))
        
        # rgba1_color=np.array([0.9, 0.1, 0.1, 1])
        # rgba2_color=np.array([0.1, 0.9, 0.1, 1])
        # rgba3_color=np.array([0.1, 0.1, 0.9, 1])
  
        # contact = comparee(self.previous_object2, self.previous_object3, np.array(self.sim.get_base_position("object2")), np.array(self.sim.get_base_position("object3")))
        # if contact<0:   #boolean value for contact with other object
        #     contact = [1]
        # else :
        #     contact = [0]

        # dist = [distance(np.array(self.sim.get_base_position("object1")),np.array(self.goal.copy()))]

        observation = np.concatenate(
            [
                object_position,
                object_rotation,
                object_velocity,
                object_angular_velocity,
                                
                object_position2,
                object_rotation2,
                object_velocity2,
                object_angular_velocity2,
         
                object_position3,
                object_rotation3,
                object_velocity3,
                object_angular_velocity3,

                # rgba1_color,
                # rgba2_color,
                # rgba3_color,

                # contact,
                # dist,
            ]
        )
        return observation

    def get_achieved_goal(self):
        object_position = np.array(self.sim.get_base_position("object1"))
        return object_position

    def reset(self):
        self.ep_counter += 1

        self.goal = self._sample_goal()
        object1_position, object2_position,object3_position = self._sample_object()
        self.sim.set_base_pose("target", self.goal, [0, 0, 0, 1])
        self.sim.set_base_pose("object1", object1_position, [0, 0, 0, 1])
        self.sim.set_base_pose("object2", object2_position, [0, 0, 0, 1])
        self.sim.set_base_pose("object3", object3_position, [0, 0, 0, 1])

        '''initialise overwrite the previous object position '''
        self.previous_object2 = []
        self.previous_object2 = object2_position
        self.previous_object3 = []
        self.previous_object3 = object3_position
        
    def _sample_goal(self):
        """Randomize goal."""
        # goal = [0.0, 0.0, self.object_size /2]
        goal = [0.0, 0.0, self.object_size * 2.5]  # z offset for the cube center
        noise = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        if self.np_random.random() < 0.3:
            noise[2] = 0.0
        goal += noise
        return goal

    def _sample_object(self):
        """Curriculum learning---make sure that the other objects are on the floor instead of the table  """
        if self.ep_counter < 10000:
            a = 1.1
            az = -0.4
        else:
            a = 0.0
            az = 0.0

        if self.ep_counter < 25000:
            b = 1.1
            bz = -0.4
        else:
            b = 0.0
            bz = 0.0
      
        while True:
            object1_position = [0.0, 0.0, self.object_size / 2]

            # object2_position = [a, a, az+self.object_size / 2]    #curriculum learning
            # object3_position = [a, a, bz+self.object_size / 2]    #curriculum learning

            object2_position = [0.0, 0.0, self.object_size / 2]
            object3_position = [0.0, 0.0, self.object_size / 2]
            
            noise1 = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
            noise2 = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
            noise3 = self.np_random.uniform(self.obj_range_low, self.obj_range_high)

            object1_position += noise1
            object2_position += noise2
            object3_position += noise3

            if (
                distance(object1_position, object2_position) > 0.05 and 
                distance(object1_position, object3_position) > 0.05 and
                distance(object2_position, object3_position) > 0.05                 
                ):
               return object1_position , object2_position, object3_position

    def is_success(self, achieved_goal, desired_goal):
        d = distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def compute_reward(self, achieved_goal, desired_goal, info):
        d = distance(achieved_goal, desired_goal)

        current_object2 = np.array(self.sim.get_base_position("object2"))
        current_object3 = np.array(self.sim.get_base_position("object3"))

        penalty = comparee(self.previous_object2, self.previous_object3,  current_object2, current_object3)
       
        ee_pos = np.array(self.sim.get_link_position("panda", 11))
        current_object1 = np.array(self.sim.get_base_position("object1"))
        pen = 0
        if (sum(abs(ee_pos-current_object1))) > 0.04:
            pen = -0.5

        '''overwrites the previous object position with new'''
        self.previous_object2 = []
        self.previous_object2 = current_object2
        self.previous_object3 = []
        self.previous_object3 = current_object3


        # return (-(d > self.distance_threshold).astype(np.float32) + penalty)
        return (-(d > self.distance_threshold).astype(np.float32) + penalty + pen) #sparse
        # return -d + penalty     #dense



'''Alternative reward function
        # link = 11
        # distance_object_threshold = 0.05
        # distance_threshold = 0.1
        # d = distance(achieved_goal, desired_goal)
        # ee_pos = self.sim.get_link_position("panda", link)
        # siz = np.shape(achieved_goal)
        # ee_pos = np.array(ee_pos-desired_goal)

        # x = ee_pos[0]
        # y = ee_pos[1]
        # z = ee_pos[2]
 
        # Prioritized all coordinates

        # if siz == (3,): 
            
        #     if  (x > distance_threshold) or (x < -distance_threshold):
        #         ee_pos=(ee_pos[0]*10, ee_pos[1],ee_pos[2])
            
        #     if  (y > diobject_size / 2]
            # object3_position = [0.0, 0.0, self
        #     if d > distance_object_threshold:
        #         a = -1-np.linalg.norm(ee_pos, axis=-1)
        #         return a
        #     else:
        #         return -np.linalg.norm(ee_pos, axis=-1)

        # else: 
        #     return -(d > self.distance_threshold).astype(np.float32)
        
        #Prioritized z-coordinates

        # if siz == (3,): 
                      
        #     if  (z > distance_threshold) or (z < -distance_threshold):
        #         ee_pos=(ee_pos[0], ee_pos[1],ee_pos[2]*5)

        #     if d > distance_object_threshold:
        #         a = -1-np.linalg.norm(ee_pos, axis=-1)
        #         return a
        #     else: 
        #         return -np.linalg.norm(ee_pos, axis=-1)

        # else: 
        #     return -(d > self.distance_threshold).astype(np.float32)

        # Constant penalties

        # if siz == (3,): 
            
        #     if  (x > distance_threshold) or (x < -distance_threshold):
        #         ee_pos=(ee_pos[0]*0+5, ee_pos[1],ee_pos[2])
            
        #     if  (y > distance_threshold) or (y < -distance_threshold):
        #         ee_pos=(ee_pos[0], ee_pos[1]*0+2.5,ee_pos[2])
            
        #     if  (z > distance_threshold) or (z < -distance_threshold):
        #         ee_pos=(ee_pos[0], ee_pos[1],ee_pos[2]*0+1)

        #     if d > distance_object_threshold:
        #         a = -1-np.linalg.norm(ee_pos, axis=-1)
        #         return a
        #     else:
        #         return -np.linalg.norm(ee_pos, axis=-1)

        # else: 
        #     return -(d > self.distance_threshold).astype(np.float32)
'''