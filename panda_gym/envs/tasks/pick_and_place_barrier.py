from email.iterators import body_line_iterator
import numpy as np
from gym import utils
import random

from panda_gym.envs.core import Task
from panda_gym.utils import distance

class PickAndPlaceBarrier(Task):
    def __init__(
        self,
        sim,
        reward_type="sparse",
        distance_threshold=0.05,
        goal_xy_range=0.0,
        goal_z_range=0.0,
        obj_xy_range=0.3,
    ):
        self.sim = sim
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.object_size = 0.04
        # self.object_size = np.random.uniform(0.025,0.05)
        self.goal_range_low = np.array([-goal_xy_range / 2, -goal_xy_range / 2, 0])
        self.goal_range_high = np.array([goal_xy_range / 2, goal_xy_range / 2, goal_z_range])
        self.obj_range_low = np.array([-obj_xy_range / 2, -obj_xy_range / 2, 0])
        self.obj_range_high = np.array([obj_xy_range / 2, obj_xy_range / 2, 0])
        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=[0, 0, 0], distance=0.9, yaw=45, pitch=-30)

    def _create_scene(self):
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_box(
            half_extents = [0.1, 0.01, 0.03],
            mass=0,
            body_name = "barrier",
            position=[0.15, 0.15, 0.03],
            specular_color=[0.0, 0.0, 0.0],
            rgba_color=[0.15, 0.15, 0.15, 1.0],
        )
        self.sim.create_box(
            half_extents = [0.01, 0.1, 0.03],
            mass=0,
            body_name = "barrier",
            position=[0.05, 0.25, 0.03],
            specular_color=[0.0, 0.0, 0.0],
            rgba_color=[0.15, 0.15, 0.15, 1.0],
        )
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        # self.sim.create_cylinder(
        #     body_name="object",
        #     radius = 0.02, 
        #     mass=2,
        #     height = 0.05,
        #     position=[0.0, 0.0, 0 ],
        #     rgba_color=[0.9, 0.1, 0.1, 1],
        #     friction=5,  # increase friction. For some reason, it helps a lot learning
        # )
        # self.sim.create_cylinder(
        #     body_name="target",
        #     radius = 0.02, 
        #     height = 0.05,
        #     mass=0.0,
        #     ghost=True,
        #     position=[0.0, 0.0, 0.05],
        #     rgba_color=[0.9, 0.1, 0.9, 0.3],
        # )
        # self.sim.create_sphere(
        #     body_name="object",
        #     radius = 0.02, 
        #     mass=2,

        #     position=[0.0, 0.0, 0 ],
        #     rgba_color=[0.9, 0.1, 0.1, 1],
        #     friction=5,  # increase friction. For some reason, it helps a lot learning
        # )
        # self.sim.create_sphere(
        #     body_name="target",
        #     radius = 0.02, 
        #     mass=0.0,
        #     ghost=True,
        #     position=[0.0, 0.0, 0.05],
        #     rgba_color=[0.9, 0.1, 0.9, 0.3],
        # )
        self.sim.create_box(
            body_name="object",
            half_extents=[
                self.object_size / 2,
                self.object_size / 2,
                self.object_size / 2,
            ],
            mass=2,
            position=[0.0, 0.0, 0 ],
            rgba_color=[0.9, 0.1, 0.1, 1],
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
            rgba_color=[0.9, 0.1, 0.9, 0.3],
        )

    def get_goal(self):
        return self.goal.copy()

    def get_obs(self):
        # position, rotation of the object
        object_position = np.array(self.sim.get_base_position("object"))
        object_rotation = np.array(self.sim.get_base_rotation("object"))
        object_velocity = np.array(self.sim.get_base_velocity("object"))
        object_angular_velocity = np.array(self.sim.get_base_angular_velocity("object"))
        observation = np.concatenate(
            [
                object_position,
                object_rotation,
                object_velocity,
                object_angular_velocity,
            ]
        )
        return observation

    def get_achieved_goal(self):
        object_position = np.array(self.sim.get_base_position("object"))
        return object_position

    def reset(self):
        # a = ep_counter()
        # b = current_ep() #previous episode because reset is called in the beginning of new episode
        # print(b-1)
        self.goal = self._sample_goal()
        object_position = self._sample_object()
        self.sim.set_base_pose("target", self.goal, [0, 0, 0, 1])
        self.sim.set_base_pose("object", object_position, [0, 0, 0, 1])

    def _sample_goal(self):
        """Randomize goal."""
        goal = [0.15, 0.25, self.object_size /2]  # z offset for the cube center
        noise = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        if self.np_random.random() < 0.3:
            noise[2] = 0.0
        goal += noise
        return goal

    def _sample_object(self):
        """Randomize start position of object."""
        object_position = [0.0, 0.0, self.object_size / 2]
        noise = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        object_position += noise
        return object_position

    def is_success(self, achieved_goal, desired_goal):
        d = distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def compute_reward(self, achieved_goal, desired_goal, info):
    
    #original_sparse    

        d = distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    #initialisation code for modified reward function
    #must enable following code to use the mod function
        
        # distance_object_threshold = 0.05
        # distance_threshold = 0.1      #end-effector threshold with the target

        # d = distance(achieved_goal, desired_goal)
        # current = current_ep()     #count the number of time compute_reward function is called
        # # print(current)
        # ee_pos = self.sim.get_link_position("panda", 11)
        # siz = np.size(achieved_goal)
        
        # ee_pos = np.array(ee_pos-desired_goal)

        # x = ee_pos[0]
        # y = ee_pos[1]
        # z = ee_pos[2]

    #Dense2Sparse using End-effector penalty reward function

        # # 1 episode = 50 timesteps

        # transition_episode = 1750   

        # if current > transition_episode: 
        #     #sparse reward
        #     return -(d > distance_object_threshold).astype(np.float32)

        # else:
        #     # Dense reward
        #     # Prioritized all coordinates
        #     if siz == 3: 
                
        #         if  (x > distance_threshold) or (x < -distance_threshold):
        #             ee_pos=(ee_pos[0]*10, ee_pos[1],ee_pos[2])
                
        #         if  (y > distance_threshold) or (y < -distance_threshold):
        #             ee_pos=(ee_pos[0], ee_pos[1]*5,ee_pos[2])
                
        #         if  (z > distance_threshold) or (z < -distance_threshold):
        #             ee_pos=(ee_pos[0], ee_pos[1],ee_pos[2])

        #         if d > distance_object_threshold:
        #             a = -1-np.linalg.norm(ee_pos, axis=-1)
        #             return a
        #         else:
        #             return -np.linalg.norm(ee_pos, axis=-1)

        #     else: 
        #         return -(d > distance_object_threshold).astype(np.float32)

    # #Dense2Sparse using Dense Reward Function
        # 1 episode = 50 timesteps

        # transition_episode = 500   

        # if current >  transition_episode: 
        #     #sparse reward
        #     return -(d > distance_object_threshold).astype(np.float32)
        # else:
        #     #dense reward
        #     return -d
        
    #Prioritized z-coordinates

        # if siz == 3: 
                      
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

        # if siz == 3: 
            
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
        