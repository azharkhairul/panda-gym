import numpy as np
from gym import utils
from turtle import pen
from panda_gym.envs.core import Task
from panda_gym.utils import distance
from pyrsistent import b

class Stack(Task):
    def __init__(
        self,
        sim,
        reward_type="sparse",
        distance_threshold=0.1,
        goal_xy_range=0.0,
        goal_z_range=0.2,
        obj_xy_range=0.3,
    ):
        self.sim = sim

        self.ep_counter = 0
        self.obj1 = 0
        self.obj2 = 0
        self.obj1total = 0
        self.obj2total = 0

        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.object_size = 0.04
        self.goal_range_low = np.array([-goal_xy_range / 2, -goal_xy_range / 2, 0])
        self.goal_range_high = np.array([goal_xy_range / 2, goal_xy_range / 2, 0])
        self.goal_range_high_train = np.array([goal_xy_range / 2, goal_xy_range / 2, goal_z_range])
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
            body_name="target1",
            half_extents=[
                self.object_size / 2,
                self.object_size / 2,
                self.object_size / 2,
            ],
            mass=0.0,
            ghost=True,
            position=[0.0, 0.0, 0.05],
            rgba_color=[0.9, 0.1, 0.1, 0.3],
        )
        self.sim.create_box(
            body_name="object2",
            half_extents=[
                self.object_size / 2,
                self.object_size / 2,
                self.object_size / 2,
            ],
            mass=2,
            position=[0.5, 0.0, self.object_size / 2],
            rgba_color=[0.1, 0.9, 0.1, 1],
            friction=5,  # increase friction. For some reason, it helps a lot learning
        )
        self.sim.create_box(
            body_name="target2",
            half_extents=[
                self.object_size / 2,
                self.object_size / 2,
                self.object_size / 2,
            ],
            mass=0.0,
            ghost=True,
            position=[0.5, 0.0, 0.05],
            rgba_color=[0.1, 0.9, 0.1, 0.3],
        )
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
    def get_goal(self):
        return self.goal.copy()

    def get_obs(self):
        # position, rotation of the object
        object1_position = np.array(self.sim.get_base_position("object1"))
        object1_rotation = np.array(self.sim.get_base_rotation("object1"))
        object1_velocity = np.array(self.sim.get_base_velocity("object1"))
        object1_angular_velocity = np.array(self.sim.get_base_angular_velocity("object1"))
        object2_position = np.array(self.sim.get_base_position("object2"))
        object2_rotation = np.array(self.sim.get_base_rotation("object2"))
        object2_velocity = np.array(self.sim.get_base_velocity("object2"))
        object2_angular_velocity = np.array(self.sim.get_base_angular_velocity("object2"))
        observation = np.concatenate(
            [
                object1_position,
                object1_rotation,
                object1_velocity,
                object1_angular_velocity,
                object2_position,
                object2_rotation,
                object2_velocity,
                object2_angular_velocity,
            ]
        )
        return observation

    def get_achieved_goal(self):
        object1_position = np.array(self.sim.get_base_position("object1"))
        object2_position = np.array(self.sim.get_base_position("object2"))
        achieved_goal = np.concatenate((object1_position, object2_position))
        return achieved_goal

    def reset(self):
        self.ep_counter += 1
        if self.obj1 > 0:
            print('Object 1 was in place')
            self.obj1total += 1
        if self.obj2 > 0:
            print('Object 2 was in place')
            self.obj2total += 1
        print('Total obj1 placed correctly = ',self.obj1total)
        print('Total obj2 placed correctly = ',self.obj2total)
        self.obj1 = 0
        self.obj2 = 0

        self.goal = self._sample_goal()
        object1_position, object2_position = self._sample_objects()
        self.sim.set_base_pose("target1", self.goal[:3], [0, 0, 0, 1])
        self.sim.set_base_pose("target2", self.goal[3:], [0, 0, 0, 1])
        self.sim.set_base_pose("object1", object1_position, [0, 0, 0, 1])
        self.sim.set_base_pose("object2", object2_position, [0, 0, 0, 1])

    def _sample_goal(self):
        # if self.ep_counter > 15000:
        #     goal1 = [0.0, 0.0, self.object_size / 2]  # z offset for the cube center
        #     goal2 = [0.0, 0.0, 3 * self.object_size / 2]  # z offset for the cube center
        #     noise = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        #     goal1 += noise
        #     goal2 += noise
        #     return np.concatenate((goal1, goal2))
        # else:
            # while True:
            #     goal1 = [0.0, 0.0, self.object_size / 2]  # z offset for the cube center
            #     goal2 = [0.0, 0.0, self.object_size / 2]  # z offset for the cube center
            #     noise1 = self.np_random.uniform(self.goal_range_low, self.goal_range_high_train)
            #     noise2 = self.np_random.uniform(self.goal_range_low, self.goal_range_high_train)
            #     goal1 += noise1
            #     goal2 += noise2
            #     if distance(goal1, goal2) > 0.05:
            #         self.goal2 = goal2
            #         return np.concatenate((goal1, goal2))

        while True:      
            goal1 = [0.15, 0.25, self.object_size /2]   # z offset for the cube center
            goal2 = [0.15, 0.25, 3 * self.object_size / 2]  # z offset for the cube center
            noise = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
            goal1 += noise
            goal2 += noise
            return np.concatenate((goal1, goal2))  
        # while True:
        #     if self.ep_counter <= 10000:
        #         a = 0.0
        #         az = 0.0 + 0.04
        #         noise1 = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        #         noise2 = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        #         b = 0.0
        #         bz = 0.0

        #     if self.ep_counter > 10000 and self.ep_counter <= 20000:
        #         a = 0.0
        #         az = 0.0
        #         b = 0.0
        #         bz = 0.0 + 0.04
        #         noise1 = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        #         noise2 = self.np_random.uniform(self.goal_range_low, self.goal_range_high)

        #     if self.ep_counter > 20000:
        #         a = 0.0
        #         az = 0.0
        #         b = 0.0
        #         bz = 0.04
        #         noise1 = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        #         noise2 = noise1

        #     goal1 = [a, a, az + self.object_size / 2]  # z offset for the cube center
        #     goal2 = [b, b, bz + self.object_size / 2]  # z offset for the cube center        
        #     goal1 += noise1
        #     goal2 += noise2
        #     if distance(goal1, goal2) > 0.05:
        #         return np.concatenate((goal1, goal2))

    def _sample_objects(self):
        # while True:  # make sure that cubes are distant enough
        #     object1_position = [0.0, 0.0, self.object_size / 2]
        #     object2_position = [0.0, 0.0, self.object_size / 2]
        #     noise1 = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        #     noise2 = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        #     object1_position += noise1
        #     object2_position += noise2
        #     if distance(object1_position, object2_position) > 0.05:
        #         return object1_position, object2_position
        # while True:
        #     if self.ep_counter <= 10000:
        #         a = 0.0
        #         az = 0.0
        #         noise1 = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        #         noise2 = self.goal[3:]
        #         b = 0.0
        #         bz = 0.0

        #     if self.ep_counter > 10000 and self.ep_counter <= 20000:
        #         a = 0.0
        #         az = 0.0
        #         b = 0.0
        #         bz = 0.0
        #         noise1 = self.goal[:3]
        #         noise2 = self.np_random.uniform(self.obj_range_low, self.obj_range_high)

        #     if self.ep_counter > 20000:
        #         a = 0.0
        #         az = 0.0
        #         b = 0.0
        #         bz = 0.0
        #         noise1 = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        #         noise2 = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
                
        #     object1_position = [a, a, az + self.object_size / 2]
        #     object2_position = [b, b, bz + self.object_size / 2]
        #     object1_position += noise1
        #     object2_position += noise2
        #     if distance(object1_position, object2_position) > 0.05:
        #         return object1_position, object2_position
        # while True:
        #     object1_position = [0.0, 0.0, self.object_size / 2]
        #     object2_position = [0.0, 0.0, self.object_size / 2]
        #     if self.ep_counter <= 100:
        #         noise1 = self.goal[:3]
        #         noise2 = self.goal[3:]
        #         object1_position += noise1
        #         object2_position += noise2
        #         return object1_position, object2_position
        #     else:
        #         noise1 = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        #         noise2 = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        #         object1_position += noise1
        #         object2_position += noise2
        #         if distance(object1_position, object2_position) > 0.05:
        #             return object1_position, object2_position

        while True:
            object1_position = [0.0, 0.0, self.object_size / 2]
            object2_position = [0.0, 0.0, self.object_size / 2]
            noise1 = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
            noise2 = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
            object1_position += noise1
            object2_position += noise2
            if ((object1_position[1]<0.03 or object1_position[2]<0.1) and (object2_position[1]<0.03 or object2_position[2]<0.1)):
                if distance(object1_position, object2_position) > 0.05:
                    return object1_position, object2_position

        # if self.ep_counter > 20000:
        #     while True:
        #         object1_position = [0.0, 0.0, self.object_size / 2]
        #         object2_position = [0.0, 0.0, self.object_size / 2]
        #         noise1 = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        #         noise2 = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        #         object1_position += noise1
        #         object2_position += noise2
        #         if distance(object1_position, object2_position) > 0.05:
        #             return object1_position, object2_position
            
        # if self.ep_counter > 10000 and self.ep_counter < 20000:
        #     object1_position = [1.1, 1.1, -0.4 + (self.object_size / 2)]
        #     object2_position = [0.0, 0.0, self.object_size / 2]
        #     noise2 = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        #     object2_position += noise2
        #     return object1_position, object2_position
        # if self.ep_counter < 10000:
        #     object2_position = [1.1, 1.1, -0.4 + (self.object_size / 2)]
        #     object1_position = [0.0, 0.0, self.object_size / 2]
        #     noise1 = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        #     object1_position += noise1
        #     return object1_position, object2_position

    def is_success(self, achieved_goal, desired_goal):
        # must be vectorized !!
        d = distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def compute_reward(self, achieved_goal, desired_goal, info):
        d = distance(achieved_goal, desired_goal)
        
        pen1 = 0
        pen2 = 0
        obj1 = 0
        obj2 = 0

        ee_pos = np.array(self.sim.get_link_position("panda", 11))
        current_object1 = np.array(self.sim.get_base_position("object1"))
        current_object2 = np.array(self.sim.get_base_position("object2"))
        
        dis1 =  distance(ee_pos, current_object1)
        dis2 =  distance(ee_pos, current_object2)

        obj1 = distance(self.goal[:3], current_object1)
        

        if obj1 > 0.04:
            # obj1 = 1
            if (sum(abs(ee_pos-current_object1))) > 0.05: #penalty to encourage contact with the target object
                pen1 = dis1/2
        else:
            self.obj1 += 1
            obj2 = distance(self.goal[3:], current_object2)
            if obj2 < 0.04:
                self.obj2 += 1
            else:
                if (sum(abs(ee_pos-current_object2))) > 0.05: #penalty to encourage contact with the target object
                    pen2 = dis2/2
        """
        The target of this reward is to make the agent learn the task in a curriculum manner:
        - It will be solving the object 1 first and then proceed with object 2
        - As long as the object 1 is not successfully in place, the agent will be ignoring the object 2
            - meaning that no reward or penalty will be given based on the object 2 as long as object 1 is not in place
            - the penalty reward for distance between the end-effector and the object will utilized only when the objects is not in place
            - the obj1 penalty will continually given to the agent to improve the placement of the object and to avoid the object from being pushed as the agent solve the object2
        """

        return -(obj1 + obj2 + pen1 + pen2)

        # if self.reward_type == "sparse":
        #     return -(d > self.distance_threshold).astype(np.float32)
        # else:
        #     return -d
