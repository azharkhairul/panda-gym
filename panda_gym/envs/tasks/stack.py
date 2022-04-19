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
        goal_xy_range=0.3,
        goal_z_range=0.2,
        obj_xy_range=0.3,
    ):
        self.sim = sim

        self.ep_counter = 0

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
        #     while True:
        #         goal1 = [0.0, 0.0, self.object_size / 2]  # z offset for the cube center
        #         goal2 = [0.0, 0.0, self.object_size / 2]  # z offset for the cube center
        #         noise1 = self.np_random.uniform(self.goal_range_low, self.goal_range_high_train)
        #         noise2 = self.np_random.uniform(self.goal_range_low, self.goal_range_high_train)
        #         goal1 += noise1
        #         goal2 += noise2
        #         if distance(goal1, goal2) > 0.05:
        #             self.goal2 = goal2
        #             return np.concatenate((goal1, goal2))
        while True:
            if self.ep_counter <= 10000:
                a = 0.0
                az = 0.0 + 0.04
                noise1 = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
                noise2 = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
                b = 0.0
                bz = 0.0

            if self.ep_counter > 10000 and self.ep_counter <= 20000:
                a = 0.0
                az = 0.0
                b = 0.0
                bz = 0.0 + 0.04
                noise1 = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
                noise2 = self.np_random.uniform(self.goal_range_low, self.goal_range_high)

            if self.ep_counter > 20000:
                a = 0.0
                az = 0.0
                b = 0.0
                bz = 0.04
                noise1 = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
                noise2 = noise1

            goal1 = [a, a, az + self.object_size / 2]  # z offset for the cube center
            goal2 = [b, b, bz + self.object_size / 2]  # z offset for the cube center        
            goal1 += noise1
            goal2 += noise2
            if distance(goal1, goal2) > 0.05:
                return np.concatenate((goal1, goal2))

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
        while True:
            if self.ep_counter <= 10000:
                a = 0.0
                az = 0.0
                noise1 = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
                noise2 = self.goal[3:]
                b = 0.0
                bz = 0.0

            if self.ep_counter > 10000 and self.ep_counter <= 20000:
                a = 0.0
                az = 0.0
                b = 0.0
                bz = 0.0
                noise1 = self.goal[:3]
                noise2 = self.np_random.uniform(self.obj_range_low, self.obj_range_high)

            if self.ep_counter > 20000:
                a = 0.0
                az = 0.0
                b = 0.0
                bz = 0.0
                noise1 = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
                noise2 = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
                
            object1_position = [a, a, az + self.object_size / 2]
            object2_position = [b, b, bz + self.object_size / 2]
            object1_position += noise1
            object2_position += noise2
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
        if self.reward_type == "sparse":
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d
