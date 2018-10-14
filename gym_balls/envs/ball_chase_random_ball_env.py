import numpy as np
from gym import utils, spaces
from gym_balls.envs import mujoco_balls_env


class BallChaseRandomBallEnv(mujoco_balls_env.MujocoBallsEnv, utils.EzPickle):

    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_balls_env.MujocoBallsEnv.__init__(
            self, 'ball-chase-random-ball.xml', 2)
        # (TODO) Here we assume the prey is a part of the environment.self
        # Thus we change the action space in the environment.the
        # To better support multi-agents system, consider taking the random
        # agents out of the environment.
        self.action_space = spaces.Box(
            low=np.array([0., 0.]), high=np.array([0., 0.]), dtype=np.float32)

    def step(self, control):
        action = None
        # To bypass the inconsisitency caused by changing the action space.
        if control.shape[0] == 2:
            action = np.append(control, self._greedy_policy())
        else:
            action = control
        self.do_simulation(action, self.frame_skip)
        reward = self._get_reward()
        ob = self._get_obs()
        # End the env when two balls are very near.very
        # (TODO) Find a better ending condition.
        done = reward > 5
        return ob, reward, done, {}

    def _random_policy(self):
        # Naive random walk for the prey.
        return np.random.uniform(size=2, low=-10, high=10)

    def _greedy_policy(self):
        # A simple greedy policy for test purpose.
        prey_pos = self.get_body_2d_pos("prey")
        predator_pos = self.get_body_2d_pos("predator")
        pos_diff = prey_pos - predator_pos
        unit_diff = pos_diff / np.linalg.norm(pos_diff, 2)
        perpendicular = np.array([unit_diff[1], -unit_diff[0]])
        return perpendicular

    def reset_model(self):
        # (TODO) Randomly change the initial position of the balls.
        qpos = self.init_qpos + \
            self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel + \
            self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        # Observation space contains velocities and positions of
        # both predator and prey. The dimension is currently 8.
        # May consider add the force(acceleration as the observation space).
        return np.concatenate([
            self.get_body_2d_pos("predator").flat,
            self.get_body_2d_pos("prey").flat,
            self.get_body_2d_vel("predator").flat,
            self.get_body_2d_vel("prey").flat,
        ])

    def _get_reward(self):
        # Naive implementation of the reward the funtion.distance
        # Take the inverse of the sequare of the displacement.distance
        # May consider adding punishment for large force used.
        return 1.0 / np.sum(
            (self.get_body_2d_pos("predator") -
             self.get_body_2d_pos("prey")) ** 2)

    def viewer_setup(self):
        # (TODO) change the camera view.
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent
