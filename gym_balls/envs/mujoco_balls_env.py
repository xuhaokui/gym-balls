import os

from gym.envs.mujoco import mujoco_env


class MujocoBallsEnv(mujoco_env.MujocoEnv):
    """Superclass for all MuJoCo Balls environments.
    """

    def __init__(self, model_path, frame_skip):
        fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        super(MujocoBallsEnv, self).__init__(fullpath, frame_skip)

    def get_body_2d_pos(self, body_name):
        return self.data.get_body_xpos(body_name)[:2]

    def get_body_2d_vel(self, body_name):
        return self.data.get_body_xvelp(body_name)[:2]
