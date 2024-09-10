import numpy as np 
import mujoco
from gymnasium.spaces import Box
import matplotlib.pyplot as plt


class Origami_Mujoco_Env():
    def __init__(self, episode_len, xml_path, action_space: dict, goal):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.step_number = 0
        self.episode_len = episode_len
        # print(self.episode_len)
        self.render_mode = "human"
        if "range" not in action_space.keys() or "shape" not in action_space.keys():
            raise KeyError("Action space dictionary must provide range : (low, high) and shape.")
        #bc action space is should be smaller since kp value. 
        self.action_space = Box(low=action_space["range"][0], high=action_space["range"][1], shape=action_space["shape"], dtype=np.float64)
        self.goal = goal
        # print("Init qpos: -----", self.init_qpos)
        
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        return self._get_obs()
    
    def _get_obs(self) :
        pass



    
        
    