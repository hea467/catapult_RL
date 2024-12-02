import math
import mujoco.viewer
import numpy as np
from gymnasium.spaces import Box
import numpy as np
import matplotlib.pyplot as plt
import random 
import copy
import pickle as pkl

from origami_mujoco_env import Origami_Mujoco_Env

class Catapult_Env(Origami_Mujoco_Env):
    
    def __init__(self, episode_len):
        action_space = {"range": (-0.25, 0.25), "shape" : (2,)}
        xml_path = "assets/catapult_two_goals.xml"
        # Let's start with just the y coordinate

        
        super().__init__(episode_len, xml_path, action_space)
      
        mujoco.mj_forward(self.model, self.data)
    
        
        self.frame_skip = 10 #used to br 100, trying a more sequential approach, more steps
        
        #Keep track of how many steps each of the threads took
        self.local_thread_steps = 0
        self.current_goal = None  
        # self.centers_for_goals ={"box1": [0, 0.45], "box2":[0, 0.32]}

        #getting initial positions
        self.reset()
        
    def _step_mujoco_simulation(self, ctrl, frame_skip):
        """
        Step over the MuJoCo simulation.
        """
        #ONLY do sth if we are releasing, right? 

        self.data.ctrl = ctrl

        mujoco.mj_step(self.model, self.data, nstep=frame_skip)

        # As of MuJoCo 2.0, force-related quantities like cacc are not computed
        # unless there's a force sensor in the model.
        # See https://github.com/openai/gym/issues/1541
        mujoco.mj_rnePostConstraint(self.model, self.data)

    def reset_goal(self): 
        """
        Randomly choose the target goal between the two boxes for this episode.
        """
        # self.current_goal = random.choice(["box1", "box2"])  # randomly choose between two goals
        self.current_goal = [0, np.random.uniform(0.25, 0.6)]
        # print(f"Target goal for this episode: {self.current_goal}")

    def step(self, ctrl):
        """
        Step the simulation n number of frames and applying a control action.
        """
        self.local_thread_steps += 1
        # Check control input is contained in the action space
        self._step_mujoco_simulation(ctrl, self.frame_skip)
        #after stepping the simulation, get observation
        observation = self._get_obs()
        if self.local_thread_steps == self.episode_len:
            #this condition is achieved at the end of every episdoe
            reward = self.reward(observation)
            done = True 
            self.local_thread_steps = 0
        else:
            reward = 0 
            done = False 
        return observation, reward, done, self.current_goal
   
    def is_in_box(self, x, y, target):
        """
        Check if the ball is in the specified box.
        """
        return -0.05 <= x <= 0.05 and target[1] - 0.05 <= y <= target[1] + 0.05
    
    def reward(self, obs):
        x, y = obs[0], obs[1]
        in_target_box = self.is_in_box(x, y, self.current_goal)
        if in_target_box: 
            return 1
        return 0
        
    def _get_obs(self):
        result = []
        # If sarvesh is reading this: I don't care 
        result.append(self.data.body(f'ball').xpos[0])
        result.append(self.data.body(f'ball').xpos[1])
        result.append(self.data.body(f'ball').xpos[2])
        result += self.current_goal # add the centers of the goal
        return np.array(result)
    
    def reset(self):
        self.reset_goal()
        mujoco.mj_resetData(self.model, self.data)
        self.model.body_pos[self.model.body("target_bot").id] = self.current_goal + [0.01]
        return self._get_obs()