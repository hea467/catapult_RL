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
        action_space = {"range": (-0.5, 0.5), "shape" : (1,)}
        xml_path = "assets/catapult.xml"
        # Let's start with just the y coordinate
        goal = -0.5
        
        super().__init__(episode_len, xml_path, action_space, goal)
      
        #getting initial positions of the pantograph
        self.reset()
        mujoco.mj_forward(self.model, self.data)
    
        
        
        self.frame_skip = 10 #used to br 100, trying a more sequential approach, more steps
        
        #Keep track of how many steps each of the threads took
        self.local_thread_steps = 0

        
    def _step_mujoco_simulation(self, release_bool, frame_skip):
        """
        Step over the MuJoCo simulation.
        """
        #ONLY do sth if we are releasing, right? 
        if release_bool: 
            self.data.ctrl = 0.25
            # self.data.ctrl = np.clip(self.data.ctrl, 0, 0.5)
        mujoco.mj_step(self.model, self.data, nstep=frame_skip)

        # As of MuJoCo 2.0, force-related quantities like cacc are not computed
        # unless there's a force sensor in the model.
        # See https://github.com/openai/gym/issues/1541
        mujoco.mj_rnePostConstraint(self.model, self.data)

    def reset_goal(): 
        pass
    
    def step(self, release_bool):
        """
        Step the simulation n number of frames and applying a control action.
        """
        self.local_thread_steps += 1
        # Check control input is contained in the action space
        self._step_mujoco_simulation(release_bool, self.frame_skip)
        #after stepping the simulation, get observation
        observation = self._get_obs()
                
        # If very good reward --> Terminate
        reward, done = self.reward(observation)
        # if timestep is up, also done?
        if self.local_thread_steps == self.episode_len:
            done = True
        if done: 
            self.local_thread_steps = 0
        return observation, reward, done
    
    def reward(self, obs):
        #If it flew further than we r done
        return obs[2], False
    
    # def reward(self, obs, ctrl):

    #     euclid_dist = np.linalg.norm(obs[1] - self.goal)
    #     done = False 
    #     if euclid_dist < 0.02:
    #         done = True
    #     return (100*euclid_dist), done
        
    def _get_obs(self):
        result = []
        # If sarvesh is reading this: I don't care 
        result.append(self.data.body(f'ball').xpos[0])
        result.append(self.data.body(f'ball').xpos[1])
        result.append(self.data.body(f'ball').xpos[2])
        #It works
        result.append(self.goal)
        #print(f"observed loc {[result[60], result[62]]}, actual: {self.data[thread_id].body(f'v21').xpos}")\
        # print(result)
        return np.array(result)