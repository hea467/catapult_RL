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

class Cube_Block_Env(Origami_Mujoco_Env):
    
    def __init__(self, episode_len):
        action_space = {"range": (-0.25, 0.25), "shape" : (4,)}
        xml_path = "assets/walker.xml"
        goal = [0, 0.25] 
        
        super().__init__(episode_len, xml_path, action_space, goal)
        # Origami_Mujoco_Env.__init__(self, episode_len, num_thread, xml_path, action_space, goal)
        
        #getting initial positions of the pantograph
        self.reset()
        mujoco.mj_forward(self.model, self.data)
    
        self.stored_all_obs = []
        self.thread_zero_completion = 0
        self.current_obs = []
        
        
        self.frame_skip = 10 #used to br 100, trying a more sequential approach, more steps
        
        #Keep track of how many steps each of the threads took
        self.local_thread_steps = 0

        #for storing output graphs
        self.stored_trajectory = {}
        self.stored_trajectory_num = 0
        self.global_step = 0
        
        
        
        
    def _step_mujoco_simulation(self, ctrl, frame_skip):
        """
        Step over the MuJoCo simulation.
        """
        #the ctrl is for x pos and y pos actuators of the input.
        # print(ctrl)
        self.data.ctrl = ctrl
        self.data.ctrl = np.clip(self.data.ctrl, 0, 0.5)
        mujoco.mj_step(self.model, self.data, nstep=frame_skip)

        # As of MuJoCo 2.0, force-related quantities like cacc are not computed
        # unless there's a force sensor in the model.
        # See https://github.com/openai/gym/issues/1541
        mujoco.mj_rnePostConstraint(self.model, self.data)

    def step(self, action):
        """
        Step the simulation n number of frames and applying a control action.
        """
        # self.global_step += 1
        self.local_thread_steps += 1
        # Check control input is contained in the action space
        if np.array(action).shape != (self.model.nu,):
            raise ValueError(
                f"Action dimension mismatch. Expected {(self.model.nu,)}, found {np.array(action).shape}"
            )
        self._step_mujoco_simulation(action, self.frame_skip)
        #after stepping the simulation, get observation
        observation = self._get_obs()
                
        # If very good reward --> Terminate
        reward, done = self.reward(observation, action, )
        # if timestep is up, also done?
        if self.local_thread_steps == self.episode_len:
            done = True
        if done: 
            # if episode is finished, we increment global step
            self.global_step += self.local_thread_steps
            self.local_thread_steps = 0
            #track how many episode thread 0 has ran, for data storing purp
            #reset the goal!
            goal_x_cor = random.uniform(0.2, 0.3)
            goal_z_cor = random.uniform(0.89, 0.99)
            self.goal = [goal_x_cor, goal_z_cor]
        # if (self.thread_zero_completion % 10 == 0) and (thread_id==0): 
        #     self.current_obs.append(copy.copy(self.data[0]))
        #     if done:
        #         self.stored_all_obs.append(self.current_obs)
        #         db_file = open("runtime_data/position.pkl",'wb' )
        #         pkl.dump(self.stored_all_obs, db_file)
        #         db_file.close
        #         # print(self.stored_all_obs)
        #         self.current_obs = []
            
        return observation, reward, done
    
    def reward(self, obs, ctrl):
        end_effector_pos = np.array([obs[3*20], obs[3*20 + 2]])
        euclid_dist = np.linalg.norm(end_effector_pos[0:2] - self.goal)
        rew = -100*(euclid_dist + (self.local_thread_steps)/5000)
        pos_ctrl_motivation = 0
        for c in ctrl: 
            if c < 0.05:
                pos_ctrl_motivation -= 1
        done = False 
        if euclid_dist < 0.02:
            done = True
        return (rew + pos_ctrl_motivation), done
        
    def _get_obs(self):
        # obs = np.concatenate((np.array(self.data[thread_id].body('v11').xpos)[3], 
        #                       np.array([])), axis=0)
        # print(obs[0:2])
        # obs = [self.data[thread_id].body('v21').xpos[0], self.data[thread_id].body('v21').xpos[2]]
        obs = []
        for i in range(1, 22):
            obs.append(self.data.body(f'v{i}').xpos[0])
            obs.append(self.data.body(f'v{i}').xpos[1])
            obs.append(self.data.body(f'v{i}').xpos[2])
        goal = list(self.goal)
        result = obs + goal
        #print(f"observed loc {[result[60], result[62]]}, actual: {self.data[thread_id].body(f'v21').xpos}")
        return np.array(result)