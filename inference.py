import time
import mujoco.viewer
import torch
import numpy as np
import os
import sac
import glfw
from threading import Lock

import pickle

# os.path.abspath("assets/ball_balance.xml")
class Visualize:
    def __init__(self, model, data, lookat=np.array((0.3, 0, 0.4)), distance=1, elevation=-20, azimuth=120):
        self.model, self.data = model, data
        self.gui_lock = Lock()
        self.setup_gui(lookat=np.array((0.3, 0, 0.4)), distance=2, elevation=-20, azimuth=120)
    
    def setup_gui(self,lookat=np.array((0.3, 0, 0.4)), distance=2, elevation=-20, azimuth=120):
        width, height = 1920, 1080

        glfw.init()
        glfw.window_hint(glfw.VISIBLE, 1)
        self.window = glfw.create_window(width, height, "window", None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        framebuffer_width, framebuffer_height = glfw.get_framebuffer_size(self.window)

        self.opt = mujoco.MjvOption()
        self.cam = mujoco.MjvCamera()
        self.cam.lookat = lookat
        self.cam.distance = distance
        self.cam.elevation = elevation
        self.cam.azimuth = azimuth
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.pert = mujoco.MjvPerturb()
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        self.viewport = mujoco.MjrRect(0, 0, framebuffer_width, framebuffer_height)
        
    def render(self):
        with self.gui_lock:
            mujoco.mjv_updateScene(self.model, self.data, self.opt, self.pert, self.cam, mujoco.mjtCatBit.mjCAT_ALL.value, self.scene)
            mujoco.mjr_render(self.viewport, self.scene, self.context)
            glfw.swap_buffers(self.window)
        glfw.poll_events()


hp_dict = {
    "exp_name": "Linear_End_Effector_Shift",
    "data_dir": "./data/rl_data",
    "tau": 0.005,
    "gamma": 0.99,
    "q_lr": 3e-3,
    "pi_lr": 1e-4,
    "eta_min": 1e-5,
    "alpha": 0.1,  # Changed from 0.2 to a slightly lower value
    "replay_size": 1000000,
    'seed': 69420,
    'optim': 'adam',
    "batch_size": 128,
    "exploration_cutoff": 5000,
    "infer_every": 4000,
    "inference_length": 10,
    "dev_rl": 'cuda' if torch.cuda.is_available() else 'cpu',
    "model_dim": 64,
    "num_layers": 2,
    "dropout": 0,
    "max_grad_norm": 1.0,
    "dont_log": True,
    "exp_num": 2
}
single_agent_env_dict = {'action_space': {'low': -0.5, 'high': 0.5, 'dim': 1},
                    'observation_space': {'dim': 4},}
logger_kwargs = {}

def load_or_create_list(filename):
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return []
    

model = mujoco.MjModel.from_xml_path("assets/catapult.xml")
data = mujoco.MjData(model)
visualize = Visualize(model, data)
frame_skip = 10
max_ep_len = 500



def convert_act_to_timestep(action, max_ep_length): 
    return round((action +0.5) * max_ep_length)


def convert_timestep_to_act(tp, max_ep_length): 
    return ((tp / 1000) - 0.5)

def inference_height():
    sac_agent = sac.SAC(single_agent_env_dict, hp_dict, logger_kwargs, ma=False, train_or_test="test")
    sac_agent.load_saved_policy("SAC_agent_saved/model_height_exp.pt")
    start = [data.body('ball').xpos[0], data.body('ball').xpos[1], data.body('ball').xpos[2], 0.5]
    action = sac_agent.get_actions(start, deterministic=True)
    release_time = convert_act_to_timestep(action[0], 100)
    for i in range(max_ep_len):
        mujoco.mj_step(model, data, nstep=frame_skip)
        visualize.render()
        if i == release_time: 
            data.ctrl[0] = 0.25
        # time_until_next_step = model.opt.timestep - (time.time() - step_start)
        # if     time.sleep(time_until_next_step)
        visualize.render()
    mujoco.mj_resetData(model, data)

def inference_dist():
    sac_agent = sac.SAC(single_agent_env_dict, hp_dict, logger_kwargs, ma=False, train_or_test="test")
    sac_agent.load_saved_policy("SAC_agent_saved/model_dist_exp.pt")
    start = [data.body('ball').xpos[0], data.body('ball').xpos[1], data.body('ball').xpos[2], 0.5]
    action = sac_agent.get_actions(start, deterministic=True)
    release_time = convert_act_to_timestep(action[0], 500)
    for i in range(max_ep_len):
        mujoco.mj_step(model, data, nstep=frame_skip)
        visualize.render()
        if i == release_time: 
            data.ctrl[0] = 0.25
        # time_until_next_step = model.opt.timestep - (time.time() - step_start)
        # if     time.sleep(time_until_next_step)
        visualize.render()
    mujoco.mj_resetData(model, data)

inference_height()