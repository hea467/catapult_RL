import time
import mujoco.viewer
import sac
import torch
import numpy as np
import os
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

model = mujoco.MjModel.from_xml_path("assets/cubes_arm_try.xml")
data = mujoco.MjData(model)
max_ep_len = 200

hp_dict = {
            "exp_name"          : "Linear_End_Effector_Shift",
            "data_dir"          : "./data/rl_data",
            "tau"               : 0.005,
            "gamma"             : 0.99,
            "q_lr"              : 3e-3,
            "pi_lr"             : 3e-4,
            "eta_min"           : 1e-5,
            "alpha"             : 0,
            "replay_size"       : 50000000,
            'seed'              : 69420,
            'optim'             : 'adam',
            "batch_size"        : 64,
            "exploration_cutoff": 512,
            "infer_every"       : 4000,
            "inference_length"  : 10,
            # "dev_sim"           : torch.device(f"cuda:{args.dev_sim}"),
            "dev_rl"            : torch.device("cuda:0"),
            # "dev_rl"            : torch.device("cpu"),
            "model_dim"         : 256,
            "num_heads"         : 8,
            "dim_ff"            : 128,
            "dropout"           : 0,
            "max_grad_norm"     : 1.0,
            "dont_log"          : True,
        }

logger_kwargs = {}
single_agent_env_dict = {'action_space': {'low': -0.5/200, 'high': 0.5/200, 'dim': 4},
                    'observation_space': {'dim': 65},}

sac_agent = sac.SAC(single_agent_env_dict, hp_dict, logger_kwargs, ma=False, train_or_test="test")
sac_agent.load_saved_policy("SAC_agent_saved/model.pt")
visualize = Visualize(model, data)
# with mujoco.viewer.launch_passive(model, data) as viewer:
start = time.time()

goal = [0.25, 0.95] 

def view_episode(goal):
    for _ in range(max_ep_len):
        step_start = time.time()
        
        # mujoco.mj_step(model, data)
        start = np.array(data.body('v21').xpos)
        xz_axis_pos = np.array([start[0], start[2]] + goal)
        next_action = sac_agent.get_actions(xz_axis_pos, deterministic=True)
        data.ctrl = next_action
        print("position", data.body('v21').xpos[0], data.body('v21').xpos[2])
        print(next_action)
        mujoco.mj_step(model, data)
        # mujoco.mj_resetData(model, data)
        # start = np.array(data.body('v11').xpos)
        #viewer.sync()
        visualize.render()
        # time_until_next_step = model.opt.timestep - (time.time() - step_start)
        # if time_until_next_step > 0:
        #     time.sleep(time_until_next_step)
    
    mujoco.mj_resetData(model, data)
        
db_file = open('runtime_data/position.pkl', 'rb')
episodes_stored = pickle.load(db_file)
# print(episodes_stored)
# print(episodes_stored)
# print(type(episodes_stored[0][0]))
# print(len(episodes_stored))
# while True:
#     for episode in episodes_stored:
#         data.ctrl = episode[0].ctrl
#         # print(type(data))
#         mujoco.mj_step(model, data)
#         visualize.render()

i = 0 
global_step_c = 0

# while global_step_c < len(episodes_stored):
#     while i < 200: 
#         step = episodes_stored[global_step_c] 
#         data.ctrl = step[0].ctrl
#         print(data.ctrl)
#         mujoco.mj_step(model, data)
#         visualize.render()
#         i += 1
#         global_step_c += 1
#     i = 0
#     mujoco.mj_resetData(model, data)
for i in range(1, 22): 
    print(f'v{i}', data.body(f'v{i}').xpos)


# for episode in episodes_stored:
#     for step in episode: 
#         data.ctrl = step.ctrl
#         # print(data.ctrl)
#         mujoco.mj_step(model, data)
#         visualize.render()
#     time.sleep(1)
#     mujoco.mj_resetData(model, data)

# print("done!!")

    # mujoco.mj_resetData(model, data)
    # print(episodes_stored)
        
# call_a_cb_fn(this_is_a_cb_fn):
#     self.fn_var = this_is_a_fn
    
#     def somedmnfgsipdmg\
#         if keypress == 'q':
#             self.fn_var()