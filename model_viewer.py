import time
import mujoco.viewer
import torch
import numpy as np
import os
import glfw
import random
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


def load_or_create_list(filename):
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return []
    

model = mujoco.MjModel.from_xml_path("assets/catapult_updated.xml")
data = mujoco.MjData(model)
pickle_file_path = "saved_episodes/log.pkl"
visualize = Visualize(model, data)
frame_skip = 10
max_ep_len = 500

timesteps_of_release = load_or_create_list(pickle_file_path)[24:]

# def view_episode():
#     # start = time.time()
#     # step_start = time.time()
#     for release_time in timesteps_of_release:
#         for i in range(max_ep_len):
#             mujoco.mj_step(model, data, nstep=frame_skip)
#             visualize.render()
#             if i > release_time: 
#                 data.ctrl[0] = 0.25
#             # time_until_next_step = model.opt.timestep - (time.time() - step_start)
#             # if     time.sleep(time_until_next_step)
#             visualize.render()
#         mujoco.mj_resetData(model, data)

def view_episode():
    # start = time.time()
    # step_start = time.time()
    # time.sleep(2)
    for i in range(5):
        y_cord = np.random.uniform(0.3, 0.5)
        for i in range(max_ep_len):
            mujoco.mj_step(model, data, nstep=frame_skip)
            model.body_pos[model.body("target_bot").id] = [0, y_cord, 0.01]
            visualize.render()
        mujoco.mj_resetData(model, data)

view_episode()