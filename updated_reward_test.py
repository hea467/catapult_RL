import time
import mujoco.viewer
import math
import numpy as np
import os
import glfw
from threading import Lock
import matplotlib.pyplot as plt
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
visualize = Visualize(model, data)
frame_skip = 10
max_ep_len = 500


# def run_episode(ctrl_applied):
#     mujoco.mj_resetData(model, data)
#     ball_y_pos_at_end = 0
#     for i in range(max_ep_len):
#         mujoco.mj_step(model, data, nstep=frame_skip)
#         visualize.render()
#         if i > 1: 
#             data.ctrl[0] = ctrl_applied
#         if data.body(f'ball').xpos[2] <= 0.002 :
#             ball_y_pos_at_end = data.body(f'ball').xpos[1]
#             break 

#     return [ ctrl_applied, ball_y_pos_at_end]

        # visualize.render(

def is_in_box(x, y):
    flag = False 
    if x >= -0.05 and x <= 0.05:
        if y >= 0.4 and y <= 0.5:
            return True
        else:
            return False
    return False
    

def run_episode(ctrl_applied):
    mujoco.mj_resetData(model, data)
    for i in range(max_ep_len):
        mujoco.mj_step(model, data, nstep=frame_skip)
        visualize.render()
        if i > 1: 
            data.ctrl[0] = ctrl_applied
        if data.body(f'ball').xpos[2] <= 0.03 :
            x = data.body(f'ball').xpos[0]
            y = data.body(f'ball').xpos[1]
            in_box = is_in_box(x, y)
            if in_box: 
                return [ctrl_applied, 1]
            else: 
                return [ctrl_applied, math.sqrt((x - 0)**2 + (y - 0.45)**2)]
    print("ball never landed:", (data.body(f'ball').xpos[2]))
    return None
    

# def run_dist_test():
#     results = []
#     ctrl_applied = 0.01
#     for i in range(25):
#         ep_rew = run_episode(ctrl_applied)
#         results.append(ep_rew)
#         ctrl_applied += 0.01
#     return results

# testing_results = run_dist_test()

def run_box_test():
    results = []
    ctrl_applied = 0.01
    for i in range(25):
        ep_rew = run_episode(ctrl_applied)
        results.append(ep_rew)
        ctrl_applied += 0.01
    return results

def plot_scatter(data):
    # Unzip the list of [x, y] values
    x_values, y_values = zip(*data)
    
    # Create the scatter plot
    plt.scatter(x_values, y_values)
    
    # Add labels and title
    plt.xlabel('Control Applied')
    plt.ylabel('Reward (distance thrown)')
    plt.title('Scatter Plot of [x, y] values')
    
    # Display the plot
    plt.show()
testing_results = run_box_test()
plot_scatter(testing_results)
# print(testing_results)