import os
import numpy as np
import sac 
from gymnasium.spaces import Box
import numpy as np
from scipy.optimize import minimize
from scipy import optimize
import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import threading 
import concurrent.futures
import mujoco
import wandb
import pickle

import matplotlib.pyplot as plt

from catapult_env_height import Catapult_Env

max_ep_len = 500

env = Catapult_Env(max_ep_len) 


def convert_act_to_timestep(action, max_ep_length): 
    return round((action +0.5) * max_ep_length)


def convert_timestep_to_act(tp, max_ep_length): 
    return ((tp / 1000) - 0.5)


def episode(max_ep_len, step_i):
    #Initialize 
    state = env.reset()
    init_state = state.copy()
    #Get action: which time step to release 
    #Action is an np array with only one element in it
    timestep_to_release = step_i
    ep_rew = 0
    assert(timestep_to_release <= max_ep_len and timestep_to_release >= 0)
    for t in range(max_ep_len):
        release_bool = (timestep_to_release == t)
        new_state, reward, done = env.step(release_bool)
        ep_rew += reward
        #reset the data always to the latest step
        state = new_state 
        if done:
            break
    print(ep_rew)
    return ep_rew
    
results = []

for i in range(500):
    rew = episode(500, i)
    results.append(rew)


def plot_list(data):
    # Create a list of indices for x-axis
    indices = list(range(len(data)))

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(indices, data, marker='o')
    plt.title('Plot of Float Values')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)

    # Save the plot as a PNG file
    plt.savefig('float_values_plot.png')
    print("Plot saved as 'float_values_plot.png'")

    # Display the plot (optional, comment out if running on a server without display)
    plt.show()

plot_list(results)


# training loop


print("===================== ALL DONE ~~ :) ===============================")