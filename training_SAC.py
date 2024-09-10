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

from walker_specific_env import Cube_Block_Env

hp_dict = {
            "exp_name"          : "Linear_End_Effector_Shift",
            "data_dir"          : "./data/rl_data",
            "tau"               : 0.005,
            "gamma"             : 0.99,
            "q_lr"              : 3e-3,
            "pi_lr"             : 1e-4,
            "eta_min"           : 1e-5,
            "alpha"             : 0.2,
            "replay_size"       : 50000000,
            'seed'              : 69420,
            'optim'             : 'adam',
            "batch_size"        : 32,
            "exploration_cutoff": 512,
            "infer_every"       : 4000,
            "inference_length"  : 10,
            # "dev_sim"           : torch.device(f"cuda:{self.args.dev_sim}"),
            "dev_rl"            : torch.device("cuda:0"),
            # "dev_rl"            : torch.device("cpu"),
            "model_dim"         : 256,
            "num_heads"         : 8,
            "dim_ff"            : 128,
            "dropout"           : 0,
            "max_grad_norm"     : 1.0,
            "dont_log"          : True,
            "exp_num"           : 2
        }


print("Started training: ")

print("============================================================================================")



# printing and logging variables
print_running_reward = 0
print_running_episodes = 0

log_running_reward = 0
log_running_episodes = 0

time_step = 0
i_episode = 0

# action_std_decay_freq = 0.05 
# min_action_std = 0.1

completed_circles = 0

max_ep_len = 50
update_freq = 10
# max_training_timesteps = max_ep_len * 3000000
max_training_timesteps = max_ep_len * 30000
# max_training_timesteps = 2e6

update_timestep = max_ep_len * update_freq

print_freq = max_ep_len * update_freq
num_threads = 128
act_scale_f = 1

env = Cube_Block_Env(max_ep_len, num_threads)
logger_kwargs = {}
single_agent_env_dict = {'action_space': {'low': -0.25/act_scale_f, 'high': 0.25/act_scale_f, 'dim': 4},
                    'observation_space': {'dim': 65},}

sac_agent = sac.SAC(single_agent_env_dict, hp_dict, logger_kwargs, ma=False, train_or_test="test")

writer = SummaryWriter(f'runs/gaol_condition_exp{hp_dict["exp_num"]}')
wandb.init(
    project="Origami RL",
    name = "exp_0",
    config = {
        "gamma"             : 0.99,
        "q_lr"              : 3e-3,
        "pi_lr"             : 1e-4,
    }
)

def parallel_episode(max_ep_len, thread_id):
    all_data_this_episode = []
    # all_data_this_episode = {}
    state = env.reset(thread_id)
    
    for t in range(max_ep_len):
        # if time_step < 33: 
        #     action = np.array([0, 0, 0, 0])
        # else:
        action = sac_agent.get_actions(state)
        # action = np.clip(action, 0, 0.5)
        new_state, reward, done = env.step(action, thread_id)
        # print([state,  *action, reward, new_state, done])
        data_this_step = {"state" : state, "action": action, "reward" : reward, "new_state": new_state, "done": done}
        #all_data_this_episode[t] = np.array([state,  *action, reward, new_state, done])
        all_data_this_episode.append(data_this_step)
        state = new_state 
        if done:
            break
    #shoould be like 10, uncpmment later when things are running
    # print(len(all_data_this_episode))
    # result = all_data_this_episode.items()
    # data = np.array(list(result))
    return np.array(all_data_this_episode)
    
def thread_initializer():
    thread_local.data = mujoco.MjData(env.model)
    
thread_local = threading.local()

# training loop
while time_step <= max_training_timesteps:
    
    current_ep_reward = 0
    q_loss_mean = 0
    # pi_loss_mean = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for i in range(num_threads):
            episode_straining_steps = executor.submit(parallel_episode, max_ep_len, i)
            futures.append(episode_straining_steps)
            
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            episode_data = future.result()
            episode_reward = 0
            episode_steps_count = 0
            for step_data in episode_data:
                sac_agent.replay_buffer.store(step_data["state"], step_data["action"], step_data["reward"], step_data["new_state"], step_data["done"])
                time_step += 1
                episode_steps_count += 1
                # if step_data["done"]:
                #     break
            last_step = episode_data[-1]
            episode_reward = last_step["reward"]
            writer.add_scalar('train/avg_reward', episode_reward, global_step=time_step)
            writer.add_scalar('train/num_episodes', episode_steps_count, global_step=time_step)
            wandb.log({'train/avg_reward': episode_reward, 'train/num_episodes' : episode_steps_count})
            episode_steps_count = 0
            

    q_losses = np.zeros(num_threads)
    pi_losses = np.zeros(num_threads)
    if time_step>hp_dict["batch_size"]*2:
        for i in range(num_threads):
            q_losses[i], pi_losses[i] = sac_agent.update(hp_dict["batch_size"], env.global_step)
        
    writer.add_scalar('train/q_loss', np.mean(q_losses), time_step)
    writer.add_scalar('train/pi_loss', np.mean(pi_losses), time_step)
    wandb.log({'train/q_loss': np.mean(q_losses), 'train/pi_loss' : np.mean(pi_losses)})
    # writer.add_scalar('train/avg_reward', current_ep_reward/curr_ep_tsteps, global_step=env.global_step)
    # writer.add_scalar('train/num_episodes', curr_ep_tsteps, global_step=env.global_step)

    # print("Episode : {} \t Average Reward : {}".format(i_episode, current_ep_reward/curr_ep_tsteps))

    i_episode += num_threads

# print(env.stored_cirlces)

model_art = run.log_artifact("./model.pt", type="model")


def scatter_plot_coordinates(coordinates, n):
    # Extract x and y values from the list of tuples
    x_values = [0.2 for coord in coordinates]
    y_values = coordinates
    
    # Create a scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(x_values, y_values, color='b', s=5)  # 's' sets the marker size
    plt.scatter(x_values[0], y_values[0], color='r', s=5)
    plt.title('Scatter Plot of Coordinates')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.axis('equal')
    plt.grid(True)
    plt.savefig(f'outputs/stored_circle_{n}.png')
    # plt.show()

# for n in env.stored_trajectory.keys():
#     scatter_plot_coordinates(env.stored_trajectory[n], n)
    
writer.close()

print("===================== ALL DONE ~~ :) ===============================")