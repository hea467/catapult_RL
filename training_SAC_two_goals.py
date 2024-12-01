import os
import numpy as np
import sac_mod as sac
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

from catapult_env_two_goals import Catapult_Env

hp_dict = {
            "exp_name"          : "Linear_End_Effector_Shift",
            "data_dir"          : "./data/rl_data",
            "tau"               : 0.005,
            "gamma"             : 0.99,
            "q_lr"              : 3e-4,
            "pi_lr"             : 1e-4,
            "eta_min"           : 1e-5,
            "alpha"             : 0.2,
            "replay_size"       : 10000,
            'seed'              : 69420,
            'optim'             : 'adam',
            "batch_size"        : 256,
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
            "max_grad_norm"     : 5.0,
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

ep_count = 0

# action_std_decay_freq = 0.05 
# min_action_std = 0.1

completed_circles = 0

max_ep_len = 200
update_freq = 10
# max_training_timesteps = max_ep_len * 3000000
max_training_eps = 70000
# max_training_timesteps = 2e6

printing_freq = 100
visual_freq = 300
pickle_file_path = "saved_episodes_height/log.pkl"
pickle_update_freq = 1

update_timestep = max_ep_len * update_freq

print_freq = max_ep_len * update_freq
num_threads = 4
act_scale_f = 1

log_to_wandb = True

env = [Catapult_Env(max_ep_len) for i in range(num_threads)]
logger_kwargs = {}
single_agent_env_dict = {'action_space': {'low': -0.25, 'high': 0.25, 'dim': 2},
                    'observation_space': {'dim': 5},}

sac_agent = sac.SAC(single_agent_env_dict, hp_dict, logger_kwargs, ma=False, train_or_test="test")

episode_records = []

if log_to_wandb:
    wandb.init(
        project="Catapult",
        name = "two_goal_no_tp",
        config = {
            "gamma"             : 0.99,
            "q_lr"              : 3e-5,
            "pi_lr"             : 1e-5,
        }
    )


def parallel_episode(max_ep_len, thread_id):
    #Initialize 
    state = env[thread_id].reset()
    init_state = state.copy()
    ep_rew = 0
    data_this_step= None 
    action = sac_agent.get_actions(state)
    timestep_to_apply_control = int(action[0]) * max_ep_len
    force_action = action[1]
    for t in range(max_ep_len):
        if timestep_to_apply_control < 0 :
            break
        if t == timestep_to_apply_control:  # Apply control only at the chosen timestep
            new_state, reward, done, current_goal = env[thread_id].step(force_action)
        elif t < timestep_to_apply_control:
             new_state, reward, done, current_goal = env[thread_id].step(0) 
        else:
            new_state, reward, done, current_goal = env[thread_id].step(force_action) 
            #maintain ctrl seems best 
        state = new_state
        ep_rew = reward
        if done:
            break 
    data_this_step = {"state" : init_state, "action": action, "reward" : ep_rew, "new_state": state, "done": done, "ep_steps": t, "current_goal" : current_goal}
    return data_this_step
    

def thread_initializer():
    thread_local.data = mujoco.MjData(env.model)

def save_list(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    
thread_local = threading.local()

print(f"print nextwork device: {sac_agent.device}")
# training loop
rounds_of_num_thread_ep_completed = 0
#I'm getting rid of the concept here of time step and instead just tracking episode
while ep_count <= max_training_eps:
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
            episode_steps_count = 0
            assert(episode_data != None)
            # We only stored the last step of the episode
            sac_agent.replay_buffer.store(episode_data["state"], episode_data["action"], episode_data["reward"], episode_data["new_state"], episode_data["done"])
            ep_count += 1
            last_rew = episode_data["reward"]
            if (ep_count % printing_freq) == 0:
                last_act = (int(abs(episode_data["action"][0]) * max_ep_len), abs(episode_data["action"][1]))
                goal_current = episode_data["current_goal"]
                done = episode_data["done"]
                print(f"finished episode {ep_count}, last recorded reward: {last_rew}, last action : {last_act}, done : {done}, goal : {goal_current}")

            ep_length = episode_data["ep_steps"]
            if log_to_wandb:
                wandb.log({'train/avg_reward': last_rew, 'train/num_episodes' : ep_length})
    
    rounds_of_num_thread_ep_completed += 1
    q_losses = np.zeros(num_threads)
    pi_losses = np.zeros(num_threads)
    if ep_count>hp_dict["batch_size"]:
        for i in range(num_threads):
            q_losses[i], pi_losses[i], alpha_loss, alpha = sac_agent.update(hp_dict["batch_size"], rounds_of_num_thread_ep_completed, "multi_goal_exp")
    if log_to_wandb:
        wandb.log({'train/q_loss': np.mean(q_losses), 'train/pi_loss' : np.mean(pi_losses)})



print("===================== ALL DONE ~~ :) ===============================")