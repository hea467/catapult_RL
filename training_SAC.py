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

from catapult_env import Catapult_Env

hp_dict = {
    "exp_name": "Linear_End_Effector_Shift",
    "data_dir": "./data/rl_data",
    "tau": 0.005,
    "gamma": 0.99,
    "q_lr": 3e-5,
    "pi_lr": 1e-5,
    "eta_min": 1e-5,
    "alpha": 0.1,  # Changed from 0.2 to a slightly lower value
    "replay_size": 1000000,
    'seed': 69420,
    'optim': 'adam',
    "batch_size": 128,
    "exploration_cutoff": 5000,
    "infer_every": 4000,
    "inference_length": 10,
    "dev_rl": torch.device("cuda:0"),
    "model_dim": 64,
    "num_layers": 2,
    "dropout": 0,
    "max_grad_norm": 1.0,
    "dont_log": True,
    "exp_num": 2
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

max_ep_len = 500
update_freq = 10
# max_training_timesteps = max_ep_len * 3000000
max_training_eps = 30000
# max_training_timesteps = 2e6

printing_freq = 100
visual_freq = 300
pickle_file_path = "saved_episodes/log.pkl"
pickle_update_freq = 1

update_timestep = max_ep_len * update_freq

print_freq = max_ep_len * update_freq
num_threads = 32
act_scale_f = 1

env = [Catapult_Env(max_ep_len) for i in range(num_threads)]
logger_kwargs = {}
single_agent_env_dict = {'action_space': {'low': -0.5, 'high': 0.5, 'dim': 1},
                    'observation_space': {'dim': 4},}

sac_agent = sac.SAC(single_agent_env_dict, hp_dict, logger_kwargs, ma=False, train_or_test="test")

episode_records = []

wandb.init(
    project="Catapult",
    name = "exp_3_lowered_lr",
    config = {
        "gamma"             : 0.99,
        "q_lr"              : 3e-3,
        "pi_lr"             : 1e-4,
    }
)

def convert_act_to_timestep(action, max_ep_length): 
    return round((action +0.5) * max_ep_length)


def convert_timestep_to_act(tp, max_ep_length): 
    return ((tp / 1000) - 0.5)


def parallel_episode(max_ep_len, thread_id):
    #Initialize 
    state = env[thread_id].reset()
    init_state = state.copy()
    #Get action: which time step to release
    #UNCOMMENT IFFF we want to repeat this action for all
    action = sac_agent.get_actions(state)
    #Action is an np array with only one element in it
    timestep_to_release = convert_act_to_timestep(action[0], max_ep_len)
    ep_rew = 0
    assert(timestep_to_release <= max_ep_len and timestep_to_release >= 0)
    data_this_step = None
    for t in range(max_ep_len):
        release_bool = (timestep_to_release == t)
        new_state, reward, done = env[thread_id].step(release_bool)
        ep_rew = reward
        #reset the data always to the latest step
        state = new_state 
        if done:
            break
    
    data_this_step = {"state" : init_state, "action": action, "reward" : ep_rew, "new_state": state, "done": done, "ep_steps": t}
    return data_this_step
    

def thread_initializer():
    thread_local.data = mujoco.MjData(env.model)

def save_list(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    
thread_local = threading.local()

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
            episode_reward = 0
            episode_steps_count = 0
            assert(episode_data != None)
            # We only stored the last step of the episode
            sac_agent.replay_buffer.store(episode_data["state"], episode_data["action"], episode_data["reward"], episode_data["new_state"], episode_data["done"])
            ep_count += 1
            last_rew = episode_data["reward"]
            if (ep_count % printing_freq) == 0:
                print(f"finished episode {ep_count}, last recorded reward: {last_rew}")

            if (ep_count % visual_freq) == 0:
                last_act = episode_data["action"][0]
                last_timestep = convert_act_to_timestep(last_act, max_ep_len)
                episode_records.append(last_timestep)
                if (ep_count % pickle_update_freq) == 0:
                    save_list(episode_records, pickle_file_path)

            ep_length = episode_data["ep_steps"]
            wandb.log({'train/avg_reward': last_rew, 'train/num_episodes' : episode_data["ep_steps"]})
    
    rounds_of_num_thread_ep_completed += 1
    q_losses = np.zeros(num_threads)
    pi_losses = np.zeros(num_threads)
    if ep_count>hp_dict["batch_size"]:
        for i in range(num_threads):
            q_losses[i], pi_losses[i] = sac_agent.update(hp_dict["batch_size"], rounds_of_num_thread_ep_completed, "dist_exp")
        
    wandb.log({'train/q_loss': np.mean(q_losses), 'train/pi_loss' : np.mean(pi_losses)})
    # writer.add_scalar('train/avg_reward', current_ep_reward/curr_ep_tsteps, global_step=env.global_step)
    # writer.add_scalar('train/num_episodes', curr_ep_tsteps, global_step=env.global_step)

    # print("Episode : {} \t Average Reward : {}".format(i_episode, current_ep_reward/curr_ep_tsteps))


# print(env.stored_cirlces)

# model_art = run.log_artifact("./model.pt", type="model")


print("===================== ALL DONE ~~ :) ===============================")