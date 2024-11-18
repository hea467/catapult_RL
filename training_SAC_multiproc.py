import os
import numpy as np
import sac 
import numpy as np
import torch
import multiprocessing
from multiprocessing import Process, Queue, Manager
import mujoco
from sth_env import RL_env

def parallel_episode(env_id, env, max_ep_len, sac_agent, return_dict):
    # Code to run parallel episodes
    state = env.reset() 
    all_data_this_episode = []
    
    for t in range(max_ep_len):
        action = sac_agent.get_actions(state)
        # new_state, reward, done = env.step(action)
        # data_this_step = {"state": state, "action": action, "reward": reward, "new_state": new_state, "done": done}
        # all_data_this_episode.append(data_this_step)
        # state = new_state 
        # if done:
        #     break
    # put necessary data in return dict
    return_dict[env_id] = np.array(all_data_this_episode)
    
if __name__ == '__main__':
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
            "batch_size"        : 256,
            "exploration_cutoff": 512,
            "infer_every"       : 4000,
            "inference_length"  : 10,
            # "dev_sim"           : torch.device(f"cuda:{self.args.dev_sim}"),
            "dev_rl"            : 'cuda' if torch.cuda.is_available() else 'cpu',
            # "dev_rl"            : torch.device("cpu"),
            "model_dim"         : 256,
            "num_heads"         : 8,
            "dim_ff"            : 128,
            "dropout"           : 0,
            "max_grad_norm"     : 1.0,
            "dont_log"          : True,
            "exp_num"           : 2
        }
    print("Started training at (GMT) : ")
    print("============================================================================================")

    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    max_ep_len = 100
    update_freq = 10
    max_training_timesteps = max_ep_len * 3000000

    episode_storage_frequency = 0

    update_timestep = max_ep_len * update_freq

    print_freq = max_ep_len * update_freq
    act_scale_f = 1


    logger_kwargs = {}
    single_agent_env_dict = {'action_space': {'low': -0.25/act_scale_f, 'high': 0.25/act_scale_f, 'dim': 4},
                            'observation_space': {'dim': 65},}
    sac_agent = sac.SAC(single_agent_env_dict, hp_dict, logger_kwargs, ma=False, train_or_test="test")

    #creating an env for every thread / process
    num_processes = 16
    envs = [RL_env(max_ep_len) for i in range(num_processes)]  # Create a single-thread environment for each process

    multiprocessing.set_start_method('spawn')  # Necessary for CUDA compatibility
    
    # num_processes = multiprocessing.cpu_count()  # Use all available CPU cores
    print(f"Using {num_processes} processes")

    # training loop
    time_step = 0
    i_episode = 0

    # parallel_ep_args = [(max_ep_len, sac_agent, envs[i]) for i in range(num_processes)]

    manager = Manager()
    return_dict = manager.dict()
    
    while time_step <= max_training_timesteps:
        
        # , maxtasksperchild = 1000
        # with Pool(processes=num_processes, maxtasksperchild = 1000) as pool:
        #     pool.map(parallel_episode, [(max_ep_len, return_dict, sac_agent, envs[i]) for i in range(num_processes)])
        # q = Queue()
    

        # lock = Lock()
        processes = []

        for i in range(num_processes):
            p = Process(target=parallel_episode, args=(i, envs[i], max_ep_len, sac_agent, return_dict))
            processes.append(p)
            p.start()
        
        for p in processes:
            p.join()

        for env_id, data in return_dict.items():
            print(f"Env {env_id}")


        for i in range(num_processes):
            episode_data = return_dict[i]
            episode_reward = 0
            episode_steps_count = 0
            for step_data in episode_data:
                sac_agent.replay_buffer.store(step_data["state"], step_data["action"], step_data["reward"], step_data["new_state"], step_data["done"])
                time_step += 1
                episode_steps_count += 1
            
            last_step = episode_data[-1]
            episode_reward = last_step["reward"]
        
        q_losses = np.zeros(num_processes)
        pi_losses = np.zeros(num_processes)
        if time_step > hp_dict["batch_size"] * 2:
            for i in range(num_processes):
                q_losses[i], pi_losses[i] = sac_agent.update(hp_dict["batch_size"], time_step)
        
        # wandb.log({'train/avg_reward': episode_reward, 'train/num_episodes': episode_steps_count, 'train/q_loss': np.mean(q_losses), 'train/pi_loss': np.mean(pi_losses)})
        i_episode += num_processes

        #Store episode to file when rest of code is more ready 
        # if i_episode % episode_storage_frequency == 0: 
        #         self.stored_all_obs.append(self.current_obs)
        #         db_file = open("runtime_data/position.pkl",'wb' )
        #         pkl.dump(self.stored_all_obs, db_file)
        #         db_file.close
        #         # print(self.stored_all_obs)
        #         self.current_obs = []
