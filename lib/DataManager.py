import numpy as np
import torch
from scipy import stats

# Gets histories from the CSV
def GetHistories(path):
    num_episodes = -1 #not used
    histories = []
    with open(path) as file:
        cur_episode = -1
        cur_timestep = -1
        for idx, line in enumerate(file):
            #not used
            if(idx == 0):
                num_episodes = int(line)
                continue
            
            data = line.split(",")
            if(len(data) == 1):
                num_time_steps = int(line)
                cur_episode += 1
                cur_timestep = 0

                traj = torch.zeros((num_time_steps, 4))
                if torch.cuda.is_available():
                    traj = traj.cuda()

                histories.append(np.zeros((num_time_steps, 4)))
                continue
                
            St, At, Rt, pib = data
            histories[cur_episode][cur_timestep, 0] = int(St)
            histories[cur_episode][cur_timestep, 1] = int(At)
            histories[cur_episode][cur_timestep, 2] = float(Rt)
            histories[cur_episode][cur_timestep, 3] = float(pib)         
            cur_timestep += 1
    return histories

# Extracts as much policy info as possible from episode
def GetPolicyFromEpisode(histories, ep_num, num_states, num_actions):
    policy = np.zeros((num_states, num_actions))
    traj = histories[ep_num]
    
    for state in range(num_states):
        for action in range(num_actions):
            valid_idx = np.logical_and(traj[:,0] == state, traj[:,1] == action)
            if(not valid_idx.any()):
                continue
            
            policy[state, action] = traj[valid_idx][0,3]
            
    return policy

# Gets the policy
def GetPolicy(histories, num_states, num_actions, num_iterartions):
    cur_policy = np.zeros((num_states, num_actions))

    # for ep in range(len(histories)):
    for ep in range(num_iterartions):
        temp_p = GetPolicyFromEpisode(histories, ep, num_states, num_actions)
        for state in range(num_states):
            for action in range(num_actions):
                if((cur_policy[state, action] == 0) and (temp_p[state, action] != 0)):
                    cur_policy[state, action] = temp_p[state, action]
    return cur_policy
        