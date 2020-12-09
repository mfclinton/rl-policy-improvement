import numpy as np
import torch
from scipy import stats
import os

# Gets histories from the CSV
def GetHistories(path, gamma):
    histories = []
    with open(path) as file:
        update_interval = 1000000
        num_episodes = -1 #not used
        cur_episode = -1
        cur_timestep = -1
        column_key = ["St", "At", "Rt", "pib"]
        column_types = [int,int,float,float]
        cur_return = None
        for idx, line in enumerate(file):
            if(idx % update_interval == 0):
                print("line " + str(idx))
            #not used
            if(idx == 0):
                num_episodes = int(line)
                continue
            
            data = line.split(",")
            if(len(data) == 1):
                if cur_return != None:
                    histories[cur_episode]["return"] = cur_return
                cur_return = 0

                num_time_steps = int(line)
                cur_episode += 1
                cur_timestep = 0

                traj = {}
                for key_idx, key in enumerate(column_key):
                    entries = None
                    #NOTE CONVERSION
                    if(key_idx == 0 or key_idx == 1):
                        entries = np.zeros((num_time_steps), dtype=int)
                    else:
                        entries = np.zeros((num_time_steps), dtype=float)

                    traj[key] = entries

                histories.append(traj)
                continue

            for e_idx, element in enumerate(data):
                value = column_types[e_idx](element)
                if(e_idx == 2):
                    #reward
                    cur_return += value * (gamma ** cur_timestep)

                histories[cur_episode][column_key[e_idx]][cur_timestep] = value
     
            cur_timestep += 1
        #edge case last
        histories[cur_episode]["return"] = cur_return
    return histories

# Extracts as much policy info as possible from episode
def GetPolicyFromEpisode(histories, ep_num, num_states, num_actions):
    policy = np.zeros((num_states, num_actions))
    traj = histories[ep_num]
    
    for state in range(num_states):
        for action in range(num_actions):
            valid_idx = np.logical_and(traj["St"] == state, traj["At"] == action)
            if(not valid_idx.any()):
                continue
            
            policy[state, action] = traj["pib"][valid_idx][0]
            
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

def SavePolicy(new_policy, J_safety_lower_bound, delta, USE_GRIDWORLD):
    folder_name = "policies\\delta_" + str(delta) + "\\"
    if(USE_GRIDWORLD):
        folder_name = "policies\\gw\\delta_" + str(delta) + "\\"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    np.save(folder_name + "safety_" + str(J_safety_lower_bound), new_policy)