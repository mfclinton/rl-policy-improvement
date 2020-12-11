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
                
                #Intialize return and timestep to 0, get the number of timesteps in episode
                cur_return = 0

                num_time_steps = int(line)
                cur_episode += 1
                cur_timestep = 0

                #initializes a traj dict that stores a vector for each column key
                traj = {}
                for key_idx, key in enumerate(column_key):
                    entries = None
                    #NOTE CONVERSION, int for state and action, otherwise float
                    if(key_idx == 0 or key_idx == 1):
                        entries = np.zeros((num_time_steps), dtype=int)
                    else:
                        entries = np.zeros((num_time_steps), dtype=float)

                    traj[key] = entries

                histories.append(traj)
                continue

            # gets the row and enumerates it
            for e_idx, element in enumerate(data):
                #casts the string entry to the type associated in the column types
                value = column_types[e_idx](element)
                if(e_idx == 2):
                    #calculates the return over time
                    cur_return += value * (gamma ** cur_timestep)

                histories[cur_episode][column_key[e_idx]][cur_timestep] = value
     
            cur_timestep += 1
        #edge case last
        histories[cur_episode]["return"] = cur_return
    return histories

#splits data consistently
def SplitData(histories):
    split_idx = int(len(histories) * 0.8)
    train = histories[:split_idx]
    test = histories[split_idx:]
    print(len(train))
    print(len(test))
    return train, test

# Computes the averate return of the policy used to get histories
def GetAverageReturn(histories):
    avg_exploratory_J = 0
    for traj in histories:
        avg_exploratory_J += traj["return"] #sums the cached returns
        
    avg_exploratory_J /= len(histories) #normalize sum of cached returns
    print("Average Baseline Return : " + str(avg_exploratory_J))
    return avg_exploratory_J

def GetTargetPerformance(USE_GRIDWORLD, avg_exploratory_J, percent_increase):
    target_performance = 1.41537
    if(USE_GRIDWORLD):
        target_performance = avg_exploratory_J
        
    target_performance += abs(target_performance)*percent_increase #% increase
    print("Target Performance : " + str(target_performance))
    return target_performance


#-----Policy Stuff----

# Extracts as much policy info as possible from episode
def GetPolicyFromEpisode(histories, ep_num, num_states, num_actions):
    policy = np.zeros((num_states, num_actions))
    traj = histories[ep_num]
    
    #finds finds indexes for (S,A) and takes the associated probability component from it
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
                #if a policy option is 0, but we find a episode where that probability is non-zero, then set it
                if((cur_policy[state, action] == 0) and (temp_p[state, action] != 0)):
                    cur_policy[state, action] = temp_p[state, action]
    return cur_policy

#saves the numpy array
def SaveNumpyPolicy(new_policy, id, delta, USE_GRIDWORLD):
    folder_name = "policies\\van\\delta_" + str(delta) + "\\"
    if(USE_GRIDWORLD):
        folder_name = "policies\\gw\\delta_" + str(delta) + "\\"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    np.save(folder_name + "policy" + str(id), new_policy)

#writes the solution row wise into a text file
def WriteSolutionToTxt(path, solution):
    result_str = ""
    for value in solution.reshape(-1):
        result_str += str(value) + "\n"
    result_str = result_str[:-1] #removes last new line character
    with open(path, "w") as f:
        f.write(result_str)

# loads in a solution numpy file from a txt
def LoadSolutionFromTxt(path, num_states, num_actions):
    loaded_solution = np.zeros(num_states * num_actions)
    with open(path, "r") as f:
        for i, line in enumerate(f):
            loaded_solution[i] = float(line)
    loaded_solution = loaded_solution.reshape(num_states, num_actions)
    return loaded_solution