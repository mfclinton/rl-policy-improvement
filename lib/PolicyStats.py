import numpy as np
import torch
from scipy import stats

# Importance Sampling Methods
def ImportanceSampling(histories, cur_policy, gamma, ep_num, new_policy):
    traj = histories[ep_num]
    is_weight = 1
    disc_return = 0
    for j in range(traj.shape[0]):
        St, At, Rt, _ = traj[j]
        St = int(St)
        At = int(At)
        is_weight *= new_policy[St, At] / cur_policy[St, At]
        disc_return += (gamma ** j) * Rt
    return is_weight * disc_return

def PDImportanceSampling(histories, cur_policy, gamma, ep_num, new_policy):
    traj = histories[ep_num]
    
    result = 0
    for t in range(traj.shape[0]):
        _, _, Rt, _ = traj[t]
        is_weight = 1
        for j in range(t + 1):
            St, At, _, _ = traj[j]
            St = int(St)
            At = int(At)
            is_weight *= new_policy[St, At] / cur_policy[St, At]
        result += (gamma ** t) * is_weight * Rt
    return result

def CalcAvgIS(histories, cur_policy, gamma, new_policy, ISFunc):
    total = 0
    print("Averaging Importance Sampling")
    update_freq = int(0.25 * len(histories))
    for ep in range(len(histories)):
        if(ep % update_freq == 0):
            print(str(ep) + " / " + str(len(histories)))
        total += ISFunc(histories, cur_policy, gamma, ep, new_policy)
    return total / len(histories)

def CalcStdDev(histories, cur_policy, gamma, new_policy, ISFunc, avgIS):
    total = 0
    for ep in range(len(histories)):
        total += (ISFunc(histories, cur_policy, gamma, ep, new_policy) - avgIS)**2
    
    return np.sqrt((1 / (len(histories) - 1)) * total)

def Safety_Prediction(histories, cur_policy, gamma, new_policy, ISFunc, delta, num_safety, avgIS = None):
    t_value = stats.t.ppf(1-delta, num_safety - 1)
    if(avgIS == None):
        avgIS = CalcAvgIS(histories, cur_policy, gamma, new_policy, ISFunc)
    std_dev = CalcStdDev(histories, cur_policy, gamma, new_policy, ISFunc, avgIS)
    
    return avgIS - 2 * (std_dev / np.sqrt(num_safety)) * t_value

def Safety_Test(histories, cur_policy, gamma, new_policy, ISFunc, delta):
    num_safety = len(histories)
    t_value = stats.t.ppf(1-delta, num_safety - 1)
    avgIS = CalcAvgIS(histories, cur_policy, gamma, new_policy, ISFunc)
    std_dev = CalcStdDev(histories, cur_policy, gamma, new_policy, ISFunc, avgIS)
    
    return avgIS - (std_dev / np.sqrt(num_safety)) * t_value