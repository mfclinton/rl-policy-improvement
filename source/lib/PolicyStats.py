import numpy as np
import torch
from scipy import stats

# T IS INCLUSIVE, TODO double check
def VectorizedImportantWeight(traj, cur_policy, new_policy):
    is_weight_vec = new_policy[traj["St"],traj["At"]] / cur_policy[traj["St"],traj["At"]]
    return is_weight_vec

# Importance Sampling Methods
def ImportanceSampling(traj, cur_policy, gamma, new_policy):
    is_weight = np.prod(VectorizedImportantWeight(traj, cur_policy, new_policy))
    disc_return = traj["return"]
    return is_weight * disc_return

# TODO double check
def PDImportanceSampling(traj, cur_policy, gamma, new_policy):
    result = 0
    is_weight_vec = VectorizedImportantWeight(traj, cur_policy, new_policy)
    is_weight = 1
    for t in range(traj["St"].shape[0]):
        Rt = traj["Rt"][t]
        is_weight *= is_weight_vec[t]
        result += (gamma ** t) * is_weight * Rt
    return result

#calculates the average importance samples over all trajectories
def CalcAvgIS(histories, cur_policy, gamma, new_policy, ISFunc):
    total = 0
    # print("Averaging Importance Sampling")
    # update_freq = int(0.25 * len(histories))
    for ep in range(len(histories)):
        # if(ep % update_freq == 0):
        #     print(str(ep) + " / " + str(len(histories)))
        total += ISFunc(histories[ep], cur_policy, gamma, new_policy)
    return total / len(histories)

def CalcStdDev(histories, cur_policy, gamma, new_policy, ISFunc, avgIS):
    total = 0
    for ep in range(len(histories)):
        total += (ISFunc(histories[ep], cur_policy, gamma, new_policy) - avgIS)**2
    
    return np.sqrt((1 / (len(histories) - 1)) * total)

#Predicts a lowerbound with candidate data
def Safety_Prediction(histories, cur_policy, gamma, new_policy, ISFunc, delta, num_safety, avgIS = None):
    t_value = stats.t.ppf(1-delta, num_safety - 1)
    if(avgIS == None):
        avgIS = CalcAvgIS(histories, cur_policy, gamma, new_policy, ISFunc)
    std_dev = CalcStdDev(histories, cur_policy, gamma, new_policy, ISFunc, avgIS)
    
    return avgIS - 2 * (std_dev / np.sqrt(num_safety)) * t_value

#predicts a lower bound with safety data
def Safety_Test(histories, cur_policy, gamma, new_policy, ISFunc, delta):
    num_safety = len(histories)
    t_value = stats.t.ppf(1-delta, num_safety - 1)
    avgIS = CalcAvgIS(histories, cur_policy, gamma, new_policy, ISFunc)
    std_dev = CalcStdDev(histories, cur_policy, gamma, new_policy, ISFunc, avgIS)
    
    return avgIS - (std_dev / np.sqrt(num_safety)) * t_value

#Confirms the actual return is lower bounded
def ConfirmBounds(is_lower_bounded, value, train, test, exploration_policy, gamma, new_policy, ISFunc, delta):
    J_bl_predicted_lower_bound = Safety_Prediction(train, exploration_policy, gamma, new_policy, ISFunc, delta, len(test))
    J_bl_safety_lower_bound = Safety_Test(test, exploration_policy, gamma, new_policy, ISFunc, delta)
    print("Value: " + str(value))
    print("Predicted Baseline: " + str(J_bl_predicted_lower_bound))
    print("Safety Baseline: " + str(J_bl_safety_lower_bound))

    c = 1
    if(not is_lower_bounded):
        c = -1

    # Ensures Lower Bound Is Lower, Otherwise Investigate
    bl_pred_looseness = (value - J_bl_predicted_lower_bound) * c
    bl_safety_looseness = (value - J_bl_safety_lower_bound) * c
    print("---distance of return from lower bounds ---")
    print("Looseness Of Prediction : " + str(bl_pred_looseness))
    print("Looseness Of Safety : " + str(bl_safety_looseness))

    if((bl_pred_looseness < 0) or (bl_safety_looseness < 0)):
        raise Exception("Lower Bound Greater Than Average Return!")
    print('-------------------------------------------')