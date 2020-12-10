#Sets state 15 to be uniform
import os
from DataManager import *
import numpy as np
directory = "C:\\Users\\mfcli\\Documents\\School\\F20\\687\\rl-policy-improvement\\"
for filename in os.listdir(directory):
    shared_name = "policy"
    idx = filename.find(shared_name)
    if(idx == -1):
        continue
    
    filepath = directory + filename
    solution = None
    with open(filepath) as f:
        print(filename)
        solution = LoadSolutionFromTxt(filepath, 18, 4)
        solution[15,:] = 1
    
    if np.any(solution != None):
        WriteSolutionToTxt(filepath, solution)
    else:
        print("ERROR WITH " + filename)