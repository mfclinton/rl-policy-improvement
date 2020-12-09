import numpy as np

class Gridworld():
    def __init__(self, GAMMA, num_eps):
        self.MAX_EPS = num_eps
        self.MAX_TIME_STEPS = 100
        self.STATE_DIM = 23
        self.NUM_ACTIONS = 4
        self.GAMMA = GAMMA
        self.NewEpisode()
        
    
    def NewEpisode(self):
        self.x = 0
        self.y = 0
        self.t = 0
        self.TAS = False
        
    def GetState(self):
        state = 0
        
        if(not self.TAS):
            state = self.y * 5 + self.x
            
            if (state > 12):
                state -= 1
            if (state > 16):
                state -= 1
            
        return state
    
    def Transition(self, a):
        self.t += 1
        if((self.x == 4) and (self.y == 4)):
            self.TAS = True
            return 0
        
        if(self.t == self.MAX_TIME_STEPS):
            self.TAS = True
            return -100
        
        effective_action = a
        temp = np.random.random_sample()
        
        if(temp <= 0.1):
            effective_action = -1 # stay
        elif(temp <= 0.15):
            effective_action = (effective_action + 1) % self.NUM_ACTIONS #rotate
        elif(temp <= 0.2):
            effective_action = (effective_action - 1) % self.NUM_ACTIONS #rotate
            
        x_prime = self.x
        y_prime = self.y
        if ((effective_action == 0) and (self.y >= 1)):
            y_prime -= 1
        elif(effective_action == 1):
            x_prime += 1
        elif(effective_action == 2):
            y_prime += 1
        elif(effective_action == 3):
            x_prime -= 1
        
        # checks location is valid
        if ((x_prime >= 0) and (y_prime >= 0) and (x_prime < 5) and (y_prime < 5) and ((x_prime != 2) or ((y_prime != 2) and (y_prime != 3)))):
            self.x = x_prime
            self.y = y_prime
            
        #compute rewards
        reward = 0
        if((self.x == 2) and (self.y == 4)):
            reward = -10
        elif((self.x == 4) and (self.y == 4)):
            reward = 10
        return reward * (self.GAMMA ** (self.t - 1))

def ExecutePolicy(policy, state):
    temp = np.random.random_sample()
    total = 0
    for a in range(policy.shape[1]):
        total += policy[state,a]
        if(temp < total):
            return a, policy[state,a]
    assert False #Error
    return -1, -1

def RunGridworld(policy, GAMMA=0.95, num_eps = 100000):
    gridworld = Gridworld(GAMMA, num_eps)

    update_freq = gridworld.MAX_EPS * 0.1
    output = str(gridworld.MAX_EPS) + "\n"
    for ep in range(gridworld.MAX_EPS):
        if(ep % update_freq == 0):
            print("Episode: " + str(ep) + " / " + str(gridworld.MAX_EPS))
        gridworld.NewEpisode()
        traj_output = ""
        while not gridworld.TAS:
            state = gridworld.GetState()
            action, pi_s_a = ExecutePolicy(policy, state)
            reward = gridworld.Transition(action)
            traj_output += str(state) + "," + str(action) + "," + str(reward) + "," + str(pi_s_a) + "\n"
        output += str(gridworld.t) + "\n" + traj_output

    return output

def GetGridworldReturn(policy, GAMMA=0.95, num_eps = 100000):
    avg_return = 0
    gridworld = Gridworld(GAMMA, num_eps)
    for ep in range(gridworld.MAX_EPS):
        gridworld.NewEpisode()
        cur_return = 0
        cur_gamma = 1
        while not gridworld.TAS:
            state = gridworld.GetState()
            action, pi_s_a = ExecutePolicy(policy, state)
            reward = gridworld.Transition(action)

            cur_return += reward * cur_gamma
            cur_gamma *= GAMMA
        avg_return += cur_return

    avg_return /= gridworld.MAX_EPS
    return avg_return