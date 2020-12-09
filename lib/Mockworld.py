import numpy as np

class Mockworld():
    def __init__(self, GAMMA, num_eps):
        self.MAX_EPS = num_eps
        self.NUM_ACTIONS = 4
        self.GAMMA = GAMMA
        self.NewEpisode()
        
    
    def NewEpisode(self):
        self.first_action = -1
        self.second_action = -1
        self.x = 0
        self.y = 0
        self.t = 0
        self.TAS = False
        
    def GetState(self):
        state = 15
        if(self.t == 0):
            state = 17
        elif(self.t == 1):
            state = 16
        elif(not self.TAS):
            state = self.y * 4 + self.x
            
        return state
    
    def Transition(self, a):
        self.t += 1
        reward = 0
        if(self.t == 1):
            self.first_action = a
            if(self.first_action >= 2):
                reward = 1
        elif(self.t == 2):
            self.second_action = a
        else:   
            x_prime = self.x
            y_prime = self.y
            if (a == 0):
                y_prime -= 1
            elif(a == 1):
                y_prime += 1
            elif(a == 2):
                x_prime -= 1
            elif(a == 3):
                x_prime += 1
        
            # checks location is valid
            if ((x_prime >= 0) and (y_prime >= 0) and (x_prime < 4) and (y_prime < 4)):
                self.x = x_prime
                self.y = y_prime

            if((self.x == 3) and (self.y == 3)):
                self.TAS = True
                reward = 1
                if((self.first_action < 2 and self.second_action < 2) or (self.first_action >= 2 and self.second_action >= 2)):
                    reward = 10
            
        #compute rewards
        return reward #* (self.GAMMA ** (self.t - 1))

def ExecutePolicy(policy, state):
    temp = np.random.random_sample()
    total = 0
    for a in range(policy.shape[1]):
        total += policy[state,a]
        if(temp < total):
            return a, policy[state,a]
    assert False #Error
    return -1, -1

def RunMockworld(policy, GAMMA=0.95, num_eps = 100000):
    mockworld = Mockworld(GAMMA, num_eps)

    update_freq = mockworld.MAX_EPS * 0.1
    output = str(mockworld.MAX_EPS) + "\n"
    # output = ""
    for ep in range(mockworld.MAX_EPS):
        if(ep % update_freq == 0):
            print("Episode: " + str(ep) + " / " + str(mockworld.MAX_EPS))
        mockworld.NewEpisode()
        traj_output = ""
        while not mockworld.TAS:
            state = mockworld.GetState()
            action, pi_s_a = ExecutePolicy(policy, state)
            reward = mockworld.Transition(action)
            traj_output += str(state) + "," + str(action) + "," + str(reward) + "," + str(pi_s_a) + "\n"
        output += str(mockworld.t) + "\n" + traj_output

    return output

def GetMockworldReturn(policy, GAMMA=0.95, num_eps = 100000):
    avg_return = 0
    mockworld = Mockworld(GAMMA, num_eps)
    for ep in range(mockworld.MAX_EPS):
        mockworld.NewEpisode()
        cur_return = 0
        cur_gamma = 1
        while not mockworld.TAS:
            state = mockworld.GetState()
            action, pi_s_a = ExecutePolicy(policy, state)
            reward = mockworld.Transition(action)

            cur_return += reward * cur_gamma
            cur_gamma *= GAMMA
        avg_return += cur_return

    avg_return /= mockworld.MAX_EPS
    return avg_return