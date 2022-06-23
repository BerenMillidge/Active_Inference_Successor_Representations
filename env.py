import numpy as np
from utils import *

class SR_Gridworld(object):
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.A = np.identity(self.grid_size**2)
        self.B = generate_B_matrices(self.grid_size)
        self.state = np.zeros(self.grid_size**2)

    def reset(self):
        init_idx = int(np.random.uniform(0, self.grid_size**2 -1))
        self.state = np.zeros(self.grid_size**2)
        self.state[init_idx] = 1
        return self.state

    def step(self, a_idx):
        self.state = self.B[:,:,a_idx] @ self.state
        return self.A @ self.state
        
    def get_likelihood_dist(self):
        return self.A

    def get_transition_dist(self):
        return self.B

  