import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import math

def state_to_idx(x,y,grid_size):
    return (x * grid_size) + y

def idx_to_state(idx, grid_size):
    x,y = divmod(idx, grid_size)
    return x,y

def onehot_state(idx, grid_size):
    state = np.zeros(grid_size**2)
    state[idx] = 1
    return state

def from_onehot(s):
    for i in range(len(s)):
        if s[i] != 0:
            return i
  
def generate_B_matrices(grid_size):
    N = grid_size**2
    N_size = (N,N)
    B_up = np.zeros(N_size)
    B_down = np.zeros(N_size)
    B_L = np.zeros(N_size)
    B_R = np.zeros(N_size)
    B = np.zeros((N,N,4))
    for i in range(N):
        for j in range(N):
            start_x, start_y = idx_to_state(i, grid_size)
            end_x, end_y = idx_to_state(j, grid_size)
            # left matrix
            if start_x ==0:
                if start_x == end_x and start_y == end_y:
                    B_L[i,j] = 1
            if start_x != 0:
                if end_x == start_x - 1 and start_y == end_y:
                    B_L[i,j] = 1
            # right matrix
            if start_x == grid_size-1:
                if start_x == end_x and start_y == end_y:
                    B_R[i,j] = 1
            if start_x != grid_size -1:
                if end_x == start_x + 1 and start_y == end_y:
                    B_R[i,j] = 1
            # up matrix
            if start_y == 0:
                if start_y == end_y and start_x == end_x:
                    B_up[i,j] = 1
            if start_y != 0:
                if end_y == start_y - 1 and start_x == end_x:
                    B_up[i,j] = 1
            # down matrix
            if start_y == grid_size-1:
                if start_y == end_y and start_x == end_x:
                    B_down[i,j] = 1
            if start_y != grid_size -1:
                if end_y == start_y + 1 and start_x == end_x:
                    B_down[i,j] = 1
    B[:,:,0] = B_L.T
    B[:,:,1] = B_R.T
    B[:,:,2] = B_up.T
    B[:,:,3] = B_down.T
    print(B.shape)
    return B
        
        
def log_stable(x):
    return np.log(x + 1e-5)

def entropy(A):
    H_A = - (A * log_stable(A)).sum(axis=0)
    return H_A