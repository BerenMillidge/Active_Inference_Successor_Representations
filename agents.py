import numpy as np
from utils import *
from env import *
import matplotlib.pyplot as plt
from copy import deepcopy
import time
import pymdp
from pymdp.agent import Agent
from pymdp import utils, maths
from pymdp.envs import GridWorldEnv
import os

class SR_Agent(object):
    def __init__(self, A=None,B=None,C=None, discount_factor=0.99,inverse_factor=1):
        self.A = A
        self.B = B
        self.C = C
        self.gamma = discount_factor
        self.inverse_factor = inverse_factor
  
    def run_episode(self, env, C = None, time_episode = True):
        if self.A is None:
            self.A = env.get_likelihood_dist()
        if self.B is None:
            self.B = env.get_transition_dist()
        if C is not None:
            self.C = C
        if C is None and self.C is None:
            print("No reward function (C vector) provided!")
            return
        
        ## begin proper function
        # setup successor matrix
        if time_episode == True:
            start_time = time.time()
        grid_size = env.grid_size
        B_avg = np.sum(self.B, axis=2) / len(self.B[:,:,0])

        #print(B_avg.shape)
        I = np.identity(grid_size**2)
        #print(I.shape)
        plt.imshow(B_avg)
        plt.show()
        self.M = np.linalg.inv(I - (self.inverse_factor * B_avg))
        plt.imshow(self.M)
        plt.show()
        V = self.M @ self.C
        plt.imshow(V.reshape(grid_size, grid_size))
        plt.show()
        max_s = onehot_state(np.argmax(self.C), grid_size)
        # init env
        s = env.reset()
        print("INIT S: ", s)
        print("max_s: ", max_s)
        print((s != max_s).any())

        # run the episode until discovered the reward
        N_steps = 0
        total_r = 0
        while (s != max_s).any():
            # hypothetical evaluation
            #print(self.B[:,:,0].shape)
            #print(s.shape)
            s_left = self.B[:,:,0] @ s
            s_right = self.B[:,:,1] @ s
            s_up = self.B[:,:,2] @ s
            s_down = self.B[:,:,3] @ s
            V_left = V[from_onehot(s_left)]
            V_right = V[from_onehot(s_right)]
            V_up = V[from_onehot(s_up)]
            V_down = V[from_onehot(s_down)]
            best_action = np.argmax(np.array([V_left, V_right, V_up, V_down]))
            # so now we actually take the next action
            r = self.C[from_onehot(s)]
            total_r += r
            s = env.step(best_action)
            N_steps +=1
  
        if time_episode == True:
            total_time = time.time() - start_time
            return N_steps, total_r, total_time
        return N_steps, total_r,0



class AIF_Agent(object):
    def __init__(self,env, C, discount_factor=0.99,action_selection="deterministic",policy_len=2, inference_horizon=5):
        self.env = env
        self.C = C
        self.A = self.env.get_likelihood_dist()
        self.B = self.env.get_transition_dist()
        self.discount_factor = discount_factor
        self.agent = Agent(A = self.A, B = self.B, C = self.C,action_selection=action_selection,policy_len = policy_len, inference_horizon=inference_horizon)


    def run_episode(self, time_episode = True):
        print("In AIF run episode")
        if time_episode == True:
            start_time = time.time()
        grid_size = self.env.grid_size
        s = self.env.reset()
        N_steps = 0
        total_r = 0
        max_s = onehot_state(np.argmax(self.C), grid_size)
        print("s: ", s)
        print("max s: ", max_s)
        # begin action-perception loop
        while (s != max_s).any():
            s_idx = from_onehot(s)
            qs = self.agent.infer_states([s_idx])
            q_pi, G = self.agent.infer_policies()
            #print(q_pi)
            next_action = self.agent.sample_action()[0]
            r = self.C[from_onehot(s)]
            total_r += r
            N_steps +=1
            #print(next_action)
            s = self.env.step(int(next_action))
        if time_episode:
            total_time = time.time() - start_time
            return N_steps, total_r, total_time
        return N_steps, total_r,0
      

def compare_SR_AIF_performance(N_runs, max_gridsize):
    SR_steps = []
    SR_rewards = []
    SR_times = []
    AIF_steps = []
    AIF_rewards = []
    AIF_times = []
    for n in range(N_runs):
        print("Run N: ", n)
        SR_step = []
        SR_reward = []
        SR_time = []
        AIF_step = []
        AIF_reward = []
        AIF_time = []
        for grid_size in range(3, max_gridsize):
            print("grid_size: ", grid_size)
            recent_env = SR_Gridworld(grid_size)
            C = np.ones(grid_size**2) * -0.1
            C[len(C)-1] = 10
            # SR agent
            inverse_factor = 1
            if grid_size > 7:
                inverse_factor = 4
        agent = SR_Agent(inverse_factor=inverse_factor)
        N_steps, total_r,total_time = agent.run_episode(recent_env, C=C)
        SR_step.append(deepcopy(N_steps))
        SR_reward.append(deepcopy(total_r))
        SR_time.append(deepcopy(total_time))
        AIF_agent = AIF_Agent(recent_env, C,action_selection="stochastic")
        N_steps, total_r, total_time = AIF_agent.run_episode()
        AIF_step.append(deepcopy(N_steps))
        AIF_reward.append(deepcopy(total_r))
        AIF_time.append(deepcopy(total_time))

        SR_step = np.array(SR_step)
        SR_reward= np.array(SR_reward)
        SR_time = np.array(SR_time)
        AIF_step = np.array(AIF_step)
        AIF_reward = np.array(AIF_reward)
        AIF_time = np.array(AIF_time)

        SR_steps.append(SR_step)
        SR_rewards.append(SR_reward)
        SR_times.append(SR_time)
        AIF_steps.append(AIF_step)
        AIF_rewards.append(AIF_reward)
        AIF_times.append(AIF_time)

    SR_steps = np.array(SR_steps)
    SR_rewards= np.array(SR_rewards)
    SR_times = np.array(SR_times)
    AIF_steps = np.array(AIF_steps)
    AIF_rewards = np.array(AIF_rewards)
    AIF_times = np.array(AIF_times)
    return SR_steps, SR_rewards, SR_times, AIF_steps, AIF_rewards, AIF_times


if __name__ == '__main__':
    SR_steps, SR_rewards, SR_times, AIF_steps, AIF_rewards, AIF_times = compare_SR_AIF_performance(10,10)
    AIF_steps, AIF_rewards, AIF_times = compare_SR_AIF_performance(10,10)
    if not os.path.exists("data/"):
        os.makedirs("data/")
        
    np.save("data/SR_steps_1.npy", SR_steps)
    np.save("data/AIF_steps_1.npy", AIF_steps)
    np.save("data/SR_rewards_1.npy", SR_rewards)
    np.save("data/AIF_rewards_1.npy", AIF_rewards)
    np.save("data/SR_times_1.npy", SR_times)
    np.save("data/AIF_times_1.npy", AIF_times)