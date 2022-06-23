import numpy as np
import matplotlib.pyplot as plt
from utils import *
import os

def visualize_main_matrices(grid_size):
    B = generate_B_matrices(grid_size)
    plt.imshow(B[:,:,1])
    plt.title("B matrix for go-right action", fontsize=20)
    plt.xticks(list(range(0,grid_size**2)),fontsize=18)
    plt.yticks(list(range(0, grid_size**2)),fontsize=18)
    plt.savefig("B_matrix_right.png", format="png")
    plt.show()
    avg_B = np.sum(B, axis=2) / len(B[:,:,0])
    plt.title("Default Policy B Matrix",fontsize=20)
    plt.imshow(avg_B)
    plt.xticks(list(range(0,grid_size**2)),fontsize=12)
    plt.yticks(list(range(0, grid_size**2)),fontsize=12)
    plt.savefig("default_B_matrix_2.png", format="png")
    plt.show()

    I = np.identity(grid_size**2)
    M = np.linalg.inv(I - (2 * avg_B))
    plt.imshow(M)
    plt.xticks(list(range(0,grid_size**2)),fontsize=12)
    plt.yticks(list(range(0,grid_size**2)),fontsize=12)
    plt.title("Successor Matrix", fontsize=20)
    plt.savefig("Successor_matrix_2.png", format="png")
    plt.show()
    R = np.ones(grid_size**2) * -0.1
    R[8] = 10
    print(R)
    V = M @ R
    print(V)
    plt.imshow(V.reshape(grid_size,grid_size))
    plt.xticks(list(range(0,grid_size)),fontsize=12)
    plt.yticks(list(range(0,grid_size)),fontsize=12)
    plt.title("Estimated Value Function",fontsize=20)
    plt.savefig("estimated_value_function_2.png", format="png")
    plt.show()
    return M,R
    
def visualize_entropic_matrices(grid_size):
    print(grid_size)
    A_visible = np.identity(grid_size**2)
    print(A_visible)
    A_perturbed = deepcopy(A_visible)
    A_perturbed[:,1] = 1 / grid_size**2
    A_perturbed[:,12] = 1/grid_size**2
    print(A_perturbed)
    qs = np.ones(grid_size**2) * 1/(grid_size**2)
    H_A = entropy(A_perturbed)
    H_A_visible = entropy(A_visible)
    plt.imshow(H_A.reshape(4,4))
    plt.title("Observation Entropy", fontsize=20)
    plt.xticks(list(range(0,grid_size)),fontsize=18)
    plt.yticks(list(range(0,grid_size)),fontsize=18)
    plt.tight_layout()
    plt.savefig("observation_entropy_matrix.png", format="png")
    plt.show()
    
    B = generate_B_matrices(grid_size)
    avg_B = np.sum(B, axis=2) / len(B[:,:,0])
    I = np.identity(grid_size**2)
    M = np.linalg.inv(I - (2 * avg_B))
    R = np.ones(grid_size**2) * -0.1
    R[8] = 10
    V = M @ R
    
    G_R = R + H_A
    print(G_R)
    V_infogain = M @ G_R
    
    print(V_infogain)
    plt.imshow(V_infogain.reshape(4,4))
    plt.title("Successor EFE Value Function", fontsize=20)
    plt.xticks(list(range(0,grid_size)),fontsize=18)
    plt.yticks(list(range(0,grid_size)),fontsize=18)
    plt.tight_layout()
    plt.savefig("successor_EFE_value_function.png", format="png")
    plt.show()
    plt.imshow(V.reshape(4,4))
    plt.title("Reward Value Function", fontsize=20)
    plt.xticks(list(range(0,grid_size)),fontsize=18)
    plt.yticks(list(range(0,grid_size)),fontsize=18)
    plt.tight_layout()
    plt.savefig("reward_value_function.png", format="png")
    plt.show()
    plt.imshow(A_perturbed.reshape(16,16))
    plt.xticks([])
    plt.yticks([])
    plt.title("Entropic A Matrix",fontsize=20)
    plt.tight_layout()
    plt.savefig("entropic_A_matrix.png", format="png")
    plt.show()

def plot_SR_AIF_rewards():
    SR_rewards = np.load("data/SR_rewards_1.npy")
    print(SR_rewards)
    AIF_rewards = np.load("data/AIF_rewards_1.npy")
    mean_SR_rewards = np.mean(SR_rewards, axis=0)
    mean_AIF_rewards = np.mean(AIF_rewards, axis=0)
    std_SR_rewards = np.std(SR_rewards, axis=0) / np.sqrt(len(SR_rewards))
    std_AIF_rewards = np.std(AIF_rewards, axis=0) / np.sqrt(len(AIF_rewards))
    xs = range(3,10)
    fig = plt.figure(figsize=(14,10))
    plt.plot(xs,mean_SR_rewards,label="SR")
    plt.fill_between(xs, mean_SR_rewards - std_SR_rewards, mean_SR_rewards + std_SR_rewards, alpha=0.5)
    plt.plot(xs,mean_AIF_rewards,label="AIF")
    plt.fill_between(xs, mean_AIF_rewards - std_AIF_rewards, mean_AIF_rewards + std_AIF_rewards, alpha=0.5)

    plt.title("Reward Obtained across grid sizes",fontsize=30)
    plt.xlabel("Grid Size", fontsize=28)
    plt.ylabel("Total Reward", fontsize=28)
    plt.legend(fontsize=40)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.tight_layout()
    plt.savefig("figures/reward_obtained.png", format="png")
    plt.show()

def plot_SR_AIF_times():
    SR_times = np.load("data/SR_times_1.npy")
    AIF_times = np.load("data/AIF_times_1.npy")
    mean_SR_times = np.mean(SR_times, axis=0)
    mean_AIF_times = np.mean(AIF_times, axis=0)
    std_SR_times = np.std(SR_times, axis=0) / np.sqrt(len(SR_times))
    std_AIF_times = np.std(AIF_times, axis=0) / np.sqrt(len(AIF_times))
    xs = range(3,10)
    fig = plt.figure(figsize=(14,10))
    plt.plot(xs,mean_SR_times, label="SR")
    plt.fill_between(xs, mean_SR_times - std_SR_times, mean_SR_times + std_SR_times, alpha=0.5)
    plt.plot(xs,mean_AIF_times, label="AIF")
    plt.fill_between(xs, mean_AIF_times - std_AIF_times, mean_AIF_times + std_AIF_times, alpha=0.5)
    plt.title("Processing Time per Episode",fontsize=30)
    plt.xlabel("Grid Size", fontsize=28)
    plt.ylabel("Mean Processing Time (s)", fontsize=28)
    plt.legend(fontsize=40,loc="upper left")
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.tight_layout()
    plt.savefig("figures/processing_times_2.png", format="png")
    plt.show()
    
def plot_SR_AIF_steps():
    SR_steps = np.load("data/SR_steps_1.npy")
    AIF_steps = np.load("data/AIF_steps_1.npy")
    mean_SR_steps = np.mean(SR_steps, axis=0)
    mean_AIF_steps = np.mean(AIF_steps, axis=0)
    std_SR_steps = np.std(SR_steps, axis=0) / np.sqrt(len(SR_steps))
    std_AIF_steps = np.std(AIF_steps, axis=0) / np.sqrt(len(AIF_steps))
    xs = range(3,10)
    fig = plt.figure(figsize=(14,10))
    plt.plot(xs,mean_SR_steps,label="SR")
    plt.fill_between(xs, mean_SR_steps - std_SR_steps, mean_SR_steps + std_SR_steps, alpha=0.5)
    plt.plot(xs,mean_AIF_steps,label="AIF")
    plt.fill_between(xs, mean_AIF_steps - std_AIF_steps, mean_AIF_steps + std_AIF_steps, alpha=0.5)

    plt.title("Number of Steps to Success",fontsize=30)
    plt.xlabel("Grid Size", fontsize=28)
    plt.ylabel("Mean Steps", fontsize=28)
    plt.legend(fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.tight_layout()
    plt.savefig("num_steps_2.png", format="png")
    plt.show()
    
if __name__ == '__main__':
    # create dirs
    if not os.path.exists("figures/"):
        os.makedirs("figures/")
    
    grid_size=3
    visualize_main_matrices(grid_size=3)
    visualize_entropic_matrices(grid_size=4)
    plot_SR_AIF_rewards()
    plot_SR_AIF_times()