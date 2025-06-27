import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from policy import policy, policy2
from Qfunc import Q,Q2
from env2 import grid
import os
import matplotlib.pyplot as plt
from a22c import A2C
from rep import Representation, Representation_action
from agent import Cat, Mouse
import numpy as np
def test(mouse:Mouse,env:grid):
    mouse.epsilon = 0
    j =0
    T = env.grid()
    plt.figure()
    plt.imshow(T)
    plt.savefig(f"image/frame{j}")
    plt.close()
    while env.terminated():
        env.reset()
    n = 0
    while not env.terminated():
        s_mouse,_ = env.transition_mouse(mouse.act(env.mouse))
        T = env.grid()
        plt.figure()
        plt.imshow(T)
        plt.savefig(f"image/frame{j}")
        plt.close()
    exit()

if __name__=="__main__":
    train = True
    testmode = False
    start = 0
    epsilon = 1.0
    gamma = .99
    nx = 5
    ny = 5
    # large learning rates implies more risk to local minima
    lr_pi = 1e-4
    lr_q = 1e-4
    batch_size = 8
    buffer_size = 16
    # learn Q with K iteration, allows more stability. We choose K=1 bc the system is simple.
    K = 1
    n_epochs = 3500
    loadpath = "loads"
    loadopt = "opt"
    max_steps = 60
    fact = .999
    tau = .01




    env = grid(nx,ny,max_steps = max_steps)
    env.reset()
    mouse = Mouse(env)
    for j in range(100):
        state_mouse,_ = env.transition_mouse(np.random.randint(env.Na))
        state_cat,_ = env.transition_cat(np.random.randint(env.Na))
        print(state_mouse, state_cat)
        print(env.mouse_pos,env.cat_pos)

