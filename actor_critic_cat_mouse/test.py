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
        env.wset()
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
    epsilon = .3
    gamma = .99
    nx = 4
    ny = 4
    # large learning rates implies more risk to local minima
    lr_pi = 1e-3
    lr_q = 1e-3
    batch_size = 3
    buffer_size = 600
    # learn Q with K iteration, allows more stability. We choose K=1 bc the system is simple.
    K = 2
    n_epochs = 3000
    loadpath = "loads"
    loadopt = "opt"
    max_steps = 100
    fact = .999
    tau = .0




    env = grid(nx,ny,max_steps = max_steps)
    env.reset()
    mouse = Mouse(env,epsilon = epsilon,buffer_size = buffer_size,lr_pi=lr_pi, lr_q=lr_q)
    cat = Cat(env,epsilon = epsilon, buffer_size = buffer_size, lr_pi=lr_pi, lr_q=lr_q)
    A2C(env,mouse,cat,batch_size,n_epochs)
