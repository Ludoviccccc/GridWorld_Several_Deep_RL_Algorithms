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
    mouse_lr_pi = 1e-3
    mouse_lr_q = 1e-3
    cat_lr_pi = 1e-5
    cat_lr_q = 1e-5
    batch_size = 32
    buffer_size = 600
    # learn Q with K iteration, allows more stability. We choose K=1 bc the system is simple.
    K = 2
    n_epochs = 200
    loadpath = "loads"
    loadopt = "opt"
    max_steps = 100
    fact = .999
    tau = .01




    env = grid(nx,ny,max_steps = max_steps)
    env.reset()
    mouse = Mouse(env,epsilon = epsilon,buffer_size = buffer_size,lr_pi=mouse_lr_pi, lr_q=mouse_lr_q, tau=tau)
    cat = Cat(env,epsilon = 0.1, buffer_size = buffer_size, lr_pi=cat_lr_pi, lr_q=cat_lr_q,tau = tau)
    A2C(env,mouse,cat,batch_size,n_epochs,fact = fact,K=K)
