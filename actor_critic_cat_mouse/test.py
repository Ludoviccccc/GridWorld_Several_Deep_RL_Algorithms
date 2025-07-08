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
def test(mouse:Mouse,cat:Mouse,env:grid):
    j =0
    plt.close()
    while env.terminated():
        env.reset()
    T = env.grid()
    plt.figure()
    plt.imshow(T)
    plt.savefig(f"image/frame{j}")
    n = 0
    while not env.terminated():
        j+=1
        s_tab = {"cat":env.state_cat(),"mouse": env.state_mouse()}
        action = {"cat":cat(s_tab),"mouse":mouse([s_tab["mouse"]])}
        env.transition(action)
        T = env.grid()
        plt.figure()
        plt.imshow(T)
        plt.savefig(f"image/frame{j}")
        plt.close()

if __name__=="__main__":
    testmode = False
    start = 0
    epsilon = .3
    gamma = .99
    nx = 4
    ny = 5
    # large learning rates implies more risk to local minima
    mouse_lr_pi = 1e-3
    mouse_lr_q = 1e-3
    cat_lr_pi = 1e-3
    cat_lr_q = 1e-3
    batch_size = 32
    buffer_size = 600
    # learn Q with K iteration, allows more stability. We choose K=1 bc the system is simple.
    K = 2
    K_cat = 2
    n_epochs =  2000
    loadpath = "loads"
    loadopt = "opt"
    max_steps = 30
    fact = .95
    tau = .01
    min_eps = .1

    env = grid(nx,ny,max_steps = max_steps)
    env.reset()
    mouse = Mouse(env,epsilon = epsilon,buffer_size = buffer_size,lr_pi=mouse_lr_pi, lr_q=mouse_lr_q, tau=tau,K = K)
    cat = Cat(env,epsilon = epsilon, buffer_size = buffer_size, lr_pi=cat_lr_pi, lr_q=cat_lr_q,tau = tau,K = K_cat)
    if start>0:
        mouse.epsilon = .1
        cat.epsilon = .1
        mouse.load(start)
        cat.load(start)
    if testmode:
        test(mouse,cat,env)
    else:
        A2C(env,mouse,cat,batch_size,n_epochs,fact = fact,min_eps=min_eps)
