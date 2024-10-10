import sys
sys.path.append("../../env")
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from policy import policy
from Vfunc import V
from env import grid
import os
from a2c import A2C, testfunc
import matplotlib.pyplot as plt

        
if __name__=="__main__":
    print("name",__name__)
    train = True
    test = True
    start = 0
    epsilon = 0.
    gamma = .99
    nx = 6
    ny = 6
    G  = 10
    batch_size = 1
    n_episodes = 1000
    loadpath = "loads"
    loadopt = "opt"

    env = grid(nx,ny,G = G) 
    p = policy(env)
    Vfunc = V(env)
    optimizerpi = optim.Adam(p.parameters(), lr = 1e-3) 
    optimizerV = optim.Adam(Vfunc.parameters(), lr = 1e-3) 
    if start>0:
        p.load_state_dict(torch.load(os.path.join(loadpath,         f"pi_load_{start}.pt"),weights_only=True))
        optimizerpi.load_state_dict(torch.load(os.path.join(loadopt,f"opt_pi_load_{start}.pt"),weights_only=True))
        Vfunc.load_state_dict(torch.load(os.path.join(loadpath,     f"q_load_{start}.pt"),weights_only=True))
        optimizerV.load_state_dict(torch.load(os.path.join(loadopt, f"opt_q_load_{start}.pt"),weights_only=True))
        print("chargement poids")
    if train:
        listLosspi = A2C(Vfunc,optimizerV,p,optimizerpi,env, n_episodes, loadpath,loadopt, epsilon = epsilon, start = start, K = 1, gamma =gamma)
    if test:
        testfunc(p, env, epsilon = epsilon, plot = True)
