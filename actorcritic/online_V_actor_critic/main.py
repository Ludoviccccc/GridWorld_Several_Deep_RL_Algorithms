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
from ac import AC, testfunc
import matplotlib.pyplot as plt
import json

        
if __name__=="__main__":
    with open("arg.json","r") as f:
        data = json.load(f)
    train = data["train"]
    test = data["test"]
    start = data["start"]
    epsilon = data["epsilon"]
    gamma = data["gamma"]
    nx = data["nx"]
    ny = data["ny"]
    G = data["G"]
    n_episodes = data["n_episodes"]
    loadpath = data["loadpath"]
    loadopt = data["loadopt"]
    graph = data["graph"]
    ob = data["ob"]
    lr = data["lr"]

    print("ob", ob)
    env = grid(nx,ny,G = G, obstacles_encod = ob) 
    p = policy(env).cuda()
    Vfunc = V(env).cuda()
    optimizerpi = optim.Adam(p.parameters(), lr = lr) 
    optimizerV = optim.Adam(Vfunc.parameters(), lr = lr) 
    if start>0:
        p.load_state_dict(torch.load(os.path.join(loadpath,         f"pi_load_{start}.pt"),weights_only=True))
        optimizerpi.load_state_dict(torch.load(os.path.join(loadopt,f"opt_pi_load_{start}.pt"),weights_only=True))
        Vfunc.load_state_dict(torch.load(os.path.join(loadpath,     f"q_load_{start}.pt"),weights_only=True))
        optimizerV.load_state_dict(torch.load(os.path.join(loadopt, f"opt_q_load_{start}.pt"),weights_only=True))
        print("chargement poids")
    if train:
        listLosspi = AC(Vfunc,optimizerV,p,optimizerpi,env, n_episodes, loadpath,loadopt, start = start, K = 1, gamma =gamma)
    if test:
        testfunc(p, env, epsilon = epsilon, plot = True, graph = graph)
