import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import sys
from policy import policy
from Qfunc import Q
sys.path.append("../env")
from env import grid
import os
from qlearning import qlearn, test
import matplotlib.pyplot as plt

        
if __name__=="__main__":
    train = False
    testmode = True
    start = 5000
    epsilon = 0.1
    gamma = .9
    nx = 5
    ny = 5
    G = 10
    ob = [ 6, 19, 18,  8]
    lr = 1e-3
    n_episodes = 10000
    loadpath = "loads"
    loadopt = "opt"

    env = grid(nx,ny,G = G, obstacles_encod = ob) 
    Qvalue = Q(env)
    optimizerQ = optim.Adam(Qvalue.parameters(), lr = lr) 
    if start>0:
        Qvalue.load_state_dict(torch.load(os.path.join(loadpath,f"q_load_{start}.pt") ,weights_only = True))
        optimizerQ.load_state_dict(torch.load(os.path.join(loadopt,f"opt_q_load_{start}.pt") ,weights_only = True))
    if train:
        qlearn(Qvalue,optimizerQ,env, n_episodes, loadpath,loadopt,gamma =gamma, epsilon = epsilon, start = 0)
    if testmode:
        iterations = test(Qvalue, env, epsilon = 0, plot = True)
        print("nombre d'iterations", iterations)
