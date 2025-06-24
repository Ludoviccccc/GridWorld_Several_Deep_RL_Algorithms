import sys
sys.path.append("../../env")
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from policy import policy
from Qfunc import Q
from buffer import Buffer
from env import grid
import os
from a2c import A2C, testfunc
import matplotlib.pyplot as plt

        
if __name__=="__main__":
    train = False
    test = True
    start = 3000
    epsilon = 1.0
    gamma = .99
    nx = 6
    ny = 6
    G = 10
    N = 3 
    batch_size = 128
    maxsize = 1024
    K = 5
    n_epochs = 3000
    loadpath = "loads"
    loadopt = "opt"
    lr = 1e-4

    env = grid(nx,ny,G = G) 
    p = policy(env)
    Qvalue = Q(env)
    optimizerpi = optim.Adam(p.parameters(), lr = lr) 
    optimizerQ = optim.Adam(Qvalue.parameters(), lr = lr) 
    buffer = Buffer(maxsize)

    if start>0:
        p.load_state_dict(torch.load(os.path.join(loadpath,         f"pi_load_{start}.pt"),weights_only=True))
        optimizerpi.load_state_dict(torch.load(os.path.join(loadopt,f"opt_pi_load_{start}.pt"),weights_only=True))
        Qvalue.load_state_dict(torch.load(os.path.join(loadpath,    f"q_load_{start}.pt"),weights_only=True))
        optimizerQ.load_state_dict(torch.load(os.path.join(loadopt, f"opt_q_load_{start}.pt"),weights_only=True))
    if train:
        listLosspi,N = A2C(buffer,Qvalue,optimizerQ,p,optimizerpi,env,N, batch_size, n_epochs, loadpath,loadopt, epsilon, start = 0, gamma = gamma, K = K)
    if test:
        testfunc(p, env, epsilon = 0, plot = True)
