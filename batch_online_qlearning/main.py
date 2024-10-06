import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import sys
from policy import policy
from Qfunc import Q
from buffer import Buffer
from env import grid
import os
from qlearning import qlearn, test
import matplotlib.pyplot as plt
if __name__=="__main__":
    train = False
    testmode = True
    start = 10
    epsilon = 0.1
    gamma = .9
    nx = 6
    ny = 6
    G = 10
    lr = 1e-3
    n_epochs = 10
    batch_size = 10
    M = 100
    N = 100
    K = 5 #nombre iterations descente de gradient pour le meme batch (s,a,r,s')
    loadpath = "loads"
    loadopt = "opt"
    maxsize = 100
    env = grid(nx,ny,G = G) 
    Qvalue = Q(env)
    optimizerQ = optim.Adam(Qvalue.parameters(), lr = lr) 
    if start>0:
        Qvalue.load_state_dict(torch.load(os.path.join(loadpath,f"q_load_{n_epochs}_0.pt"), weights_only=True))
        optimizerQ.load_state_dict(torch.load(os.path.join(loadopt,f"opt_q_load_{n_epochs}_0.pt"), weights_only=True))
    bffer = Buffer(maxsize = maxsize)
    if train:
        qlearn(bffer,
               batch_size,
               M,
               N,
               K,
               Qvalue,
               optimizerQ,
               env,
               n_epochs,
               loadpath,
               loadopt,
               gamma = gamma
               )
    if test:
        test(Qvalue, env, epsilon = 0, plot = True)
