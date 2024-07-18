import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import sys
from policy import policy
from Qfunc import Q
from buffer import Buffer
sys.path.append("../../env")
from env import grid
import os
from a2c import A2C
import matplotlib.pyplot as plt

        
if __name__=="__main__":
    start = 0
    epsilon = 0.2
    gamma = .99
    nx = 4
    ny = 4
    G = 3
    na = 4
    env = grid(nx,ny,G = G, gamma =gamma) 
    p = policy(nx,ny,na)
    Qvalue = Q(nx,ny,na)
    optimizerpi = optim.Adam(p.parameters(), lr = 1e-4) 
    optimizerQ = optim.Adam(Qvalue.parameters(), lr = 1e-4) 
    buffer = Buffer()
    N = 200 
    batch_size = 100
    n_epochs = 100000
    loadpath = "loads"
    loadopt = "opt"

    if start>0:
        p.load_state_dict(torch.load(os.path.join(loadpath,"pi_load_{start}.pt")))
        optimizerpi.load_state_dict(torch.load(os.path.join(loadopt,"opt_pi_load_{start}.pt")))
        Qvalue.load_state_dict(torch.load(os.path.join(loadpath,"q_load_{start}.pt")))
        optimizerQ.load_state_dict(torch.load(os.path.join(loadopt,"opt_q_load_{start}.pt")))


    listLosspi = A2C(buffer,Qvalue,optimizerQ,p,optimizerpi,env,N, batch_size, n_epochs, loadpath,loadopt, epsilon, start = 0)
    """
    Il faut que la negativ pseudo loss decroisse en avec le nombre d'epochs, ce qui est bien le cas
    """
    plt.figure()
    plt.plot(listLosspi, label="Negativ pseudo loss")
    plt.legend()
    plt.savefig("loss/NegativPseudoLossPi")
    plt.shot()
