import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from policy import policy
from Qfunc import Q
from buffer import Buffer
from env import grid
import os
from qlearning import qlearn, test
import matplotlib.pyplot as plt

        
if __name__=="__main__":
    train = True
    testmode = True
    start =0
    epsilon = 0.1
    gamma = .9
    nx = 5
    ny = 5
    lr = 1e-2
    env = grid(nx,ny, gamma =gamma) 

    Qvalue_chat = Q(nx,ny,env.Na)
    Qvalue_souris = Q(nx,ny,env.Na)
    optimizerQ_chat = optim.Adam(Qvalue_chat.parameters(), lr = lr) 
    optimizerQ_souris = optim.Adam(Qvalue_souris.parameters(), lr = lr) 
    N = 20 
    n_epochs = 100000

    loadpath = "loads"
    loadopt = "opt"


    if start>0:
        Qvalue_chat.load_state_dict(torch.load(os.path.join(loadpath,f"q_chat_load_{start}.pt")))
        Qvalue_souris.load_state_dict(torch.load(os.path.join(loadpath,f"q_souris_load_{start}.pt")))
        optimizerQ_chat.load_state_dict(torch.load(os.path.join(loadopt,f"opt_q_chat_load_{start}.pt")))
        optimizerQ_souris.load_state_dict(torch.load(os.path.join(loadopt,f"opt_q_souris_load_{start}.pt")))

    if train:
        listLosspi = qlearn(Qvalue_chat,Qvalue_souris,optimizerQ_chat,optimizerQ_souris,env, n_epochs, loadpath,loadopt,epsilon,start=0)
        qlearn(Qvalue,optimizerQ,env, n_epochs, loadpath,loadopt, epsilon, start = start)
        plt.figure()
        plt.plot(listLosspi, label="Negativ pseudo loss")
        plt.legend()
        plt.savefig("loss/NegativPseudoLossPi")
        plt.show()

    if testmode:
        iterations =test(Qvalue_chat,Qvalue_souris, env, epsilon, plot = True)
        print("nombre d'iterations", iterations)
