import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import sys
from policy import policy
from Qfunc import Q
from buffer import Buffer
#sys.path.append("../env")
from env import grid
import os
from qlearning import qlearn, test
import matplotlib.pyplot as plt

        
if __name__=="__main__":
    train = True
    testmode = False
    start =0
    epsilon = 0.0
    gamma = .9
    nx = 5
    ny = 5
    G = 10
    na = 4
    env = grid(nx,ny,G = G, gamma =gamma) 

    Qvalue_chat = Q(nx,ny,na)
    Qvalue_souris = Q(nx,ny,na)
    lr = 1e-3
    optimizerQ_chat = optim.Adam(Qvalue_chat.parameters(), lr = lr) 
    optimizerQ_souris = optim.Adam(Qvalue_souris.parameters(), lr = lr) 
    N = 20 
    batch_size = 10
    n_epochs = 100000

    loadpath = "loads"
    loadopt = "opt"



    #print("representation",env.representation())
    #env.transition_cat(2)
    #r_chat   = env.reward_cat()
    #r_souris = env.reward_souris()
    #print("representation",env.representation())

    if start>0:
        Qvalue_chat.load_state_dict(torch.load(os.path.join(loadpath,f"q_load_{start}.pt")))
        Qvalue_souris.load_state_dict(torch.load(os.path.join(loadpath,f"q_load_{start}.pt")))
        optimizerQ_chat.load_state_dict(torch.load(os.path.join(loadopt,f"opt_q_load_{start}.pt")))
        optimizerQ_souris.load_state_dict(torch.load(os.path.join(loadopt,f"opt_q_load_{start}.pt")))

    if train:
        listLosspi = qlearn(Qvalue_chat,Qvalue_souris,optimizerQ_chat,optimizerQ_souris,env, n_epochs, loadpath,loadopt,epsilon,start=0)
        qlearn(Qvalue,optimizerQ,env, n_epochs, loadpath,loadopt, epsilon, start = 0)
        """
        Il faut que la negativ pseudo loss decroisse en avec le nombre d'epochs, ce qui est bien le cas
        """
        plt.figure()
        plt.plot(listLosspi, label="Negativ pseudo loss")
        plt.legend()
        plt.savefig("loss/NegativPseudoLossPi")
        plt.show()

    if testmode:
        iterations = test(Qvalue, env, epsilon, plot = True)
        print("nombre d'iterations", iterations)
