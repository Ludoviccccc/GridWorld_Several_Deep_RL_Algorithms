import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from policy import policy
from Qfunc import Q
from buffer import Buffer
from env import grid
import os
#from qlearning import qlearn, test
import matplotlib.pyplot as plt
from a2c import A2C
from rep import Representation, Representation_action
def test(q_tab:list[Q], pi_tab,R, env,buffer):
    #s_tab = [torch.randint(0,env.Nx*env.Ny,(1,)) for j in range(2)]
    s_tab = [env.cat, env.mouse]
    j =0
    T = env.grid()
    plt.imshow(T)
    plt.savefig(f"image/frame{j}")
    while s_tab[0]!=s_tab[1]:
        a_tab = []
        for k in range(2):
            rep0 = R.states_encod[s_tab[0]]
            rep1 = R.states_encod[s_tab[1]]
            a_tab.append(pi_tab[k](rep0,rep1))
        s_tab_prim,reward_tab = env.transition(a_tab)
        buffer.store({"state":[s_tab],"action":[a_tab],"new_state":[s_tab_prim],"reward":[reward_tab]})
        T = env.grid()
        s_tab = s_tab_prim
        j+=1
        plt.imshow(T)
        plt.savefig(f"image/frame{j}")
    exit()
if __name__=="__main__":
    train = False
    testmode = True
    start = 2000
    epsilon = .2
    gamma = .97
    nx = 5
    ny = 5
    lr = 1e-3
    N = 10
    batch_size = 32
    n_epochs = 5001
    loadpath = "loads"
    loadopt = "opt"
    take_load = False
    env = grid(nx,ny, gamma =gamma)
    buffer = Buffer()
    q_tab = [Q(nx,ny,env.Na) for j in range(2)]
    pi_tab = [policy(nx,ny,env.Na) for j in range(2)]
    optimizerQ_tab = [optim.Adam(q_tab[j].parameters(), lr = lr) for j in range(2)]
    optimizerPi_tab = [optim.Adam(pi_tab[j].parameters(),lr=lr) for j in range(2)]
    R = Representation(env.Nx, env.Ny)
    R_a = Representation_action(env.Na)
    if testmode:
        if take_load:
            q_tab[0].load_state_dict(torch.load(os.path.join(loadpath,f"q_0_load_{start}.pt"),weights_only=False))
            q_tab[1].load_state_dict(torch.load(os.path.join(loadpath,f"q_1_load_{start}.pt"),weights_only=False))
            pi_tab[0].load_state_dict(torch.load(os.path.join(loadpath,f"pi_0_load_{start}.pt"),weights_only=False))
            pi_tab[1].load_state_dict(torch.load(os.path.join(loadpath,f"pi_1_load_{start}.pt"),weights_only=False))
            #q_tab[0].load_state_dict(torch.load(os.path.join(loadopt,f"opt_0_q_load_{start}.pt")))
            #q_tab[1].load_state_dict(torch.load(os.path.join(loadopt,f"opt_1_q_load_{start}.pt")))
        test(q_tab, pi_tab,R, env,buffer)
    if train:
        tup = A2C(buffer, R,R_a,q_tab,optimizerQ_tab,pi_tab,optimizerPi_tab,env,N,batch_size,n_epochs,loadpath, loadopt,K=1)
