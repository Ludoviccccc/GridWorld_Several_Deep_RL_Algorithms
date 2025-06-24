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
    s_tab = [env.cat, env.mouse]
    j =0
    T = env.grid()
    plt.imshow(T)
    plt.savefig(f"image/frame{j}")
    terminated = env.cat==env.mouse
    print("terminated", terminated)
    while not terminated:
        a_tab = []
        for k in range(2):
            rep0 = R.states_encod[s_tab[0]]
            rep1 = R.states_encod[s_tab[1]]
            
            a_tab.append(pi_tab[k](rep0,rep1))
        print("a_tab", a_tab)
        s_tab_prim,reward_tab, terminated = env.transition(a_tab)
        buffer.store({"state":[s_tab],"action":[a_tab],"new_state":[s_tab_prim],"reward":[reward_tab]})
        T = env.grid()
        s_tab = s_tab_prim
        j+=1
        plt.imshow(T)
        plt.savefig(f"image/frame{j}")
    exit()
if __name__=="__main__":
    train = True
    testmode = True
    start = 500
    epsilon = 1.0
    gamma = .99
    nx = 6
    ny = 5
    lr = 1e-4
    N = 10
    batch_size = 8
    K = 5
    n_epochs = 500
    loadpath = "loads"
    loadopt = "opt"
    env = grid(nx,ny, gamma =gamma)
    buffer = Buffer()
    Q_cat = Q(nx,ny,env.Na)
    p_cat = policy(nx,ny,env.Na)
    Q_mouse = Q(nx,ny,env.Na)
    p_mouse = policy(nx,ny,env.Na)
    q_tab = [Q_cat,Q_mouse]
    pi_tab = [p_cat,p_mouse]
    optimizerQ_tab = [optim.Adam(Q_cat.parameters(), lr = lr), optim.Adam(Q_mouse.parameters(), lr = lr)]
    optimizerPi_tab = [optim.Adam(p_cat.parameters(),lr=lr),optim.Adam(p_mouse.parameters(),lr=lr)]
    R = Representation(env.Nx, env.Ny)
    R_a = Representation_action(env.Na)
    if start>0:
        q_tab[0].load_state_dict(torch.load(os.path.join(loadpath,f"q_0_load_{start}.pt"),weights_only=True))
        q_tab[1].load_state_dict(torch.load(os.path.join(loadpath,f"q_1_load_{start}.pt"),weights_only=True))
        pi_tab[0].load_state_dict(torch.load(os.path.join(loadpath,f"pi_0_load_{start}.pt"),weights_only=True))
        pi_tab[1].load_state_dict(torch.load(os.path.join(loadpath,f"pi_1_load_{start}.pt"),weights_only=True))
    if testmode:
        test(q_tab, pi_tab,R, env,buffer)
    if train:
        tup = A2C(buffer, R,R_a,q_tab,optimizerQ_tab,pi_tab,optimizerPi_tab,env,N,batch_size,n_epochs,loadpath, loadopt,K=K, gamma=gamma, epsilon = epsilon)
