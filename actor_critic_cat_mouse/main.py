import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from policy import policy, policy2
from Qfunc import Q,Q2
from env import grid
import os
import matplotlib.pyplot as plt
from a22c import A2C
from rep import Representation, Representation_action
from agent import Cat, Mouse
def test(mouse):
    mouse.epsilon = 0
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
    testmode = False
    start = 0
    epsilon = 0.1
    gamma = .99
    nx = 5
    ny = 5
    lr_pi = 1e-4
    lr_q = 1e-4
    batch_size = 128
    K = 5
    n_epochs = 3500
    loadpath = "loads"
    loadopt = "opt"
    env = grid(nx,ny, gamma =gamma)

    mouse = Mouse(env,lr_pi,lr_q,epsilon)
    cat = Cat(env,lr_pi,lr_q,epsilon)
    if start>0:
        mouse.load(start)
        cat.load(start)
    if testmode:
        test(q_tab, pi_tab,R, env,buffer)
    if train:
        print("train")
        tup = A2C(env,mouse,cat,batch_size,n_epochs, epsilon = epsilon, gamma=gamma)
