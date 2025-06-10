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

class Representation:
    def __init__(self,Nx,Ny):
        self.Nx = Nx
        self.Ny = Ny
        self.states_encod = torch.eye(self.Nx*self.Ny)
    def representation(self,s,c):
        out1 = (-1)*pad_sequence([self.states_encod[0,:,int(i)] for i in c]).permute(1,0)
        out2 =      pad_sequence([self.states_encod[0,:,int(i)] for i in s]).permute(1,0)
        return torch.cat((out1,out2))
def test(q_tab, pi_tab,R, env,buffer):
    s_tab = [torch.randint(0,env.Nx*env.Ny,(1,)) for j in range(2)]
    while s_tab[0]!=s_tab[1]:
        a_tab = []
        for k in range(2):
            rep = R.states_encod[s_tab[k]]
            a_tab.append(pi_tab[k](rep))
        s_tab_prim,reward_tab = env.transition(a_tab)
        buffer.store({"state":[s_tab],"action":[a_tab],"new_state":[s_tab_prim],"reward":[reward_tab]})
        s_tab = s_tab_prim
if __name__=="__main__":
    train = False
    testmode = True
    start =0
    epsilon = 0.1
    gamma = .9
    nx = 5
    ny = 5
    lr = 1e-2
    env = grid(nx,ny, gamma =gamma)
    buffer = Buffer()
    q_tab = [Q(nx,ny,env.Na) for j in range(2)]
    pi_tab = [policy(nx,ny,env.Na) for j in range(2)]
    optimizerQ_tab = [optim.Adam(q_tab[j].parameters(), lr = lr) for j in range(2)]
    R = Representation(env.Nx, env.Ny)
    test(q_tab, pi_tab,R, env,buffer)
