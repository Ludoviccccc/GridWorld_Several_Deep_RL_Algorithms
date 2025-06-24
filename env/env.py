import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn


import numpy as np
import matplotlib.pyplot as plt


class grid:
    """
    Envirnonement grille sur laquelle se deplace l'agent jusqu'à atteindre le point self.G
    """
    def __init__(self,Nx,Ny, G = 50,S = 3,epsilon=0.05, obstacles_encod = torch.Tensor([])):
        assert(0<=G<Nx*Ny)
        self.actions = [(0,1), (0, -1), (1, 0), (-1, 0),
                        (1,1),(-1,1),(1,-1),(-1,-1)]
        self.epsilon = epsilon
        self.Na = len(self.actions)
        self.Nx = Nx
        self.Ny = Ny
        self.R = -1
        self.G = G
        self.states_encod = torch.eye(self.Nx*self.Ny).unsqueeze(0)
        self.obstacles_encod = obstacles_encod
        self.actions_encod = torch.eye(self.Na).unsqueeze(0)
    def transition(self,a,s):
        assert(0<=s<self.Nx*self.Ny)
        #self.previous = s
        d = self.actions[a]
        s_couple = (s//self.Ny, s%self.Ny)
        sp = (s_couple[0]+ d[0], s_couple[1]+d[1])
        s_temp = sp[0]*self.Ny+sp[1]  
        condition = self.Nx>s_couple[0]+ d[0]>=0 and self.Ny>s_couple[1]+d[1]>=0 and not(s_temp in self.obstacles_encod)
        if condition:
            assert(0<=sp[0]*self.Ny+sp[1]<self.Nx*self.Ny)
            s = s_temp
            R = torch.Tensor([(s==self.G)*100.0])
        else:
            R=torch.Tensor([-1])# la recompense est -1 si l'agent essai de sortir de la grille
        return s,R
    def grid(self,s, name = False):
        assert(type(s)==int)
        assert(0<=s<=self.Nx*self.Ny)
        T = torch.zeros((self.Nx,self.Ny))
        T[self.G//self.Ny, self.G%self.Ny] = 6
        T[s//self.Ny, s%self.Ny] = 11
        for p in self.obstacles_encod:
            T[p//self.Ny, p%self.Ny] = -1
        print(T)
        if name:
            plt.imshow(T.numpy())
            plt.xticks([])
            plt.yticks([])
            plt.savefig(name,bbox_inches='tight')
            plt.close()
            
    def tensor_state(self,s):
        return self.states_encod[:,:,s]
    def zero_one(self,state,J):
        x = nn.functional.one_hot(state,J)
        x = x.reshape((len(state),-1))
        x = x.type(torch.float32)
        return x
    def representation_action(self,a):
        return torch.Tensor([self.actions[int(i)][0] for i in a]), torch.Tensor([self.actions[int(i)][1] for i in a])
    def representation(self,state):
       return  pad_sequence([self.states_encod[0,:,int(i)] for i in state]).permute(1,0)
    def representationaction(self,action):
       return  pad_sequence([self.actions_encod[0,:,int(i)] for i in action]).permute(1,0)
