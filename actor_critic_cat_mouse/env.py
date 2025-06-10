import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import numpy as np

class grid:
    """
    Envirnonement grille sur laquelle se deplace l'agent jusqu'à atteindre le point G
    """
    def __init__(self,Nx,Ny,gamma = .9,S = 3,C= 12,epsilon=0.05):
        self.actions = [(0,1), (0, -1), (1, 0), (-1, 0),(1,1),(1,-1),(-1,-1),(-1,1)]
        self.table_fromage = torch.zeros((Nx,Ny)) 
        self.fromage = torch.randn((3,2))
        for f in self.fromage:
            print(f)
            self.table_fromage[int(f[0]),int(f[1])] = 1
        self.epsilon = epsilon
        self.Na = len(self.actions)
        self.Nx = Nx
        self.Ny = Ny
        self.R = -1
        self.gamma = gamma
        self.S = S
        self.C = C
        self.states_encod = torch.eye(self.Nx*self.Ny).unsqueeze(0)
        self.actions_encod = torch.eye(self.Na).unsqueeze(0)
        self.cat = torch.randint(0,self.Nx*self.Ny,(1,)) 
        self.mouse = torch.randint(0,self.Nx*self.Ny,(1,)) 
    def reset(self):
        self.S = torch.randint(0,self.Nx*self.Ny,(1,))
        self.C = torch.randint(0,self.Nx*self.Ny,(1,))
    def transition(self,a_tab):
        self.cat = self.transition_single_agent(self.cat,a_tab[0]) 
        self.mouse = self.transition_single_agent(self.mouse,a_tab[1]) 
        reward = [self.reward_chat(self.cat,self.mouse), self.reward_souris(self.mouse,self.cat)]
        return [self.cat, self.mouse],reward 
    def transition_single_agent(self,s,a):
        assert(0<=s<self.Nx*self.Ny)
        d = self.actions[a]
        s_couple = (s//self.Ny, s%self.Ny)
        if self.Nx>s_couple[0]+ d[0]>=0 and self.Ny>s_couple[1]+d[1]>=0:
            sp = (s_couple[0]+ d[0], s_couple[1]+d[1])
            assert(0<=sp[0]*self.Ny+sp[1]<self.Nx*self.Ny)
            s_out = sp[0]*self.Ny+sp[1]
        else:
            s_out = s
        return s_out
    def reward_chat(self,s_chat,s_souris):
        reward = self.cat==self.mouse 
        return reward
    def reward_souris(self,s_souris,s_chat):
        reward = (self.cat==self.mouse)*(-10)
        #if s_out in self.fromage and self.table_fromage[s_out[0],s_out[1]]>0:
        #    reward+=5
        #self.table_fromage[s_out[0],s_out[1]]=0
        return reward
    def transition_souris(self,a,s,s_chat):
        assert(0<=s<self.Nx*self.Ny)
        d = self.actions[a]
        s_couple = (s//self.Ny, s%self.Ny)
        if self.Nx>s_couple[0]+ d[0]>=0 and self.Ny>s_couple[1]+d[1]>=0:
            sp = (s_couple[0]+ d[0], s_couple[1]+d[1])
            assert(0<=sp[0]*self.Ny+sp[1]<self.Nx*self.Ny)
            s_out = sp[0]*self.Ny+sp[1]
        else:
            s_out = s
        return s_out
    def grid(self,s_souris,s_chat):
        T = np.zeros((self.Nx,self.Ny))
        T[s_souris//self.Ny, s_souris%self.Ny] = 1
        T[s_chat//self.Ny, s_chat%self.Ny] = -1
        print(T)
    def tensor_state(self,s):
        return self.states_encod[:,:,s]
    def zero_one(self,state,J):
        x = nn.functional.one_hot(state,J)
        x = x.reshape((len(state),-1))
        x = x.type(torch.float32)
        return x
    def representation_action(self,a):
        return torch.Tensor([self.actions[i][0] for i in a]), torch.Tensor([self.actions[i][1] for i in a])
    def representation(self,s,c):
        out1 = (-1)*pad_sequence([self.states_encod[0,:,int(i)] for i in c]).permute(1,0)
        out2 =      pad_sequence([self.states_encod[0,:,int(i)] for i in s]).permute(1,0)
        return out1+out2
    def representationaction(self,action):
        return  pad_sequence([self.actions_encod[0,:,int(i)] for i in action]).permute(1,0)
