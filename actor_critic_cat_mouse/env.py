import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import numpy as np

class grid:
    """
    Envirnonement grille sur laquelle se deplace l'agent jusqu'à atteindre le point G
    """
    def __init__(self,Nx,Ny,max_steps = 100):
        self.actions = [(0,1), (0, -1), (1, 0), (-1, 0),(1,1),(1,-1),(-1,-1),(-1,1)]
        self.table_fromage = torch.zeros((Nx,Ny)) 
        self.fromage = torch.randn((3,2))
        for f in self.fromage:
            self.table_fromage[int(f[0]),int(f[1])] = 1
        self.Na = len(self.actions)
        self.Nx = Nx
        self.Ny = Ny
        self.states_encod = torch.eye(self.Nx*self.Ny).unsqueeze(0)
        self.actions_encod = torch.eye(self.Na).unsqueeze(0)
        self.cat = torch.randint(0,self.Nx*self.Ny,(1,)) 
        self.mouse = torch.randint(0,self.Nx*self.Ny,(1,)) 
        self.target_idx = 10
        self.target_mouse = (self.target_idx//self.Ny,self.target_idx%self.Ny)
        self.max_steps = max_steps
    def reset(self):
        self.count = 0
        self.cat = torch.randint(0,self.Nx*self.Ny,(1,)) 
        self.mouse = torch.randint(0,self.Nx*self.Ny,(1,)) 
    def transition_cat(self,a:int):
        self.cat_previous = self.cat
        self.cat = self.transition_single_agent(self.cat,a) 
        reward = self.reward_cat()
        terminated = self.terminated()
        return self.cat,reward
    def transition_mouse(self,a:int):
        self.mouse_previous = self.mouse
        self.mouse = self.transition_single_agent(self.mouse,a) 
        reward = self.reward_mouse()
        terminated = self.terminated()
        self.count+=1
        return self.mouse,reward
    def terminated(self):
        "says if the episode is teminated"
        s_mouse = (self.mouse//self.Ny, self.mouse%self.Ny)
        cheese_reached = (s_mouse[0] ==self.target_mouse[0])*(s_mouse[1]==self.target_mouse[1])
        terminated = cheese_reached
        return terminated
    def truncated(self):
        return self.count>self.max_steps
    def transition_single_agent(self,s,a):
        assert(0<=a<=self.Na)
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
    def reward_cat(self):
        s_cat = (self.cat//self.Ny, self.cat%self.Ny)
        s_mouse = (self.mouse//self.Ny, self.mouse%self.Ny)
        reward = (self.cat==self.mouse) *(100.0) + (-10.0)*(self.cat ==self.cat_previous) 
        return reward
    def reward_mouse(self):
        reward = (100.0)*(self.target_idx==self.mouse) + (-10.0)*(self.mouse==self.mouse_previous)
        return reward
    def grid(self):
        s_mouse = self.mouse
        s_cat = self.cat
        T = np.zeros((self.Nx,self.Ny))
        T[s_mouse//self.Ny, s_mouse%self.Ny] = 1
        T[s_cat//self.Ny, s_cat%self.Ny] = -1
        T[self.target_mouse[0],self.target_mouse[1]] = 5
        print(T)
        return T
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
