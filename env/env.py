import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import numpy as np

class grid:
    """
    Envirnonement grille sur laquelle se deplace l'agent jusqu'à atteindre le point self.G
    """
    def __init__(self,Nx,Ny, G = 50,gamma = .9,S = 3,epsilon=0.05, obstacles_encod = torch.Tensor([])):
        assert(0<=G<Nx*Ny)
        self.actions = [(0,1), (0, -1), (1, 0), (-1, 0),
                        (1,1),(-1,1),(1,-1),(-1,-1)]
        self.epsilon = epsilon
        self.Na = len(self.actions)
        self.Nx = Nx
        self.Ny = Ny
        self.R = -1
        self.G = G
        self.gamma = gamma
        #self.S = S
        self.states_encod = torch.eye(self.Nx*self.Ny).unsqueeze(0)
        self.obstacles_encod = obstacles_encod
        oo = [j for j in range(self.Nx*self.Ny) if not (j in self.obstacles_encod)]
        self.S = oo[np.random.randint(0,len(oo))]
        #placer obstacles

        self.actions_encod = torch.eye(self.Na).unsqueeze(0)
    def transition(self,a,s):
        assert(0<=s<self.Nx*self.Ny)
        d = self.actions[a]
        s_couple = (s//self.Ny, s%self.Ny)

        sp = (s_couple[0]+ d[0], s_couple[1]+d[1])
        s_temp = sp[0]*self.Ny+sp[1]  


        #if self.Nx>s_couple[0]+ d[0]>=0 and self.Ny>s_couple[1]+d[1]>=0:
        condition = self.Nx>s_couple[0]+ d[0]>=0 and self.Ny>s_couple[1]+d[1]>=0# and not(s_couple[0]+ d[0] in self.obstacles_encod 
        condition = self.Nx>s_couple[0]+ d[0]>=0 and self.Ny>s_couple[1]+d[1]>=0 and not(s_temp in self.obstacles_encod)
        if condition:
            assert(0<=sp[0]*self.Ny+sp[1]<self.Nx*self.Ny)
            s = s_temp
            R = torch.Tensor([(s==self.G)*1.0])
        else:
            R=torch.Tensor([-1])# la recompense est -1 si l'agent essai de sortir de la grille
            #R = -1
        return s,R
    def grid(self,s):
        assert(type(s)==int)
        assert(0<=s<=self.Nx*self.Ny)
        T = torch.zeros((self.Nx,self.Ny))
        T[self.G//self.Ny, self.G%self.Ny] = 6
        T[s//self.Ny, s%self.Ny] = 11
        for p in self.obstacles_encod:
            #print("p",p)
            T[p//self.Ny, p%self.Ny] = -1
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
    def transitionvec(self,a,s):
        "a un est un iterable de valeurs scalaires"
        "s un est un iterable de valeurs scalaires"
        couples = {0:s//self.Ny,1:s%self.Ny}
        mouv1,mouv2 = self.representation_action(a)
        A =(couples[0]+mouv1>=0)*(couples[0]+mouv1<self.Nx)*(couples[1]+mouv2>=0)*(couples[1]+mouv2<self.Ny)
        couples2 = {0:(couples[0]+mouv1)*A,
                    1:(couples[1]+mouv2)*A
                    }
        newstate = couples2[0]*self.Ny+couples2[1]
        reward = (newstate==self.G) +(A*(-1)+1)*(-10)
        #a l'interieur 1 dans A et 0 dans A*(-1)+1 --> 0
        #a l'exterieur 0 dans A et 1 dans A*(-1)+1 --> -10
        return newstate,reward
    def representation(self,state):
       return  pad_sequence([self.states_encod[0,:,int(i)] for i in state]).permute(1,0)
    def representationaction(self,action):
       return  pad_sequence([self.actions_encod[0,:,int(i)] for i in action]).permute(1,0)
