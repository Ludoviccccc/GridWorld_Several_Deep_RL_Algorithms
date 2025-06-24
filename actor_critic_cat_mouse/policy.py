import torch
from torch import distributions
import torch.nn as nn
import torch.nn.functional as F
class policy(nn.Module):
    def __init__(self, Nx,Ny, Na):
        super(policy,self).__init__()
        self.Nx = Nx
        self.Ny = Ny
        self.Na = Na
        self.linear1 = nn.Linear(2*self.Ny*self.Nx,64)
        self.linear2 = nn.Linear(64,64)
        self.linear4 = nn.Linear(64,self.Na)
        self.actv = nn.Sigmoid()
    def forward(self, s0,s1, logit =False):
        x = torch.cat((s0,s1), dim=1)
        out = self.linear1(x)
        out = self.actv(out)
        out = self.linear2(out)
        out = self.actv(out)
        logits = self.linear4(out)
        dist = distributions.Categorical(F.softmax(logits,dim=1))  
        action  = dist.sample([1]).squeeze()
        if logit:
            out =  action, logits
        else:
            out = action
        return out
class policy2(nn.Module):
    def __init__(self, Nx,Ny, Na):
        super(policy2,self).__init__()
        self.Nx = Nx
        self.Ny = Ny
        self.Na = Na
        self.linear1 = nn.Linear(self.Ny*self.Nx,32)
        self.linear2 = nn.Linear(32,16)
        self.linear4 = nn.Linear(16,self.Na)
        self.actv = nn.Sigmoid()
    def forward(self, s0,s1, logit =False):
        x = s1
        out = self.linear1(x)
        out = self.actv(out)
        out = self.linear2(out)
        out = self.actv(out)
        logits = self.linear4(out)
        #print("logits", logits)
        dist = distributions.Categorical(F.softmax(logits,dim=1))  
        action  = dist.sample([1]).squeeze()
        if logit:
            out =  action, logits
        else:
            out = action
        return out
