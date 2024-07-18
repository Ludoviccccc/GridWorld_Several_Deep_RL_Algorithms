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
        self.linear1 = nn.Linear(self.Ny*self.Nx,64)
        self.linear2 = nn.Linear(64,32)
        self.linear3 = nn.Linear(32,self.Na)
        self.actv = nn.ReLU()
    def forward(self, x, logit =False):
        out = self.linear1(x)
        out = self.actv(out)
        out = self.linear2(out)
        out = self.actv(out)
        out = self.linear3(out)
        if not logit:
            dist = distributions.Categorical(F.softmax(out,dim=1))  
            out = dist.sample([1]).squeeze()
        return out
