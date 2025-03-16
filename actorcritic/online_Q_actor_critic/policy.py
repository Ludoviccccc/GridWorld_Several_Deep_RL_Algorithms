import torch
from torch import distributions
import torch.nn as nn
import torch.nn.functional as F
class policy(nn.Module):
    def __init__(self, env):
        super(policy,self).__init__()
        self.Nx = env.Nx
        self.Ny = env.Ny
        self.Na = env.Na
        self.env = env
        self.linear1 = nn.Linear(self.Ny*self.Nx,32)
        self.linear2 = nn.Linear(32,16)
        self.linear3 = nn.Linear(16,self.Na)
        self.actv = nn.ReLU()
    def forward(self, x, logit =False):
        out = self.linear1(self.env.representation(x))
        out = self.actv(out)
        out = self.linear2(out)
        out = self.actv(out)
        logits  = self.linear3(out)
        dist = distributions.Categorical(F.softmax(logits,dim=1))  
        #action  = dist.sample([1]).reshape((1,))#.squeeze()
        action  = dist.sample([1]).flatten()
        print("action", action)
        if logit:
            out =  action, logits
        else:
            out = action
        return out
