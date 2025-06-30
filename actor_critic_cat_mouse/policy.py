import torch
from torch import distributions
import torch.nn as nn
import torch.nn.functional as F
class policy(nn.Module):
    def __init__(self,env):
        super(policy,self).__init__()
        self.Nx = env.Nx
        self.Ny = env.Ny
        self.Na = env.Na
        self.linear1 = nn.Linear(2,8)
        self.linear2 = nn.Linear(8,16)
        self.linear4 = nn.Linear(16,self.Na)
        self.actv = nn.ReLU()
    def forward(self, x, logit =False):
        x = torch.mul(torch.Tensor(x),1.0/self.Nx)
        out = self.linear1(x)
        out = self.actv(out)
        out = self.linear2(out)
        out = self.actv(out)
        out = self.linear3(out)
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
    def __init__(self, env):
        super(policy2,self).__init__()
        self.Nx = env.Nx
        self.Ny = env.Ny
        self.Na = env.Na
        self.linear1 = nn.Linear(2,8)
        self.linear2 = nn.Linear(8,16)
        self.linear4 = nn.Linear(16,self.Na)
        self.actv = nn.ReLU()
    def forward(self,s1, logit =False):
        x = torch.mul(torch.Tensor(s1),1.0/self.Nx)
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
