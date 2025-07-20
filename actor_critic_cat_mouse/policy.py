import torch
from torch import distributions
import torch.nn as nn
import torch.nn.functional as F
from rep import Representation
class policy(nn.Module):
    def __init__(self, env):
        super(policy,self).__init__()
        self.Nx = env.Nx
        self.Ny = env.Ny
        self.Na = env.Na
        self.linear1 = nn.Linear(4,64)
        self.linear2 = nn.Linear(64,64)
        self.linear3 = nn.Linear(64,64)
        self.linear4 = nn.Linear(64,self.Na)
        self.actv = nn.ReLU()
        self.rep_cl = Representation(self.Nx,self.Ny)
    def forward(self,s1,s2, logit =False):
        x = torch.Tensor([[s1[j]//self.Ny,s1[j]%self.Ny,s2[j]//self.Ny,s2[j]%self.Ny] for j in range(len(s1))])
        x = torch.mul(x,1.0/self.Nx)
        #x = torch.cat((s1//self.Ny,s1%self.Ny,s2//self.Ny,s2%self.Ny),dim=1)
        #print("x", x.shape)

        #x = self.rep_cl(s1)
        out = self.linear1(x)
        out = self.actv(out)
        out = self.linear2(out)
        out = self.actv(out)
        out = self.linear3(out)
        out = self.actv(out)
        logits = self.linear4(out)
        dist = distributions.Categorical(F.softmax(logits,dim=1))  
        #print(F.softmax(logits,dim=1))
        action  = dist.sample([1]).squeeze()
        if logit:
            out =  action, logits
        else:
            out = action
        return out
