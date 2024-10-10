import torch.nn as nn
import torch
class V(nn.Module):
    def __init__(self,env):
        super(V,self).__init__()
        self.Nx = env.Nx
        self.Ny = env.Ny
        self.env = env
        self.linear1 = nn.Linear(self.Ny*self.Nx,16)
        self.linear2 = nn.Linear(16,16)
        self.linear3 = nn.Linear(16,1)
        self.actv = nn.Sigmoid()

    def forward(self, s):
        x = self.env.representation(s)
        out = self.linear1(x)
        out = self.actv(out)
        out = self.linear2(out)
        out = self.actv(out)
        out = self.linear3(out)
        return out
