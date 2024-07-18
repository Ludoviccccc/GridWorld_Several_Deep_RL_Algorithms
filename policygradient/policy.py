import torch
import torch.nn as nn
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
    def forward(self, x):
        out = self.linear1(x)
        out = self.actv(out)
        out = self.linear2(out)
        out = self.actv(out)
        out = self.linear3(out)
        return out
