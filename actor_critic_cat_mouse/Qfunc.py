import torch.nn as nn
import torch
class Q(nn.Module):
    def __init__(self,Nx,Ny,Na):
        super(Q,self).__init__()
        self.Nx = Nx
        self.Ny = Ny
        self.Na = Na
        self.linear1 = nn.Linear(2*self.Ny*self.Nx + 2*Na,64)
        self.linear2 = nn.Linear(64,64)
        self.linear3 = nn.Linear(64,64)
        self.linear4 = nn.Linear(64,32)
        self.linear5 = nn.Linear(32,1)
        self.actv = nn.Tanh()
        self.actv2 = nn.Sigmoid()
    def forward(self, s0,s1,a0,a1):
        x = torch.cat((s0,s1,a0,a1),dim=1)
        out = self.linear1(x)
        out = self.actv(out)
        out = self.linear2(out)
        out = self.actv(out)
        out = self.linear3(out)
        out = self.actv(out)
        out = self.linear4(out)
        out = self.actv(out)
        out = self.linear5(out)
        return out
