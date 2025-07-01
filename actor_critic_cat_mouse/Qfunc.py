import torch.nn as nn
import torch
class Q(nn.Module):
    def __init__(self,env):
        super(Q,self).__init__()
        self.Na = env.Na
        self.Nx = env.Nx
        self.Ny = env.Ny
        self.linear1 = nn.Linear(2*self.Nx*self.Ny + 2*self.Na,16)
        self.linear2 = nn.Linear(16,16)
        self.linear3 = nn.Linear(16,8)
        self.linear4 = nn.Linear(8,1)
        self.actv = nn.ReLU()
    def forward(self, s_cat,s_mouse,a_cat,a_mouse):
        x = torch.cat((s_cat,s_mouse,a_cat,a_mouse),dim=1)
        out = self.linear1(x)
        out = self.actv(out)
        out = self.linear2(out)
        out = self.actv(out)
        out = self.linear3(out)
        out = self.actv(out)
        out = self.linear4(out)
        return out
class Q2(nn.Module):
    def __init__(self,env):
        super(Q2,self).__init__()
        self.Na = env.Na
        self.Ny = env.Ny
        self.Nx = env.Nx
        self.linear1 = nn.Linear(self.Nx*self.Ny + env.Na,16)
        self.linear2 = nn.Linear(16,16)
        self.linear3 = nn.Linear(16,1)
        self.actv = nn.ReLU()
    def forward(self, s1,a1):
        x = torch.cat((s1,a1),dim=1)
        out = self.linear1(x)
        out = self.actv(out)
        out = self.linear2(out)
        out = self.actv(out)
        out = self.linear3(out)
        return out
