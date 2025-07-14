import torch.nn as nn
import torch
class Q(nn.Module):
    def __init__(self,env):
        super(Q,self).__init__()
        self.Na = env.Na
        self.Nx = env.Nx
        self.Ny = env.Ny
        self.linear1 = nn.Linear(2+2*env.Na,16)
        self.linear2 = nn.Linear(16,16)
        self.linear3 = nn.Linear(16,16)
        self.linear4 = nn.Linear(16,16)
        self.linear5 = nn.Linear(16,1)
        self.actv = nn.ReLU()
    def forward(self, s_cat,s_mouse,a_cat,a_mouse):
        s = torch.Tensor([[s_cat[j]//self.Ny - s_mouse[j]//self.Ny,s_cat[j]%self.Ny-s_mouse[j]%self.Ny] for j in range(len(s_cat))])
        x = torch.cat((s,a_cat,a_mouse),dim=1)
        out = self.linear1(x)
        out = self.linear2(out)
        out = self.actv(out)
        out = self.linear3(out)
        out = self.actv(out)
        out = self.linear4(out)
        out = self.actv(out)
        out = self.linear5(out)
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
