import torch.nn as nn
import torch
from rep import Representation_action
class Q(nn.Module):
    def __init__(self,env):
        super(Q,self).__init__()
        self.Na = env.Na
        self.Nx = env.Nx
        self.Ny = env.Ny
        self.linear1 = nn.Linear(4+self.Na*2,64) 
        self.linear2 = nn.Linear(64,64)
        self.linear3 = nn.Linear(64,16)
        self.linear4 = nn.Linear(16,1)
        self.actv = nn.LeakyReLU(1e-2)
        self.rep = Representation_action(self.Na)
    def forward(self, s_cat,s_mouse,a_cat,a_mouse):
        s = torch.Tensor([[s_cat[j]//self.Ny,s_cat[j]%self.Ny, s_mouse[j]//self.Ny,s_mouse[j]%self.Ny] for j in range(len(s_cat))])
        s = torch.mul(s,1.0/self.Nx)
        #out = self.linear1(torch.cat((s,self.rep(a_cat),self.rep(a_mouse)),dim=1))
        x = torch.cat((s,self.rep(a_cat),self.rep(a_mouse)),dim=1)
        out = self.linear1(x)
        out = self.actv(out)
        out = self.linear2(out)
        out = self.actv(out)
        out = self.linear3(out)
        out = self.actv(out)
        out = self.linear4(out)
        #out = out[:,a_cat*self.Na+a_mouse]
        return out
#class Q(nn.Module):
#    def __init__(self,env):
#        super(Q,self).__init__()
#        self.Na = env.Na
#        self.Nx = env.Nx
#        self.Ny = env.Ny
#        self.linear1 = nn.Linear(2,8) 
#        self.linear2 = nn.Linear(8,16)
#        self.linear3 = nn.Linear(16,32)
#        self.linear4 = nn.Linear(32,self.Na*self.Na)
#        self.actv = nn.ReLU()
#    def forward(self, s_cat,s_mouse,a_cat,a_mouse):
#        s = torch.Tensor([[s_cat[j]//self.Ny - s_mouse[j]//self.Ny,s_cat[j]%self.Ny-s_mouse[j]%self.Ny] for j in range(len(s_cat))])
#        out = self.linear1(s)
#        out = self.linear2(out)
#        out = self.actv(out)
#        out = self.linear3(out)
#        out = self.actv(out)
#        out = self.linear4(out)
#        out = out[:,a_cat*self.Na+a_mouse]
#        return out
