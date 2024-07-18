import torch.nn as nn
import torch
class Q(nn.Module):
    def __init__(self,Nx,Ny,Na):
        super(Q,self).__init__()
        self.Nx = Nx
        self.Ny = Ny
        self.Na = Na
        self.linear1 = nn.Linear(self.Ny*self.Nx + Na,32)
        self.linear2 = nn.Linear(32,16)
        self.linear3 = nn.Linear(16,1)
        self.actv = nn.ReLU()
    def forward(self, s,a):
        x = torch.cat((s,a),dim=1)
        out = self.linear1(x)
        out = self.actv(out)
        out = self.linear2(out)
        out = self.actv(out)
        out = self.linear3(out)
        return out
