import torch.nn as nn
import torch
class Q(nn.Module):
    def __init__(self,env):
        super(Q,self).__init__()
        self.Nx = env.Nx
        self.Ny = env.Ny
        self.Na = env.Na
        self.env = env
        self.linear1 = nn.Linear(self.Ny*self.Nx + self.Na,16)
        self.linear2 = nn.Linear(16,16)
        self.linear3 = nn.Linear(16,1)
        self.actv = nn.ReLU()
        self.atv2 = nn.Tanh()
    def forward(self, s_souris,s_chat,a):
        x = torch.cat((self.env.representation(s_souris,s_chat),self.env.representationaction(a)),dim=1)
        out = self.linear1(x)
        out = self.atv2(out)
        out = self.linear2(out)
        out = self.actv(out)
        out = self.linear3(out)
        return out.squeeze()
    def argmax(self,s_souris,s_chat):
        return torch.argmax(self.__call__([s_souris]*self.env.Na, [s_chat]*self.env.Na,torch.arange(self.env.Na))).reshape((1,))
