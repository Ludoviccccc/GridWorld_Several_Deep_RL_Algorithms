import torch.nn as nn
import torch
class Q(nn.Module):
    def __init__(self,env):
        super(Q,self).__init__()
        self.Nx = env.Nx
        self.Ny = env.Ny
        self.Na = env.Na
        self.linear1 = nn.Linear(self.Ny*self.Nx + self.Na,16)
        self.linear2 = nn.Linear(16,16)
        self.linear3 = nn.Linear(16,1)
        self.actv = nn.ReLU()
        self.env = env
    def forward(self, s,a):
        x = torch.cat((self.env.representation(s),self.env.representationaction(a)),dim=1)
        out = self.linear1(x)
        out = self.actv(out)
        out = self.linear2(out)
        out = self.actv(out)
        out = self.linear3(out)
        return out
    def Qmax(self,state_vec):
        return torch.Tensor([max(self.__call__([s]*self.Na, torch.arange(self.Na)).detach().squeeze()) for s in state_vec])

    def amax_epsilon(self,state_vec, epsilon):
        out = []
        for s in state_vec:
            if torch.bernoulli(torch.Tensor([epsilon])):
                out.append(torch.randint(0,self.Na,(1,))[0])
            else:
                out.append(torch.argmax(self.__call__([s]*self.Na, torch.arange(self.Na)).detach().squeeze()))
        return out

