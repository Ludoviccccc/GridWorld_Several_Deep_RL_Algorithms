import torch.nn as nn
import torch
class Q(nn.Module):
    def __init__(self,env):
        super(Q,self).__init__()
        self.Na = env.Na
        self.linear1 = nn.Linear(2 + 2*self.Na,16)
        self.linear2 = nn.Linear(16,32)
        self.linear3 = nn.Linear(32,16)
        self.linear4 = nn.Linear(16,1)
        self.actv = nn.ReLU()
    def forward(self, s_cat,a_cat,a_mouse):
        x = torch.cat((s_cat,a_cat,a_mouse),dim=1)
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
        self.linear1 = nn.Linear(2 + env.Na,32)
        self.linear4 = nn.Linear(32,16)
        self.linear5 = nn.Linear(16,1)
        self.actv = nn.ReLU()
    def forward(self, s1,a1):
        x = torch.cat((s1,a1),dim=1)
        out = self.linear1(x)
        out = self.actv(out)
        out = self.linear4(out)
        out = self.actv(out)
        out = self.linear5(out)
        return out
