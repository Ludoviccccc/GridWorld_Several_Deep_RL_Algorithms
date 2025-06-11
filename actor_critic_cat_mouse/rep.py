import torch
from torch.nn.utils.rnn import pad_sequence
class Representation:
    def __init__(self,Nx,Ny):
        self.Nx = Nx
        self.Ny = Ny
        self.states_encod = torch.eye(self.Nx*self.Ny)
    def __call__(self,s:list[int]):
        out = pad_sequence([self.states_encod[int(i)] for i in s]).permute(1,0)
        return out 
class Representation_action:
    def __init__(self,Na):
        self.Na = Na
        self.actions_encod = torch.eye(self.Na)
    def __call__(self,s:list[int]):
        out = pad_sequence([self.actions_encod[int(i)] for i in s]).permute(1,0)
        return out 
