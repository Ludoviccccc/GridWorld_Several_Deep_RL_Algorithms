import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import numpy as np

class Agent:
    def __init__(self, env,epsilon):
        self.Na = env.Na
        self.Nx = env.Nx
        self.Ny = env.Ny
        self.epsilon = epsilon
    def amax_epsilon(self,Q,state_vec):
          out = []
          for s in state_vec:
              if torch.bernoulli(torch.Tensor([self.epsilon])):
                  out.append(torch.randint(0,self.Na,(1,))[0])
              else:
                  out.append(torch.argmax(Q([s]*self.Na, torch.arange(self.Na)).detach().squeeze()))
          return out

