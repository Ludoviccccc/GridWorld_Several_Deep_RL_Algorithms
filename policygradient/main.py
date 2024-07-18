import torch.nn as nn
import sys
import torch
import numpy as np
from torch import distributions
from IPython.display import clear_output
import time 
import torch.nn.functional
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
import matplotlib.pyplot as plt
from policy import policy
from train import trainfunc
sys.path.append("../env")
from env import grid
if __name__=="__main__":
    Nx =    8
    Ny =    8
    Na =    4
    lr = 1e-3
    n_episodes = 2000
    p= policy(Nx,Ny,Na) 
    env = grid(Nx,Ny)
    optimizer = optim.Adam(p.parameters(), lr=lr)

    recompense,nombre_iterations = trainfunc(n_episodes,p, env,optimizer)
    plt.figure()
    plt.plot(recompense, label="recompense en fonction du nombre d'épisode")
    plt.legend()
    plt.savefig("recompense_apprentissage")
    plt.show()
