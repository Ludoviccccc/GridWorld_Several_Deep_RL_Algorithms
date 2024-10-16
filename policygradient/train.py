import torch
from torch.nn.utils.rnn import pad_sequence
import time
from torch import distributions
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
def trainfunc(n_episodes, p, env, optimizer):
    Loss = torch.nn.CrossEntropyLoss(weight = None, reduction = 'none')
    softm = nn.Softmax(dim=1)
    RR = 0
    nombres_iterations = []
    recompense = []
    for n in range(n_episodes):
        #s = env.S
        s = torch.randint(0,env.Nx*env.Ny, (1,))[0]
        R_list = []
        States = [s]
        actions_list = []
        out_list = []
        sum_R = 0
        i = 0
        optimizer.zero_grad()
        while True:
            #clear_output(wait=True)
            print('n', n+1)
            print(i)
            state = env.tensor_state(s)
            out  = p(state)
            out_list.append(out)
            dist  = distributions.Categorical(softm(out))
            if np.random.binomial(1, env.epsilon):
                a = torch.tensor([[np.random.randint(0,env.Na)]])
            else:
                a = dist.sample([1])
            actions_list.append(a.numpy()[0][0])  
            sp,R = env.transition(a,s)
            #env.grid(sp)
            s = sp
            States.append(s)
            print('reward previous ep', RR)
            #time.sleep(0.05)
            i+=1
            R_list.append(R)
            if sp==env.G:
                break
        T = torch.tensor([sum(R_list[i:])for i in range(len(R_list))]).type(torch.float32)
        T = (T - torch.mean(T))#/torch.std(T)
        out_p = pad_sequence(out_list)
        #print("actions_list", actions_list)
        actions_tens = env.zero_one(torch.tensor(actions_list),env.Na)
        print("actions_tens",actions_tens.shape)
        print("out_p[0]", out_p[0].shape)
        loss = Loss(out_p[0],actions_tens)
        RR =sum(R_list)
        recompense.append(RR)
        nombres_iterations.append(i)
        PseudoLoss_w = torch.multiply(loss,T)
        #print(PseudoLoss_w)
        NegativPseudoLoss = torch.mean(PseudoLoss_w)
        #calcul du gradient de la pseudoloss
        NegativPseudoLoss.backward()
        #On maximise la pseudo loss en minimisant son oppose
        #le signe moins vient de l'entropie croisee
        optimizer.step()
   # print("recompense", recompense)
   # print("nombre iterations", nombres_iterations)
    return recompense, nombres_iterations
