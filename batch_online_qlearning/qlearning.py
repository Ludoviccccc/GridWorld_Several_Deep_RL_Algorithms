import torch
import torch.nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np
from Qfunc import Q
def qlearn(buffer,
           batch_size,
           M,
           N,
           K,
           Qvalue,
           optimizerQ,
           env,
           n_epochs,
           loadpath,
           loadopt, 
           gamma = .9,
           freqsave=100,
           epsilon = 0.1,
           start = 0
           ):
    Loss = torch.nn.MSELoss()
    listLossQ = []
    recompense_episodes = []
    Qprim = Q(env)
    def swap():
        Qprim.load_state_dict(Qvalue.state_dict())
    def collection(M):
        #choose some policy for the collection like the greedy one
        init_samp = {"state": [],
                     "action": [],
                     "new_state":[],
                     "reward": []}
        init_samp["state"] = torch.randint(0,env.Nx*env.Ny,(M,))  
        init_samp["action"]  = Qvalue.amax_epsilon(init_samp["state"], epsilon)
        init_samp["new_state"], init_samp["reward"] = env.transitionvec(init_samp["action"], init_samp["state"])
        buffer.store(init_samp)
    def updateQ(Samp):
        target = Samp["reward"] + gamma*Qprim.Qmax(Samp["new_state"])
        optimizerQ.zero_grad()
        pred = Qvalue(Samp["state"],Samp["action"]).squeeze()
        loss = Loss(pred,target)
        loss.backward()
        optimizerQ.step()
        listLossQ.append(loss.detach().to("cpu"))
################################################################
    for i in range(start,n_epochs+1):
        swap()
        collection(M)
        for j in range(N):
            Samp = buffer.sample(batch_size)
            for k in range(K):
                 updateQ(Samp)
            if j%1000==0:
                print(f"i = {i}, n = {j},k={k}")
                torch.save(Qvalue.state_dict(), os.path.join(loadpath,f"q_load_{i}_{j}.pt"))
                torch.save(optimizerQ.state_dict(), os.path.join(loadopt,f"opt_q_load_{i}_{j}.pt"))
    return None 
def test(Qvalue,
         env,
         epsilon, 
         plot = False
         ):
    i = 0
    s = torch.randint(0,env.Nx*env.Ny,(1,)).item()
    rewardlist = []
    while True:
        if torch.bernoulli(torch.Tensor([epsilon])):
            a = torch.randint(0,env.Na,(1,)) 
        else:
            a = torch.argmax(Qvalue([s]*env.Na,torch.arange(env.Na)).squeeze())
        sp,R = env.transition(a,s)
        s = sp
        i+=1
        if plot:
            env.grid(s)
        rewardlist.append(R)
        if s==env.G:
            break
    print(f"{i} pas de temps")
    return i
