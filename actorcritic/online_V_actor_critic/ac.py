import torch
import torch.nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np
import subprocess
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def AC(Vfunc,optimizerV,p,optimizerpi,env, n_episodes, loadpath,loadopt, freqsave=100, epsilon = 0., K = 20, start = 0, gamma = 0.9):
    def epsilon_greedy_policy(state_vec):
        out = [] 
        for s in state_vec:
            if torch.bernoulli(torch.Tensor([epsilon])):
                out.append(torch.randint(0,env.Na,(1,))[0])
            else:
                out.append(p([s]).detach().squeeze())
        return out
    def updateV(samp, targets):
        optimizerV.zero_grad()
        loss = F.mse_loss(Vfunc(samp["state"]).squeeze(),targets.squeeze().cuda())
        loss.backward()
        optimizerV.step()
        return loss
    def updatePolicy(advantage, sample):
        _, logits_a = p(samp["state"], logit = True)
        logpi = F.cross_entropy(logits_a,env.representationaction(sample["action"]).cuda(),weight = None, reduction = 'none')
        NegativPseudoLoss = torch.mean(torch.mul(logpi,advantage)) # This is online update for actor critic
        NegativPseudoLoss.backward()
        optimizerpi.step()
        return NegativPseudoLoss

    listLosspi = []
    listLossV = []
    nombre_iteration_episodes = []
    #Initial state
    for j in range(start,n_episodes+1):
        k = 0
        samp = {"state": torch.randint(0,env.Nx*env.Ny,(1,))}
        while samp["state"][0]!=env.G:
            samp["action"] = epsilon_greedy_policy(samp["state"])
            samp["new_state"], samp["reward"] = env.transitionvec(samp["action"], samp["state"])
            #targets computation
            targets = samp["reward"].to(device) + gamma*Vfunc(samp["new_state"]).squeeze()
            targets = targets.detach()
            #V update 
            for l in range(K):
                loss = updateV(samp, targets)
            #advantage evaluation
            advantage = samp["reward"].cuda() + gamma*Vfunc(samp["new_state"]).squeeze() - Vfunc(samp["state"]).squeeze()
            advantage = advantage.detach()
            optimizerpi.zero_grad()
            #Policy update
            NegativPseudoLoss = updatePolicy(advantage, samp) # theta <-- theta + alpha* grad J(theta) using negativpseudoloss for policy pi
            k+=1
            samp["state"] = samp["new_state"]

            listLosspi.append(NegativPseudoLoss.item())
            listLossV.append(loss.item())
        if j%100==0:
            print(f"episod {j} finished with {k} iterations")
        if j%100==0:
            print("episodes", j,f"/{n_episodes}")
            print("NegativPseudoLoss",torch.mean(torch.Tensor(listLosspi)))
            print("Loss Q", torch.mean(torch.Tensor(listLossV)))
            if len(nombre_iteration_episodes)>0:
                print("number of iterations for the last episod", nombre_iteration_episodes)

        if j%500==0:
            torch.save(p.state_dict(), os.path.join(loadpath,f"pi_load_{j}.pt"))
            torch.save(optimizerpi.state_dict(), os.path.join(loadopt,f"opt_pi_load_{j}.pt"))
            torch.save(Vfunc.state_dict(), os.path.join(loadpath,f"q_load_{j}.pt"))
            torch.save(optimizerV.state_dict(), os.path.join(loadopt,f"opt_q_load_{j}.pt"))

    return listLosspi, nombre_iteration_episodes



def testfunc(p, env, epsilon, plot = False, graph = False):
    i = 0
    s = torch.randint(0,env.Nx*env.Ny,(1,)).item()
    rewardlist = []
    while True:
        if torch.bernoulli(torch.Tensor([epsilon])):
            a = torch.randint(0,env.Na,(1,)) 
        else:
            a  = p([s])
        sp,R = env.transition(a,s)
        s = sp
        i+=1
        rewardlist.append(R)
        if plot:
            if graph:
                env.grid(s,os.path.join("image",str(i)))
            else:
                env.grid(s)
            print("")
        if s==env.G:
            print(f"{i} iterations")
            break
    return i
