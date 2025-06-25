import torch
import torch.nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np
from rep import Representation, Representation_action
from env import grid
from agent import Mouse, Cat

def A2C(env:grid,
        mouse:Mouse,
        cat:Cat,
        batch_size:int,
        n_episodes:int,
        freqsave=100, 
        epsilon = 0.1,
        K:int = 5,
        start:int = 0, 
        gamma:int =.99):
    retour_episodes = {"cat":[],"mouse":[]}
    loss_pi = {"cat":[],"mouse":[]}
    loss_Q = {"cat":[],"mouse":[]}
    rep_cl = Representation(env.Nx,env.Ny)
    rep_ac = Representation_action(env.Na)

    for j in range(start,n_episodes+1):
        #step 1
        env.reset()
        s_tab = {"cat":env.cat,"mouse": env.mouse}
        while env.terminated():
            env.reset()
        print(f"episode {j}/{n_episodes}")
        n = 0
        return_cat = torch.Tensor([0])
        return_mouse = torch.Tensor([0])
        while not env.terminated():
            a_tab = {"cat":cat.act([s_tab["cat"],s_tab["mouse"]]), "mouse":mouse.act(s_tab["mouse"])}
            s_cat,reward_cat = env.transition_cat(a_tab["cat"])
            s_mouse,reward_mouse = env.transition_mouse(a_tab["mouse"])
            s_tab_prim = {"cat":s_cat,"mouse":s_mouse}
            cat.buffer.store({"state":s_tab,"action":a_tab,"new_state":s_tab_prim,"reward":reward_cat})
            mouse.buffer.store({"state":s_tab,"action":a_tab,"new_state":s_tab_prim,"reward":reward_mouse})
            s_tab = s_tab_prim
            n+=1
            return_cat += reward_cat*gamma**n
            return_mouse += reward_mouse*gamma**n
            if j<1:
                continue
            epsilon = max(.1,epsilon*.999)
            for label,agent in [("mouse",mouse)]:
                sample = agent.buffer.sample(min(batch_size,len(agent.buffer.memory_state)))
                a_prim_tab = {"cat":[],"mouse":[]}
                a_prim_tab["cat"] = cat.p(rep_cl(sample["new_state"]["cat"]),rep_cl(sample["new_state"]["cat"]))
                a_prim_tab["mouse"] = mouse.p(rep_cl(sample["new_state"]["mouse"]))
                if isinstance(agent,Mouse):
                    targets =  sample["reward"][:] + gamma * agent.q(
                                                             rep_cl(sample["new_state"]["mouse"]),
                                                             rep_ac(a_prim_tab["mouse"])).detach().squeeze()
                else:
                    targets =  sample["reward"][:] + gamma * agent.q(rep_cl(sample["new_state"]["cat"]),
                                                             rep_cl(sample["new_state"]["mouse"]),
                                                             rep_ac(a_prim_tab["cat"]),
                                                             rep_ac(a_prim_tab["mouse"])).detach().squeeze()
                #update critic
                for k in range(K):
                    if isinstance(agent,Mouse):
                        loss_ = agent.updateQ(
                                        rep_cl(sample["state"]["mouse"]),
                                        rep_ac(sample["action"]["mouse"]),
                                        targets)   
                    else:
                        loss_ = agent.updateQ(
                                        rep_cl(sample["state"]["cat"]),
                                        rep_cl(sample["state"]["mouse"]),
                                        rep_ac(sample["action"]["cat"]),
                                        rep_ac(sample["action"]["mouse"]),
                                        targets)   
                api0, logits_ap0 = cat.p(rep_cl(sample["state"]["cat"]),rep_cl(sample["state"]["mouse"]), logit = True)
                api1, logits_ap1 = mouse.p(rep_cl(sample["state"]["mouse"]), logit = True)
                if label=="cat":

                    logpi = F.cross_entropy(logits_ap0,rep_ac(api0),weight = None, reduction = 'none')
                    #print("api0",rep_ac(api0))
                    #print("logits_ap0",logits_ap0)
                    #print(logpi)
                    #print(torch.sum(torch.mul(torch.log(F.softmax(logits_ap0,dim=1)),rep_ac(api0)),dim=1))
                    #exit()
                elif label=="mouse":
                    logpi = F.cross_entropy(logits_ap1,rep_ac(api1),weight = None, reduction = 'none')
                else:
                    print("erreur")
                    exit()
                #update actor
                if isinstance(agent,Mouse):
                    nploss = agent.updatePi(
                                    api1,
                                    logpi,
                                    rep_cl(sample["state"]["mouse"]))
                else:
                    nploss = agent.updatePi(
                                    api0,
                                    api1,
                                    logpi,
                                    [rep_cl(sample["state"]["cat"]),rep_cl(sample["state"]["mouse"])])
                loss_pi[label].append(nploss.item())
                loss_Q[label].append(loss_.item())
        if n>0:
            retour_episodes["mouse"].append(return_mouse.item())
            retour_episodes["cat"].append(return_cat.item())
        if j%20==0 and j>0:
            print("episodes", j,f"/{n_episodes}")
            print("epsilon", epsilon)
            print("return_episodes mouse last five episodes",np.mean(retour_episodes["mouse"][-5:]))
            print("return_episodes cat last five episodes",np.mean(retour_episodes["cat"][-5:]))
        if j%20==0 and j>0:
            cat.save(j)
            mouse.save(j)
        if j%100==0 and j>=100:
            plt.figure()
            plt.plot(retour_episodes["cat"], label="return cat episode for initial state")
            plt.legend()
            plt.savefig("plot/reward_cat")
            plt.close()

            plt.figure()
            plt.plot(retour_episodes["mouse"], label="retour mouse episode for initial state")
            plt.legend()
            plt.savefig("plot/reward_mouse")
            plt.close()

            plt.figure()
            plt.plot(loss_pi["mouse"])
            plt.savefig("plot/loss_pi_mouse")
            plt.close()

            plt.figure()
            plt.plot(loss_Q["mouse"])
            plt.savefig("plot/loss_Q_mouse")
            plt.close()

            plt.figure()
            plt.plot(loss_pi["cat"])
            plt.savefig("plot/loss_pi_cat")
            plt.close()
            
            plt.figure()
            plt.plot(loss_Q["cat"])
            plt.savefig("plot/loss_Q_cat")
            plt.close()
    return 0

