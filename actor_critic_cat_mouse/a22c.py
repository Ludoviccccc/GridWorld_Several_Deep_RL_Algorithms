import torch
import torch.nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np
from rep import Representation, Representation_action
from env2 import grid
from agent import Mouse, Cat

def A2C(env:grid,
        mouse:Mouse,
        cat:Cat,
        batch_size:int,
        n_episodes:int,
        freqsave=100, 
        start:int = 0, 
        gamma:int =.99,
        fact = .99,
        min_eps:float=0.1):
    retour_episodes = {"cat":[],"mouse":[]}
    loss_pi = {"cat":[],"mouse":[]}
    loss_Q = {"cat":[],"mouse":[]}
    rep_cl = Representation(env.Nx,env.Ny)
    rep_ac = Representation_action(env.Na)

    for j in range(start,n_episodes+1):
        #step 1
        env.reset()
        s_tab = {"cat":env.state_cat(),"mouse": env.state_mouse()}
        while env.terminated():
            env.reset()
        print(f"episode {j}/{n_episodes}")
        n = 0
        return_cat = torch.Tensor([0])
        return_mouse = torch.Tensor([0])
        while not env.terminated() and not env.truncated():
            a_tab = {"cat":cat(s_tab), "mouse":mouse(s_tab)}
            s_tab_prim,reward = env.transition(a_tab)
            cat.buffer.store({"state":s_tab,"action":a_tab,"new_state":s_tab_prim,"reward":reward["cat"]})
            mouse.buffer.store({"state":s_tab,"action":a_tab,"new_state":s_tab_prim,"reward":reward["mouse"]})
            s_tab = s_tab_prim
            n+=1
            return_cat += reward["cat"]*(gamma**n)
            return_mouse += reward["mouse"]*(gamma**n)
            if j<2:
                continue
            for label,agent in [("mouse",mouse),("cat",cat)]:
                sample = agent.buffer.sample(min(batch_size,len(agent.buffer.memory_state["mouse"])))
                #print(sample)
                a_prim_tab = {"cat":[],"mouse":[]}
                a_prim_tab["cat"] = cat.p(sample["new_state"]["cat"],sample["new_state"]["mouse"])
                a_prim_tab["mouse"] = mouse.p(sample["new_state"]["cat"],sample["new_state"]["mouse"])

                targets =  torch.Tensor(sample["reward"]) + gamma * agent.Qf_target(sample["new_state"],a_prim_tab).detach().squeeze()
                #update critic

                loss_ = agent.updateQ(sample["state"],sample["action"],targets)   
                api = {"cat":[],"mouse":[]}
                logits = {"cat":[],"mouse":[]}
                agent.optimizerpi.zero_grad()
                api["cat"],   logits["cat"] = cat.p(sample["state"]["cat"],sample["state"]["mouse"], logit = True)
                api["mouse"],   logits["mouse"] = cat.p(sample["state"]["cat"],sample["state"]["mouse"], logit = True)
                if label=="cat":
                    logpi = F.cross_entropy(logits["cat"],rep_ac(api["cat"]),weight = None, reduction = 'none')
                elif label=="mouse":
                    logpi = F.cross_entropy(logits["mouse"],rep_ac(api["mouse"]),weight = None, reduction = 'none')
                else:
                    print("erreur")
                    exit()
                #continue
                #update actor
                nploss = agent.updatePi(
                                api,
                                logpi,
                                sample["state"])
                #exit()
                loss_pi[label].append(nploss.item())
                loss_Q[label].append(loss_.item())
        mouse.update_target_net()
        cat.update_target_net()
        mouse.epsilon=max(min_eps,mouse.epsilon*fact)
        cat.epsilon=max(min_eps,cat.epsilon*fact)
        if n>0:
            retour_episodes["mouse"].append(return_mouse.item())
            retour_episodes["cat"].append(return_cat.item())
        if j%20==0 and j>0:
            print("episodes", j,f"/{n_episodes}")
            print("epsilon mouse", mouse.epsilon)
            print("epsilon cat", cat.epsilon)
            print("return_episodes mouse last five episodes",np.mean(retour_episodes["mouse"][-5:]))
            print("return_episodes cat last five episodes",np.mean(retour_episodes["cat"][-5:]))
        if j%20==0 and j>0:
            cat.save(j)
            mouse.save(j)
        if j%100==0 and j>=200:
            plt.figure()
            plt.plot(retour_episodes["cat"][:], label="return cat episode for initial state")
            plt.legend()
            plt.savefig("plot/reward_cat")
            plt.close()

            plt.figure()
            plt.plot(retour_episodes["mouse"][:], label="retour mouse episode for initial state")
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

