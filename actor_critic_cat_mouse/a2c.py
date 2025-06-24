import torch
import torch.nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np
from rep import Representation, Representation_action
from env import grid
from Qfunc import Q
from policy import policy

def updatePi(Q:Q,
             optimizerpi,
             api0:int,
             api1:int,
             logpi,
             s_tab:list,
             rep_ac:Representation_action):
    optimizerpi.zero_grad()
    advantage = Q(s_tab[0],s_tab[1],rep_ac(api0),rep_ac(api1)).squeeze().detach()
    advantage = advantage.detach()
    NegativPseudoLoss = torch.mean(torch.mul(logpi.squeeze(),advantage)) 
    NegativPseudoLoss.backward()
    optimizerpi.step()
    return NegativPseudoLoss
def updateQ(optimizerQ,
            Qvalue,
            states0,
            states1,
            actions0, 
            actions1, 
            targets):  
    optimizerQ.zero_grad()
    loss = F.mse_loss(Qvalue(states0,states1,actions0,actions1).squeeze(),targets.squeeze())
    loss.backward()
    optimizerQ.step()
    return loss
def epsilon_greedy_policy(state_vec:list,p:float,epsilon:float,env:grid,rep_cl:Representation):
    if np.random.binomial(1,epsilon):
        out = np.random.randint(0,env.Na)
    else:
        out = int(p(rep_cl([state_vec[0]]),rep_cl([state_vec[1]])).detach().item())
    return out 
def A2C(buffer,
        rep_cl:Representation,
        rep_ac:Representation_action,
        Q_tab:list[Q],
        optimizer_tab:list,
        p_tab:list[policy],
        optimizerpi_tab:list,
        env:grid,
        N:int,
        batch_size:int,
        n_episodes:int,
        loadpath:str,
        loadopt:str,
        freqsave=100, 
        epsilon = 0.1,
        K:int = 5,
        start:int = 0, 
        gamma:int =.99):
    retour_episodes = {"cat":[],"mouse":[]}
    loss_pi = {"cat":[],"mouse":[]}
    loss_Q = {"cat":[],"mouse":[]}
    for j in range(start,n_episodes+1):
        #step 1
        env.reset()
        s_tab = [env.cat, env.mouse]
        terminated = env.cat==env.mouse
        print(f"episode {j}")
        n = 0
        list_recompense = {"mouse":[],"cat":[]}
        while not terminated and n<=200:
            a_tab = []
            for k in range(2):
                a_tab.append(epsilon_greedy_policy(s_tab,p_tab[k],epsilon,env,rep_cl))
            s_tab_prim,reward_tab,terminated = env.transition(a_tab)
            buffer.store({"state":[s_tab],"action":[a_tab],"new_state":[s_tab_prim],"reward":[reward_tab]})
            s_tab = s_tab_prim
            n+=1
            list_recompense["cat"].append(reward_tab[0]*gamma**n)
            list_recompense["mouse"].append(reward_tab[1]*gamma**n)
            if j<1:
                continue
            epsilon = max(.2,epsilon*.999)
            for l,Q in enumerate(Q_tab):
                sample = buffer.sample(min(batch_size,len(buffer.memory_state)))
                a_prim_tab = []
                for m in range(2):
                    a_prim_tab.append(p_tab[m](rep_cl(sample["new_state"][:,0]),rep_cl(sample["new_state"][:,1])))
                targets =  sample["reward"][:,l] + gamma * Q(rep_cl(sample["new_state"][:,0]),
                                                             rep_cl(sample["new_state"][:,1]),
                                                             rep_ac(a_prim_tab[0]),
                                                             rep_ac(a_prim_tab[1])).detach().squeeze()
                #update critic
                for k in range(K):
                    loss_ = updateQ(optimizer_tab[l],
                                    Q_tab[l],
                                    rep_cl(sample["state"][:,0]),
                                    rep_cl(sample["state"][:,1]),
                                    rep_ac(sample["action"][:,0]),
                                    rep_ac(sample["action"][:,1]),
                                    targets)   
                api0, logits_ap0 = p_tab[0](rep_cl(sample["state"][:,0]),rep_cl(sample["state"][:,1]), logit = True)
                api1, logits_ap1 = p_tab[1](rep_cl(sample["state"][:,0]),rep_cl(sample["state"][:,1]), logit = True)
                if l==0:
                    logpi = F.cross_entropy(logits_ap0,rep_ac(api0),weight = None, reduction = 'none')
                elif l==1:
                    logpi = F.cross_entropy(logits_ap1,rep_ac(api1),weight = None, reduction = 'none')
                else:
                    print("erreur")
                    exit()
                #update actor
                nploss = updatePi(Q,
                                optimizerpi_tab[l],
                                api0,
                                api1,
                                logpi,
                                [rep_cl(sample["state"][:,0]),rep_cl(sample["state"][:,1])],
                                rep_ac)
                loss_pi[["mouse","cat"][l]].append(nploss.item())
                loss_Q[["mouse","cat"][l]].append(loss_.item())
        if n>0:
            retour_episodes["mouse"].append(sum(list_recompense["mouse"]))
            retour_episodes["cat"].append(sum(list_recompense["cat"]))
        if j%20==0 and j>0:
            print("episodes", j,f"/{n_episodes}")
            print("epsilon", epsilon)
            print("return_episodes mouse last five episodes",np.mean(retour_episodes["mouse"][-5:]))
            print("return_episodes cat last five episodes",np.mean(retour_episodes["cat"][-5:]))
        if j%20==0 and j>0:
            torch.save(p_tab[0].state_dict(), os.path.join(loadpath,f"pi_0_load_{j}.pt"))
            torch.save(p_tab[1].state_dict(), os.path.join(loadpath,f"pi_1_load_{j}.pt"))
            torch.save(optimizerpi_tab[0].state_dict(), os.path.join(loadopt,f"opt_0_pi_load_{j}.pt"))
            torch.save(optimizerpi_tab[1].state_dict(), os.path.join(loadopt,f"opt_1_pi_load_{j}.pt"))
            torch.save(Q_tab[0].state_dict(), os.path.join(loadpath,f"q_0_load_{j}.pt"))
            torch.save(Q_tab[1].state_dict(), os.path.join(loadpath,f"q_1_load_{j}.pt"))
            torch.save(optimizer_tab[0].state_dict(), os.path.join(loadopt,f"opt_0_q_load_{j}.pt"))
            torch.save(optimizer_tab[1].state_dict(), os.path.join(loadopt,f"opt_1_q_load_{j}.pt"))

        if j%100==0 and j>=100:
            plt.figure()
            plt.plot(retour_episodes["cat"], label="return cat episode for initial state")
            plt.plot(retour_episodes["mouse"], label="retour mouse episode for initial state")
            plt.legend()
            plt.savefig("plot/recompense")
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

