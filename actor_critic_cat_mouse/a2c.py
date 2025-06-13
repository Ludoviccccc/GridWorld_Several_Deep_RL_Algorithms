import torch
import torch.nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np
from rep import Representation, Representation_action
from env import grid
def A2C(buffer,
        rep_cl:Representation,
        rep_ac:Representation_action,
        Q_tab:list,
        optimizer_tab:list,
        p_tab,
        optimizerpi_tab,
        env:grid,
        N,
        batch_size,
        n_epochs,
        loadpath,
        loadopt,
        freqsave=100, 
        epsilon = 0.1,
        K = 1,
        start = 0, 
        gamma =.9):
    listLosspi0 = []
    listLossQ0 = []
    listLosspi1 = []
    listLossQ1 = []
    retour_episodes = []
    def updateQ(optimizerQ, Qvalue,states,actions, targets):  
        optimizerQ.zero_grad()
        loss = F.mse_loss(Qvalue(states,actions).squeeze(),targets.squeeze())
        loss.backward()
        optimizerQ.step()
        return loss
    def updatePi(Q,optimizerpi, p, state):
        api, logits_ap = p(state, logit = True)
        #print("api",api)
        #print("logits", logits_ap)
        optimizerpi.zero_grad()
        logpi = F.cross_entropy(logits_ap,rep_ac(api),weight = None, reduction = 'none')
        advantage = Q(state,rep_ac(api)).squeeze().detach()#-V_batch.detach() 
        advantage = advantage.detach()
        NegativPseudoLoss = torch.mean(torch.mul(logpi,advantage))
        NegativPseudoLoss.backward()
        optimizerpi.step()
        return NegativPseudoLoss
    def epsilon_greedy_policy(state_vec,p):
        out = []
        for s in state_vec:
            if torch.bernoulli(torch.Tensor([epsilon])):
                out.append(torch.randint(0,env.Na,(1,))[0])
            else:
                out.append(int(p(rep_cl([s])).detach().item()))
        return out[0] 
    for j in range(start,n_epochs+1):
        #step 1
        env.reset()
        s_tab = [env.cat, env.mouse]
        print(f"episode {j}")
        while s_tab[0]!=s_tab[1]:
            a_tab = []
            for k in range(2):
                a_tab.append(epsilon_greedy_policy([s_tab[k]],p_tab[k]))
            s_tab_prim,reward_tab = env.transition(a_tab)
            {"state":[s_tab],"action":[a_tab],"new_state":[s_tab_prim],"reward":[reward_tab]}
            buffer.store({"state":[s_tab],"action":[a_tab],"new_state":[s_tab_prim],"reward":[reward_tab]})
            s_tab = s_tab_prim
            if j<1:
                continue
            for l,Q in enumerate(Q_tab):
                sample = buffer.sample(min(batch_size,len(buffer.memory_state)))
                a_prim_tab = []
                for m in range(2):
                    a_prim_tab.append(p_tab[m](rep_cl(sample["new_state"][:,m])))
                targets =  [sample["reward"][:,j] + torch.mul(gamma,q(rep_cl(sample["new_state"][:,j]),rep_ac(a_prim_tab[j])).squeeze()) for j,q in enumerate(Q_tab)]
                #update critic
                loss = []
                for m in range(2):
                    for k in range(K):
                        loss_ = updateQ(optimizer_tab[m], Q_tab[m],rep_cl(sample["state"][:,m]),rep_ac(sample["action"][:,m]), targets[m])   
                    loss.append(loss_)
                #update actor
                NegativPseudoLoss = []
                for m in range(2):
                    NegativPseudoLoss.append(updatePi(Q_tab[m],optimizerpi_tab[m], p_tab[m], rep_cl(sample["state"][:,m])))

            listLosspi0.append(NegativPseudoLoss[0].item())
            listLosspi1.append(NegativPseudoLoss[1].item())
            listLossQ0.append(loss[0].item())
            listLossQ1.append(loss[1].item())

        if j%100==0:
            print("epochs", j,f"/{n_epochs}")
            print("NegativPseudoLoss0",torch.mean(torch.Tensor(listLosspi0)))
            print("NegativPseudoLoss1",torch.mean(torch.Tensor(listLosspi1)))
            print("Loss Q0", torch.mean(torch.Tensor(listLossQ0)))
            print("Loss Q1", torch.mean(torch.Tensor(listLossQ1)))
        #    if len(retour_episodes)>0:
        #        print("retour dernier episode", retour_episodes[-1])
            torch.save(p_tab[0].state_dict(), os.path.join(loadpath,f"pi_0_load_{j}.pt"))
            torch.save(p_tab[1].state_dict(), os.path.join(loadpath,f"pi_1_load_{j}.pt"))
            torch.save(optimizerpi_tab[0].state_dict(), os.path.join(loadopt,f"opt_0_pi_load_{j}.pt"))
            torch.save(optimizerpi_tab[1].state_dict(), os.path.join(loadopt,f"opt_1_pi_load_{j}.pt"))
            torch.save(Q_tab[0].state_dict(), os.path.join(loadpath,f"q_0_load_{j}.pt"))
            torch.save(Q_tab[1].state_dict(), os.path.join(loadpath,f"q_1_load_{j}.pt"))
            torch.save(optimizer_tab[0].state_dict(), os.path.join(loadopt,f"opt_0_q_load_{j}.pt"))
            torch.save(optimizer_tab[1].state_dict(), os.path.join(loadopt,f"opt_1_q_load_{j}.pt"))

        #if j%100==0 and j>100:
        #    list_retour, iterations = testfunc(p,env, epsilon)
        #    retour_episodes.append(list_retour[0])
        #    print("plot")
        #    plt.figure()
        #    plt.semilogy(retour_episodes, label="retour episode pour l'etat initial")
        #    plt.legend()
        #    plt.savefig("recompense")
        #    plt.close()

        #    plt.figure()
        #    plt.semilogy(listLosspi, label="negativ pseudo loss")
        #    plt.legend()
        #    plt.savefig("negatigpseudoloss")
        #    plt.close()

        #    plt.figure()
        #    plt.plot(listLossQ, label="Q loss")
        #    plt.legend()
        #    plt.savefig("Qloss")
        #    plt.close()
    return listLosspi0,listLosspi1,listLossQ0, listLossQ1



def testfunc(p, env, epsilon, plot = False, gamma = .9):
    i = 0
    s = torch.randint(0,env.Na,(1,)).item()
    rewardlist = []
    list_recompense =[]
    while True:
        if torch.bernoulli(torch.Tensor([epsilon])):
            a = torch.randint(0,env.Na,(1,)) 
        else:
            a  = p([s])
        sp,R = env.transition(a,s)
        s = sp
        rewardlist.append(R)
        if plot:
            env.grid(s)
            print("")
        list_recompense.append(R*gamma**i)
        if s==env.G:
            break
        i+=1
    list_retour = [sum(list_recompense[i:]) for i in range(len(list_recompense))]
    print(f"{i} iterations")
    return list_retour, i
