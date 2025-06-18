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

def updatePi(Q_tab:list[Q],optimizerpi_tab:list, p_tab:list[policy], s_tab:list,rep_ac:Representation_action):
    api0, logits_ap0 = p_tab[0](s_tab[0],s_tab[1], logit = True)
    api1, logits_ap1 = p_tab[1](s_tab[0],s_tab[1], logit = True)

    optimizerpi_tab[0].zero_grad()
    logpi0 = F.cross_entropy(logits_ap0,rep_ac(api0),weight = None, reduction = 'none')
    advantage0 = Q_tab[0](s_tab[0],s_tab[1],rep_ac(api0),rep_ac(api1)).squeeze().detach()#-V_batch.detach() 
    advantage0 = advantage0.detach()
    NegativPseudoLoss0 = torch.mean(torch.mul(logpi0,advantage0)) 
    NegativPseudoLoss0.backward()
    optimizerpi_tab[0].step()

    optimizerpi_tab[1].zero_grad()
    logpi1 = F.cross_entropy(logits_ap1,rep_ac(api1),weight = None, reduction = 'none')
    advantage1 = Q_tab[1](s_tab[0],s_tab[1],rep_ac(api0),rep_ac(api1)).squeeze().detach()#-V_batch.detach() 
    advantage1 = advantage1.detach()
    NegativPseudoLoss1 = torch.mean(torch.mul(logpi1,advantage1)) 
    NegativPseudoLoss1.backward()
    optimizerpi_tab[1].step()
    return NegativPseudoLoss0, NegativPseudoLoss1
def A2C(buffer,
        rep_cl:Representation,
        rep_ac:Representation_action,
        Q_tab:list[Q],
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
    def updateQ(optimizerQ, Qvalue,states0, states1,actions0, actions1, targets):  
        optimizerQ.zero_grad()
        loss = F.mse_loss(Qvalue(states0,states1,actions0,actions1).squeeze(),targets.squeeze())
        loss.backward()
        optimizerQ.step()
        return loss
    def epsilon_greedy_policy(state_vec,p):
        out = []
        for s in state_vec:
            if torch.bernoulli(torch.Tensor([epsilon])):
                out.append(torch.randint(0,env.Na,(1,))[0])
            else:
                out.append(int(p(rep_cl([state_vec[0]]),rep_cl([state_vec[1]])).detach().item()))
        return out[0] 
    for j in range(start,n_epochs+1):
        #step 1
        env.reset()
        s_tab = [env.cat, env.mouse]
        print(f"episode {j}")
        while s_tab[0]!=s_tab[1]:
            a_tab = []
            for k in range(2):
                a_tab.append(epsilon_greedy_policy(s_tab,p_tab[k]))
            s_tab_prim,reward_tab = env.transition(a_tab)
            buffer.store({"state":[s_tab],"action":[a_tab],"new_state":[s_tab_prim],"reward":[reward_tab]})
            s_tab = s_tab_prim
            if j<10:
                continue
            epsilon = max(.1,epsilon*.98)
            for l,Q in enumerate(Q_tab):
                sample = buffer.sample(min(batch_size,len(buffer.memory_state)))
                a_prim_tab = []
                for m in range(2):
                    a_prim_tab.append(p_tab[m](rep_cl(sample["new_state"][:,0]),rep_cl(sample["new_state"][:,1])))
                targets =  sample["reward"][:,l] + torch.mul(gamma,Q(rep_cl(sample["new_state"][:,0]),
                                                                     rep_cl(sample["new_state"][:,1]),
                                                                     rep_ac(a_prim_tab[0]),
                                                                     rep_ac(a_prim_tab[1])).squeeze())
                #update critic
                loss = []
                loss_ = updateQ(optimizer_tab[l],
                                Q_tab[l],
                                rep_cl(sample["state"][:,0]),
                                rep_cl(sample["state"][:,1]),
                                rep_ac(sample["action"][:,0]),
                                rep_ac(sample["action"][:,1]),
                                targets)   
                loss.append(loss_)
                #update actor
                NegativPseudoLoss = []
                nploss = updatePi(Q_tab,
                                optimizerpi_tab,
                                p_tab,
                                [rep_cl(sample["state"][:,0]),
                                rep_cl(sample["state"][:,1])],
                                rep_ac
                                )
                NegativPseudoLoss.append(nploss)
            #listLossQ0.append(loss[0].item())
            #listLossQ1.append(loss[1].item())

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
