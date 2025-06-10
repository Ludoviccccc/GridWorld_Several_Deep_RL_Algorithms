import torch
import torch.nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np

def A2C(buffer,
        Q_tab,
        optimizer_tab,
        p_tab,
        optimizerpi_tab,
        Qvalue,
        optimizerQ,
        env,N,
        batch_size,
        n_epochs,
        loadpath,
        loadopt,
        freqsave=100, 
        epsilon = 0.1,
        K = 1,
        start = 0, 
        gamma =.9):
    listLosspi = []
    listLossQ = []
    retour_episodes = []
    def updateQ(optimizerQ, Qvalue,samp, targets):  
        optimizerQ.zero_grad()
        loss = F.mse_loss(Qvalue(samp["state"],samp["action"]).squeeze(),targets.squeeze())
        loss.backward()
        optimizerQ.step()
        return loss
    def updatePi(optimizerpi, p, samp):
        print("samp",samp["state"].shape)
        api, logits_ap = p(samp["state"], logit = True)
        print("api",api)
        print("logits", logits_ap)
        optimizerpi.zero_grad()
        logpi = F.cross_entropy(logits_ap,env.representationaction(api),weight = None, reduction = 'none')
        advantage = Qvalue(samp["state"],api).squeeze().detach()#-V_batch.detach() 
        advantage = advantage.detach()
        NegativPseudoLoss = torch.mean(torch.mul(logpi,advantage))
        NegativPseudoLoss.backward()
        optimizerpi.step()
        return NegativPseudoLoss
    def epsilon_greedy_policy(state_vec):                                                                                                                                                                   
        out = []                                                                                                                                                                                            
        for s in state_vec:                                                                                                                                                                                 
            if torch.bernoulli(torch.Tensor([epsilon])):                                                                                                                                                    
                out.append(torch.randint(0,env.Na,(1,))[0])                                                                                                                                                 
            else:                                                                       
                out.append(int(p([s]).detach().item()))
        return torch.Tensor(out)

    for j in range(start,n_epochs+1):
        #step 1
        s_tab = [torch.randint(0,env.Nx*env.Ny,(1,)) for j in range(2)]
        print(f"episode {j}")
        while s_tab[0]!=s_tab[1]:
            a_tab = []
            for k in range(2):
                a_tab.append(p_tab[k](s_tab[k]))
            s_tab_prim,reward_tab = env.transition(a_tab)
            buffer.store({"state":s_tab,"action":a_tab,"new_state":s_tab_prim,"reward":reward_tab})
            s_tab = s_tab_prim
            for l,Q in enumerate(Q_tab):
                sample = buffer.sample(min(batch_size,len(buffer.memory_state)))
                a_prim_tab = []
                for m in range(2):
                    a_prim_tab.append(p_tab[m](sample["new_state"][m]))
                targets = samp["reward"] + gamma*Q(samp["new_state"],a_prim_tab).squeeze()
                #update critic
                for k in range(K):
                    loss = updateQ(optimizer_tab[l], Q, targets)   
                #update actor
                NegativPseudoLoss = updatePi(optimizerpi_tab[l], p_tab[l], sample)   

        #    listLosspi.append(NegativPseudoLoss.item())
        #    listLossQ.append(loss.item())

        #if j%100==0:
        #    print("epochs", j,f"/{n_epochs}")
        #    print("NegativPseudoLoss",torch.mean(torch.Tensor(listLosspi)))
        #    print("Loss Q", torch.mean(torch.Tensor(listLossQ)))
        #    if len(retour_episodes)>0:
        #        print("retour dernier episode", retour_episodes[-1])
        #    torch.save(p.state_dict(), os.path.join(loadpath,f"pi_load_{j}.pt"))
        #    torch.save(optimizerpi.state_dict(), os.path.join(loadopt,f"opt_pi_load_{j}.pt"))
        #    torch.save(Qvalue.state_dict(), os.path.join(loadpath,f"q_load_{j}.pt"))
        #    torch.save(optimizerQ.state_dict(), os.path.join(loadopt,f"opt_q_load_{j}.pt"))

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
    #return listLosspi, retour_episodes



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
