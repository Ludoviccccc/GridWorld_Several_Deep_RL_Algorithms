import torch
import torch.nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np

def A2C(buffer,Qvalue,optimizerQ,p,optimizerpi,env,N, batch_size, n_epochs, loadpath,loadopt, freqsave=100, epsilon = 0.1, K = 1, start = 0, gamma =.9):

    def collection(M):
        #choose some policy for the collection like the greedy one
        init_samp = {"state": [],
                     "action": [],
                     "new_state":[],
                     "reward": []}
        init_samp["state"] = torch.randint(0,env.Nx*env.Ny,(M,))
        init_samp["action"]  =  p(init_samp["state"])
        init_samp["new_state"], init_samp["reward"] = env.transitionvec(init_samp["action"], init_samp["state"])
        buffer.store(init_samp)
    listLosspi = []
    listLossQ = []
    retour_episodes = []

    for j in range(start,n_epochs+1):
        #step 1
        collection(batch_size)
        #step 2
        samp = buffer.sample(batch_size)
        for i in range(K):
            ap = p(samp["new_state"])
            #step 3 Q update
            targets = samp["reward"] + gamma*Qvalue(samp["new_state"],ap).squeeze()
            #targets computation
            optimizerQ.zero_grad()
            loss = F.mse_loss(Qvalue(samp["state"],samp["action"]).squeeze(),targets.squeeze())
            loss.backward()
            optimizerQ.step()
            #step 4 and 5
            #computing negativpseudoloss for policy pi
            optimizerpi.zero_grad()
            api, logits_ap = p(samp["state"], logit = True)
            logpi = F.cross_entropy(logits_ap,env.representationaction(api),weight = None, reduction = 'none')
            #V_batch est l'espérance conditionnelle de Q par rapport à s.
            #V_batch = torch.stack([torch.sum(torch.mul(Qvalue([si]*env.Na,torch.arange(env.Na)).squeeze(),F.softmax(logits_ap,dim=1).squeeze().detach())) for si in samp["state"]])
            ##print("V_batch shape", V_batch.shape)
            advantage = Qvalue(samp["state"],api).squeeze().detach()#-V_batch.detach() 
            NegativPseudoLoss = torch.mean(torch.mul(logpi,advantage))
            NegativPseudoLoss.backward()
            optimizerpi.step()
            #print("NevatigPseudo", NegativPseudoLoss.item())
            listLosspi.append(NegativPseudoLoss.item())
            listLossQ.append(loss.item())

        if j%100==0:
            print("epochs", j,f"/{n_epochs}")
            print("NegativPseudoLoss",torch.mean(torch.Tensor(listLosspi)))
            print("Loss Q", torch.mean(torch.Tensor(listLossQ)))
            if len(retour_episodes)>0:
                print("retour dernier episode", retour_episodes[-1])
            torch.save(p.state_dict(), os.path.join(loadpath,f"pi_load_{j}.pt"))
            torch.save(optimizerpi.state_dict(), os.path.join(loadopt,f"opt_pi_load_{j}.pt"))
            torch.save(Qvalue.state_dict(), os.path.join(loadpath,f"q_load_{j}.pt"))
            torch.save(optimizerQ.state_dict(), os.path.join(loadopt,f"opt_q_load_{j}.pt"))

        if j%1000==0 and j>2000:
            list_retour, iterations = testfunc(p,env, epsilon)
            retour_episodes.append(list_retour[0])
            print("plot")
            plt.figure()
            plt.semilogy(retour_episodes, label="retour episode pour l'etat initial")
            plt.legend()
            plt.savefig("recompense")
            plt.close()

            plt.figure()
            plt.semilogy(listLosspi, label="negativ pseudo loss")
            plt.legend()
            plt.savefig("negatigpseudoloss")
            plt.close()

            plt.figure()
            plt.plot(listLossQ, label="Q loss")
            plt.legend()
            plt.savefig("Qloss")
            plt.close()
        #return 0
    return listLosspi, retour_episodes



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
