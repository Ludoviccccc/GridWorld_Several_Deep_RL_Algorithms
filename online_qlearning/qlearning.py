import torch
import torch.nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np

def qlearn(Qvalue,optimizerQ,env, n_epochs, loadpath,loadopt, freqsave=100, epsilon = 0.1, K = 1, start = 0):
    listLossQ = []
    recompense_episodes = []
    nb_iteration_episodes = []
    print("K", K)
    print("freqsave", freqsave)
    for j in range(start,n_epochs+1):

        #step 1
        s = torch.randint(0,env.Nx*env.Ny,(1,))
        if torch.bernoulli(torch.Tensor([epsilon])):
            a = torch.randint(0,env.Na,(1,)) 
        else:
            a = torch.argmax(Qvalue(env.representation([s]*env.Na), env.representationaction(torch.arange(env.Na))).squeeze()).reshape((1,))
        sp,r = env.transition(a,s)
        #step 2
        for i in range(K):
            #step 3 Q update
            Qvec = Qvalue(env.representation([sp]*env.Na), env.representationaction(torch.arange(env.Na))).squeeze()
            target = r + env.gamma*torch.max(Qvec).detach()
            optimizerQ.zero_grad()
            loss = (Qvalue(env.representation(s),env.representationaction(a)).squeeze() - target[0])**2
            loss.backward()
            optimizerQ.step()
            listLossQ.append(loss.detach().to("cpu"))
        if j%500==0:
            print("epochs", j,f"/{n_epochs}")
            print("Loss Q", torch.mean(torch.Tensor(listLossQ)))
            if len(recompense_episodes)>0:
                #print("recompense", np.mean(recompense_episodes))
                print("dernier retour", recompense_episodes[-1])
                #print(recompense_episodes)
        if j%freqsave==0:
            torch.save(Qvalue.state_dict(), os.path.join(loadpath,f"q_load_{j}.pt"))
            torch.save(optimizerQ.state_dict(), os.path.join(loadopt,f"opt_q_load_{j}.pt"))

        if j%100==0 and j>1000:
            retour, nb = test(Qvalue,env, epsilon)
            recompense_episodes.append(retour)
            nb_iteration_episodes.append(nb)
            print("plot")
            plt.figure()
            plt.plot(recompense_episodes, label="retour")
            plt.legend()
            plt.savefig("retour")
            plt.close()

            plt.figure()
            plt.semilogy(nb_iteration_episodes, label="nb iteration")
            plt.legend()
            plt.savefig("nb")
            plt.close()

            plt.figure()
            plt.plot(listLossQ, label="Q loss")
            plt.legend()
            plt.savefig("Qloss")
            plt.close()

def test(Qvalue, env, epsilon, plot = False):
    #la fonction renvoie le retour, nombre iterations pour finir
    i = 0
    s = torch.randint(0,env.Nx*env.Ny,(1,)).item()
    rewardlist = []
    while True:
        if torch.bernoulli(torch.Tensor([epsilon])):
            a = torch.randint(0,env.Na,(1,)) 
        else:
            a = torch.argmax(Qvalue(env.representation([s]*env.Na), env.representationaction(torch.arange(env.Na))).squeeze())
        sp,R = env.transition(a,s)
        s = sp
        i+=1
        if plot:
            env.grid(s)
        rewardlist.append(R)
        if s==env.G:
            break
    #print(rewardlist)
    #print("sortie", sum(rewardlist))
    return sum(rewardlist),i
