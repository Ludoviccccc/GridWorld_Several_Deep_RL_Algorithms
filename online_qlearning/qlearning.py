import torch
import torch.nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np

def qlearn(Qvalue,optimizerQ,env, n_episodes, loadpath,loadopt,gamma = .9, freqsave=100, epsilon = 0.1,  start = 0):
    def policy(s):
        if torch.bernoulli(torch.Tensor([epsilon])):
            a = torch.randint(0,env.Na,(1,)) 
        else:
            a = Qvalue.argmax(s)
        return a
    listLossQ = []
    retour_episodes = []
    nb_iteration_episodes = []
    print("freqsave", freqsave)
    for j in range(start,n_episodes+1):
        k = 1
        s = torch.randint(0,env.Nx*env.Ny,(1,))
        Retour = 0
        while True:
            a = policy(s)
            sp,r = env.transition(a,s)
            #step 2
            #step 3 Q update
            Qvec = Qvalue([sp]*env.Na,torch.arange(env.Na))
            target = r + gamma*torch.max(Qvec).detach()
            optimizerQ.zero_grad()
            loss = (Qvalue(s,a) - target[0])**2
            loss.backward()
            optimizerQ.step()
            listLossQ.append(loss.detach().to("cpu"))
            k+=1
            if sp==env.G:
                if j%100==0:
                    print(f"episod with {k} iterations")
                break
            Retour += gamma*r
            s = sp
        if j%500==0:
            print("episodes", j,f"/{n_episodes}")
            print("Loss Q", torch.mean(torch.Tensor(listLossQ)))
            if len(retour_episodes)>0:
                print("last reward", retour_episodes[-1])
                #print(retour_episodes)
        if j%freqsave==0:
            torch.save(Qvalue.state_dict(), os.path.join(loadpath,f"q_load_{j}.pt"))
            torch.save(optimizerQ.state_dict(), os.path.join(loadopt,f"opt_q_load_{j}.pt"))

        if j%100==0 and j>1000:
            retour, nb = test(Qvalue,env, epsilon)
            retour_episodes.append(retour)
            nb_iteration_episodes.append(nb)
            print("plot")
            plt.figure()
            plt.plot(retour_episodes, label="retour")
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

def test(Qvalue, env, epsilon, gamma = .9,plot = False):
    #la fonction renvoie le retour, nombre iterations pour finir
    i = 0
    s = torch.randint(0,env.Nx*env.Ny,(1,)).item()
    Retour =0
    while True:
        if torch.bernoulli(torch.Tensor([epsilon])):
            a = torch.randint(0,env.Na,(1,)) 
        else:
            a = Qvalue.argmax(s)
        sp,R = env.transition(a,s)
        s = sp
        i+=1
        if plot:
            env.grid(s)
        Retour +=gamma*R
        if s==env.G:
            break
    return Retour,i
