import torch
import torch.nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np

def qlearn(agent, Qvalue,optimizerQ,env, n_episodes, loadpath,loadopt,gamma = .9, freqsave=100, epsilon = 0.1,  start = 0):
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
            #a = policy(s)
            a = agent.amax_epsilon(Qvalue,s)
            sp,r = env.transition(a[0],s)
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
                print("Return for last trial", retour_episodes[-1])
                #print(retour_episodes)
        if j%freqsave==0:
            torch.save(Qvalue.state_dict(), os.path.join(loadpath,f"q_load_{j}.pt"))
            torch.save(optimizerQ.state_dict(), os.path.join(loadopt,f"opt_q_load_{j}.pt"))

        if j%100==0 and j>1000:
            list_retour,list_Q, nb = test(agent,Qvalue, env, epsilon)
            retour = list_retour[0]
            #retour, nb = test(Qvalue,env, epsilon)
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

def test(agent, Qvalue, env, epsilon, gamma = .9,plot = False, graph = False):
    #la fonction renvoie le retour, nombre iterations pour finir
    i = 0
    s = torch.randint(0,env.Nx*env.Ny,(1,)).item()
    idx = torch.randint(0,env.Nx*env.Ny-len(env.obstacles_encod),(1,)).item()
    s = [a for a in range(env.Nx*env.Ny) if a not in env.obstacles_encod][idx]
    list_recompense = []
    list_Q = []
    if plot:
        if graph:
            env.grid(s,name=os.path.join("image",str(i)))
    while True:
        a = agent.amax_epsilon(Qvalue,[s])[0]

        sp,R = env.transition(a,s)
        s = sp
        if plot:
            env.grid(s,name=os.path.join("image",str(i)))
        list_recompense.append(R.item()*(gamma**i))
        list_Q.append(Qvalue([s],[a]).item())
        if s==env.G:
            break
        i+=1
    list_retour = [sum(list_recompense[i:]) for i in range(len(list_recompense))]
    return list_retour,list_Q, i

