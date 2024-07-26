import torch
import torch.nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np


def updateQ(s_chat,s_souris,a,r,sp_chat,sp_souris,Qvalue,optimizerQ,env):
    Qvec = Qvalue(env.representation([sp_souris]*env.Na,[sp_chat]*env.Na), env.representationaction(torch.arange(env.Na))).squeeze()
    target = r + env.gamma*torch.max(Qvec).detach()
    optimizerQ.zero_grad()
    loss = (Qvalue(env.representation([s_souris],[s_chat]), env.representationaction(a)).squeeze() - target[0])**2
    loss.backward()
    optimizerQ.step()
    return loss.item()

def qlearn(Qvalue_chat,Qvalue_souris,optimizerQ_chat,optimizerQ_souris,env, n_epochs, loadpath,loadopt, freqsave=100, epsilon = 0.1, K = 1, start = 0):
    listLossQ_chat = []
    listLossQ_souris = []
    recompense_episodes = []
    recompense_chat = []
    recompense_souris = []
    for j in range(start,n_epochs+1):
        s_souris = torch.randint(0,env.Nx*env.Ny,(1,))
        #print("s_souris", s_souris)
        s_chat = torch.randint(0,env.Nx*env.Ny,(1,))
        #step 1
        if torch.bernoulli(torch.Tensor([epsilon])):
            #print("random action")
            a_chat = torch.randint(0,env.Na,(1,)) 
        else:
            #print("Q action")
            a_chat = torch.argmax(Qvalue_chat(env.representation([s_souris]*env.Na,[s_chat]*env.Na), env.representationaction(torch.arange(env.Na))).squeeze()).reshape((1,))

        if torch.bernoulli(torch.Tensor([epsilon])):
            #print("random action")
            a_souris = torch.randint(0,env.Na,(1,)) 
        else:
            #print("Q action")
            a_souris = torch.argmax(Qvalue_souris(env.representation([s_souris]*env.Na, [s_chat]*env.Na), env.representationaction(torch.arange(env.Na))).squeeze()).reshape((1,))
        sp_chat = env.transition_chat(a_chat,s_chat,s_souris)
        sp_souris = env.transition_souris(a_souris,s_souris,sp_chat)
        r_chat = env.reward_souris(sp_chat,s_souris) 
        r_souris = env.reward_souris(sp_souris,sp_chat) 
        #step 2
        for i in range(K):
            #step 3 Q update
            loss = updateQ(s_chat,s_souris,a_chat,r_chat,sp_chat,sp_souris,Qvalue_chat,optimizerQ_chat,env)
            listLossQ_chat.append(loss)
            loss = updateQ(s_chat,s_souris,a_souris,r_souris,sp_chat,sp_souris,Qvalue_souris,optimizerQ_souris,env)
            listLossQ_souris.append(loss)

        if j%100==0:
            print("epochs", j,f"/{n_epochs}")
            print("Loss Q souris", torch.mean(torch.Tensor(listLossQ_souris)))
            print("Loss Q chat", torch.mean(torch.Tensor(listLossQ_chat)))
            if len(recompense_episodes)>0:
                print("moyenne nombre iterations", np.mean(recompense_episodes))
        if j%500==0:
            torch.save(Qvalue_chat.state_dict(), os.path.join(loadpath,f"q_chat_load_{j}.pt"))
            torch.save(Qvalue_souris.state_dict(), os.path.join(loadpath,f"q_souris_load_{j}.pt"))
            torch.save(optimizerQ_chat.state_dict(), os.path.join(loadopt,f"opt_q_chat_load_{j}.pt"))
            torch.save(optimizerQ_souris.state_dict(), os.path.join(loadopt,f"opt_q_souris_load_{j}.pt"))

        if j%5000==0 and j>2000:
            m,rewardlist_chat, rewardlist_souris =  test(Qvalue_chat,
                                                     Qvalue_souris,
                                                     env, 
                                                     epsilon,
                                                     plot =True)

            recompense_episodes.append(m)
            recompense_chat.append(r_chat.item())
            recompense_souris.append(r_souris.item())

            print("plot")
            plt.figure()
            plt.plot(recompense_episodes, label="nombre d'iterations pour que l'agent reuisse")
            plt.legend()
            plt.savefig("recompense")
            plt.close()


            plt.figure()
            plt.plot(recompense_chat, label="nombre d'iterations pour que le chat reuisse")
            plt.legend()
            plt.savefig("recompense_chat")
            plt.close()


            plt.figure()
            plt.plot(recompense_souris, label="nombre d'iterations pour que la souris reuisse")
            plt.legend()
            plt.savefig("recompense_souris")
            plt.close()


            plt.figure()
            plt.plot(listLossQ_chat, label="Q chat loss")
            plt.legend()
            plt.savefig("Qlosschat")
            plt.close()

            plt.figure()
            plt.plot(listLossQ_chat, label="Q chat loss")
            plt.legend()
            plt.savefig("Qlosssouris")
            plt.close()
    return  recompense_episodes

def test(Qvalue_chat,Qvalue_souris, env, epsilon, plot = False):
    print("test")
    i = 0
    s_souris = torch.randint(0,env.Nx*env.Ny,(1,)).item()
    s_chat = torch.randint(0,env.Nx*env.Ny,(1,)).item()
    rewardlist_chat = []
    rewardlist_souris = []
    while True:
        if torch.bernoulli(torch.Tensor([epsilon])):
            a_chat = torch.randint(0,env.Na,(1,)) 
        else:
            a_chat = torch.argmax(Qvalue_chat(env.representation([s_souris]*env.Na,[s_chat]*env.Na), env.representationaction(torch.arange(env.Na))).squeeze()).reshape((1,))

        if torch.bernoulli(torch.Tensor([epsilon])):
            a_souris = torch.randint(0,env.Na,(1,)) 
        else:
            a_souris = torch.argmax(Qvalue_souris(env.representation([s_souris]*env.Na, [s_chat]*env.Na), env.representationaction(torch.arange(env.Na))).squeeze()).reshape((1,))


        sp_chat = env.transition_chat(a_chat,s_chat,s_souris)
        sp_souris = env.transition_souris(a_souris,s_souris,sp_chat)
        r_chat = env.reward_souris(sp_chat,s_souris) 
        r_souris = env.reward_souris(sp_souris,sp_chat) 
##########
        s_souris = sp_souris
        s_chat = sp_chat
        i+=1
        if plot:
            env.grid(s_souris,s_chat)
        rewardlist_chat.append(r_chat)
        rewardlist_souris.append(r_souris)
        if s_souris==s_chat:
            break
    return i,np.sum(rewardlist_chat), np.sum(rewardlist_souris)
