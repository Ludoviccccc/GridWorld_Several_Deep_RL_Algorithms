import torch
import torch.nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np


def updateQ(s_chat,s_souris,a,r,sp_chat,sp_souris,Qvalue,optimizerQ,env):
    Qvec = Qvalue([sp_souris]*env.Na,[sp_chat]*env.Na, torch.arange(env.Na))
    target = r + env.gamma*torch.max(Qvec).detach()
    optimizerQ.zero_grad()
    loss = (Qvalue([s_souris],[s_chat], a) - target[0])**2
    loss.backward()
    optimizerQ.step()
    return loss.item()

def qlearn(Qvalue_chat,Qvalue_souris,optimizerQ_chat,optimizerQ_souris,env, n_episodes, loadpath,loadopt, freqsave=100, epsilon = 0.1, start = 0):
    listLossQ_chat = []
    listLossQ_souris = []
    recompense_episodes = []
    recompense_chat = []
    recompense_souris = []
    for j in range(start,n_episodes+1):
        s_souris = torch.randint(0,env.Nx*env.Ny,(1,))
        s_chat = torch.randint(0,env.Nx*env.Ny,(1,))
        k = 0
        while True:
            if torch.bernoulli(torch.Tensor([epsilon])):
                a_chat = torch.randint(0,env.Na,(1,)) 
            else:
                a_chat = Qvalue_chat.argmax(s_souris,s_chat)

            if torch.bernoulli(torch.Tensor([epsilon])):
                a_souris = torch.randint(0,env.Na,(1,)) 
            else:
                a_souris = Qvalue_souris.argmax(s_souris,s_chat)

            sp_chat = env.transition_chat(a_chat,s_chat,s_souris)
            sp_souris = env.transition_souris(a_souris,s_souris,sp_chat)
            r_chat = env.reward_souris(sp_chat,s_souris) 
            r_souris = env.reward_souris(sp_souris,sp_chat) 

            loss = updateQ(s_chat,s_souris,a_chat,  r_chat,  sp_chat,sp_souris,Qvalue_chat,optimizerQ_chat,env)
            loss = updateQ(s_chat,s_souris,a_souris,r_souris,sp_chat,sp_souris,Qvalue_souris,optimizerQ_souris,env)
            k +=1
            if sp_souris==sp_chat:
                if j%5==0:
                    print(f"episod {j} finished with {k} iterations")
                break
            #env.grid(s_souris,s_chat)
            s_chat = sp_chat
            s_souris = sp_souris

            listLossQ_souris.append(loss)
            listLossQ_chat.append(loss)

        if j%100==0:
            print("episodes", j,f"/{n_episodes}")
            print("Loss Q souris", torch.mean(torch.Tensor(listLossQ_souris)))
            print("Loss Q chat", torch.mean(torch.Tensor(listLossQ_chat)))
            if len(recompense_episodes)>0:
                print("moyenne nombre iterations", np.mean(recompense_episodes))
        if j%20==0:
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
            #print("random action")
            a_chat = torch.randint(0,env.Na,(1,)) 
        else:
            #print("Q action")
            a_chat = Qvalue_chat.argmax(s_souris,s_chat)
        
        if torch.bernoulli(torch.Tensor([epsilon])):
            #print("random action")
            a_souris = torch.randint(0,env.Na,(1,)) 
        else:
            #print("Q action")
            a_souris = Qvalue_souris.argmax(s_souris,s_chat)

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
