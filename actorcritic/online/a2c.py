import torch
import torch.nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

def A2C(buffer,Qvalue,optimizerQ,p,optimizerpi,env,N, batch_size, n_epochs, loadpath,loadopt, freqsave=500, epsilon = 0.1, K = 1, start = 0):
    initial_states = torch.randint(0,env.Nx*env.Ny,(N,))
    #test fonction de politique
    actions = p(env.representation(initial_states))
    #test fonction de transition
    new_state,reward = env.transitionvec(actions,initial_states)
    buffer.store(initial_states,actions,new_state,reward)
    listLosspi = []
    listLossQ = []
    recompense_episodes = []
    for j in range(start,n_epochs+1):
        #step 1
        s = torch.randint(0,env.Nx*env.Ny,(1,))
        if torch.bernoulli(torch.Tensor([epsilon])):
            a = torch.randint(0,env.Na,(1,)) 
        else:
            a = p(env.representation(s))
        sp,r = env.transition(a,s)
        buffer.store(s,torch.Tensor([a]),sp,r)
        #step 2
        s,a,sp,r = buffer.sample(batch_size)
        for i in range(K):
            ap = p(env.representation(sp))
            #step 3 Q update
            targets = r + env.gamma*Qvalue(env.representation(sp),env.representationaction(ap)).squeeze()#targets computation
            optimizerQ.zero_grad()
            loss = F.mse_loss(Qvalue(env.representation(s),env.representationaction(a)).squeeze(),targets)
            loss.backward()
            optimizerQ.step()
            #step 4 and 5
            #computing negativpseudoloss for policy pi

            optimizerpi.zero_grad()
            api = p(env.representation(s)).detach() 
            logpi = F.cross_entropy(p(env.representation(s),logit = True),api,weight = None, reduction = 'none' )
            NegativPseudoLoss = torch.mean(torch.mul(logpi,Qvalue(env.representation(s),env.representationaction(api)).squeeze()))
            NegativPseudoLoss.backward()
            optimizerpi.step()

            listLosspi.append(NegativPseudoLoss.item())
            listLossQ.append(loss.item())

        if j%100==0:
            print("epochs", j,f"/{n_epochs}")
            print("NegativPseudoLoss",torch.mean(torch.Tensor(listLosspi)))
            print("Loss Q", torch.mean(torch.Tensor(listLossQ)))
            if len(recompense_episodes)>0:
                print("moyenne recompense", np.mean(recompense_episodes))
        if j%freqsave==0:
            torch.save(p.state_dict(), os.path.join(loadpath,f"pi_load_{j}.pt"))
            torch.save(optimizerpi.state_dict(), os.path.join(loadopt,f"opt_pi_load_{j}.pt"))
            torch.save(Qvalue.state_dict(), os.path.join(loadpath,f"q_load_{j}.pt"))
            torch.save(optimizerQ.state_dict(), os.path.join(loadopt,f"opt_q_load_{j}.pt"))

        if j%1000==0 and j>5000:
            recompense_episodes.append(test(p,env, epsilon))
            print("plot")
            plt.figure()
            plt.semilogy(recompense_episodes, label="nombre d'iterations pour que l'agent reuisse")
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
    return listLosspi, recompense_episodes



def test(p, env, epsilon):
    i = 0
    s = 5
    rewardlist = []
    while True:
        if torch.bernoulli(torch.Tensor([epsilon])):
            a = torch.randint(0,env.Na,(1,)) 
        else:
            a  = p(env.representation([s]))
        sp,R = env.transition(a,s)
        s = sp
        i+=1
        rewardlist.append(R)
        if s==env.G:
            break
    return i
