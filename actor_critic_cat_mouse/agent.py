from policy import policy, policy2
from Qfunc import Q,Q2
import torch.optim as optim
import torch.nn.functional as F
import torch
from rep import Representation_action, Representation
from env import grid
import numpy as np
import os
from buffer import Buffer
class Mouse:
    def __init__(self,
                env:grid,
                lr_pi:float=.3,
                lr_q:float=.3,
                epsilon:float=.1,
                tau:float=0.01,
                loadpath:str = "loads",
                optpath:str ="opt",
                buffer_size:int=10000
                ):
        self.p = policy2(env.Nx,env.Ny,env.Na)
        self.q = Q2(env.Nx,env.Ny,env.Na)
        self.q_target = Q2(env.Nx,env.Ny,env.Na)
        self.optimizerpi = optim.Adam(self.p.parameters(),lr=   lr_pi)
        self.optimizer_q = optim.Adam(self.q.parameters(), lr = lr_q)
        self.rep_ac = Representation_action(env.Na)
        self.rep_cl = Representation(env.Nx,env.Ny)
        self.epsilon = epsilon
        self.tau = tau
        self.Na = env.Na
        self.loadpath = loadpath
        self.optpath = optpath
        self.buffer = Buffer(maxsize=buffer_size)
    def load(self,start:int):
        self.q.load_state_dict(torch.load(os.path.join(self.loadpath,f"q_1_load_{start}.pt"),weights_only=True))
        self.p.load_state_dict(torch.load(os.path.join(self.loadpath,f"pi_1_load_{start}.pt"),weights_only=True))
    def save(self,j):
        torch.save(self.p.state_dict(), os.path.join(self.loadpath,f"pi_1_load_{j}.pt"))
        torch.save(self.optimizerpi.state_dict(), os.path.join(self.optpath,f"opt_1_pi_load_{j}.pt"))
        torch.save(self.q.state_dict(), os.path.join(self.loadpath,f"q_1_load_{j}.pt"))
        torch.save(self.optimizer_q.state_dict(), os.path.join(self.optpath,f"opt_1_pi_load_{j}.pt"))
    def update_target_net(self):
        for (name_q, param_q),(name_q_target,param_q_target) in zip(self.q.state_dict().items(),self.q_target.state_dict().items()):
            param_q_target.copy_(self.tau*param_q + (1.0 - self.tau)*param_q_target)

    def updatePi(self,
                api1:int,
                 logpi,
                 s):
        self.optimizerpi.zero_grad()
        advantage = self.q(s,self.rep_ac(api1)).squeeze().detach()
        advantage = advantage.detach()
        NegativPseudoLoss = torch.mean(torch.mul(logpi.squeeze(),advantage)) 
        NegativPseudoLoss.backward()
        self.optimizerpi.step()
        return NegativPseudoLoss
    def updateQ(self,
                states1,
                actions1, 
                targets):  
        self.optimizer_q.zero_grad()
        loss = F.mse_loss(self.q(states1,actions1).squeeze(),targets.squeeze())
        loss.backward()
        self.optimizer_q.step()
        return loss
    def epsilon_greedy_policy(self,
                            state:int):
        if np.random.binomial(1,self.epsilon):
            out = np.random.randint(0,self.Na)
        else:
            out = int(self.p(self.rep_cl(state)).detach().item())
        return out 
    def act(self,state:int):
        return self.epsilon_greedy_policy(state)
class Cat:
    def __init__(self,
                env:grid,
                lr_pi:float=.3,
                lr_q:float=.3,
                epsilon:float=.1,
                tau:float=0.01,
                loadpath:str = "loads",
                optpath:str ="opt",
                buffer_size:int=10000
                ):
        self.p = policy(env.Nx,env.Ny,env.Na)
        self.q = Q(env.Nx,env.Ny,env.Na)
        self.q_target = Q(env.Nx,env.Ny,env.Na)
        self.optimizerpi = optim.Adam(self.p.parameters(),lr=lr_pi)
        self.optimizer_q = optim.Adam(self.q.parameters(), lr = lr_q)
        self.rep_ac = Representation_action(env.Na)
        self.rep_cl = Representation(env.Nx,env.Ny)
        self.epsilon = epsilon
        self.tau = tau
        self.Na = env.Na
        self.optpath = optpath
        self.loadpath = loadpath
        self.buffer = Buffer()
    def load(self,start:int):
        self.q.load_state_dict(torch.load(os.path.join(self.loadpath,f"q_0_load_{start}.pt"),weights_only=True))
        self.p.load_state_dict(torch.load(os.path.join(self.loadpath,f"pi_0_load_{start}.pt"),weights_only=True))
    def save(self,j):
        torch.save(self.p.state_dict(), os.path.join(self.loadpath,f"pi_0_load_{j}.pt"))
        torch.save(self.optimizerpi.state_dict(), os.path.join(self.optpath,f"opt_0_pi_load_{j}.pt"))
        torch.save(self.q.state_dict(), os.path.join(self.loadpath,f"q_0_load_{j}.pt"))
        torch.save(self.optimizer_q.state_dict(), os.path.join(self.optpath,f"opt_0_pi_load_{j}.pt"))
    def update_target_net(self):
        for (name_q, param_q),(name_q_target,param_q_target) in zip(self.q.state_dict().items(),self.q_target.state_dict().items()):
            param_q_target.copy_(self.tau*param_q + (1.0 - self.tau)*param_q_target)
    def updatePi(self,
                api0:int,
                 api1:int,
                 logpi,
                 s_tab:list):
        self.optimizerpi.zero_grad()
        advantage = self.q(s_tab[0],s_tab[1],self.rep_ac(api0),self.rep_ac(api1)).squeeze().detach()
        advantage = advantage.detach()
        NegativPseudoLoss = torch.mean(torch.mul(logpi.squeeze(),advantage)) 
        NegativPseudoLoss.backward()
        self.optimizerpi.step()
        return NegativPseudoLoss
    def updateQ(self,
                states0,
                states1,
                actions0, 
                actions1, 
                targets):  
        self.optimizer_q.zero_grad()
        loss = F.mse_loss(self.q(states0,states1,actions0,actions1).squeeze(),targets.squeeze())
        loss.backward()
        self.optimizer_q.step()
        return loss
    def epsilon_greedy_policy(self,state_vec:list):
        if np.random.binomial(1,self.epsilon):
            out = np.random.randint(0,self.Na)
        else:
            out = int(self.p(self.rep_cl([state_vec[0]]),self.rep_cl([state_vec[1]])).detach().item())
        return out 
    def act(self,state_vec:list):
        return self.epsilon_greedy_policy(state_vec)
