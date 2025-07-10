from policy import policy, policy2
from Qfunc import Q,Q2
import torch.optim as optim
import torch.nn.functional as F
import torch
from rep import Representation_action, Representation
from env2 import grid
import numpy as np
import os
from buffer import Buffer
from torch.nn.utils import clip_grad_norm_
class Tool:
    def __init__(self,env):
        self.rep_ac = Representation_action(env.Na)
        self.rep_cl = Representation(env.Nx,env.Ny)
    def Qf(self,state:dict,actions:dict):
        return self.q(self.rep_cl(state["cat"]),self.rep_cl(state["mouse"]),self.rep_ac(actions["cat"]),self.rep_ac(actions["mouse"]))
    def Qf_target(self,state:dict,actions:dict):
        return self.q_target(self.rep_cl(state["cat"]),self.rep_cl(state["mouse"]),self.rep_ac(actions["cat"]),self.rep_ac(actions["mouse"]))
        #return self.q_target(state["cat"],state["mouse"],self.rep_ac(actions["cat"]),self.rep_ac(actions["mouse"]))
    def update_target_net(self):
        for (name_q, param_q),(name_q_target,param_q_target) in zip(self.q.state_dict().items(),self.q_target.state_dict().items()):
            param_q_target.copy_(self.tau*param_q + (1.0 - self.tau)*param_q_target)
    def updatePi(self,
                 api:dict,
                 logpi,
                 state:dict):
#        self.optimizerpi.zero_grad()
        advantage = self.Qf(state,api).squeeze()
        advantage = advantage.detach()
        NegativPseudoLoss = torch.mean(torch.mul(logpi.squeeze(),advantage)) 
        NegativPseudoLoss.backward()
        self.optimizerpi.step()
        return NegativPseudoLoss
    def updateQ(self,
                states:dict,
                actions:dict,
                targets):  
        for k in range(self.k):
            self.optimizer_q.zero_grad()
            loss = F.mse_loss(self.Qf(states,actions).squeeze(),targets.squeeze())
            loss.backward()
            self.optimizer_q.step()
        return loss
    def epsilon_greedy_policy(self,state:dict):
        if np.random.binomial(1,self.epsilon):
            out = np.random.randint(0,self.Na)
        else:
            out = self.p([state["cat"]],[state["mouse"]])
            out = int(out.detach().item())
        return out 
    def __call__(self,state:dict):
        return self.epsilon_greedy_policy(state)
class Mouse(Tool):
    def __init__(self,
                env:grid,
                lr_pi:float=.3,
                lr_q:float=.3,
                epsilon:float=.1,
                tau:float=0.01,
                loadpath:str = "loads",
                optpath:str ="opt",
                buffer_size:int=10000,
                K:int=1
                ):
        super(Mouse,self).__init__(env)
        self.p = policy(env)
        self.q = Q(env)
        self.q_target = Q(env)
        for (name_q, param_q),(name_q_target,param_q_target) in zip(self.q.state_dict().items(),self.q_target.state_dict().items()):
            param_q_target.copy_(param_q)
        self.optimizerpi = optim.Adam(self.p.parameters(),lr=   lr_pi)
        self.optimizer_q = optim.Adam(self.q.parameters(), lr = lr_q)
        #clip_grad_norm_(self.p.parameters(),1e-4)
        self.epsilon = epsilon
        self.tau = tau
        self.Na = env.Na
        self.loadpath = loadpath
        self.optpath = optpath
        self.buffer = Buffer(maxsize=buffer_size)
        self.k = K
    def load(self,start:int):
        self.q.load_state_dict(torch.load(os.path.join(self.loadpath,f"q_1_load_{start}.pt"),weights_only=True))
        self.p.load_state_dict(torch.load(os.path.join(self.loadpath,f"pi_1_load_{start}.pt"),weights_only=True))
    def save(self,j):
        torch.save(self.p.state_dict(), os.path.join(self.loadpath,f"pi_1_load_{j}.pt"))
        torch.save(self.optimizerpi.state_dict(), os.path.join(self.optpath,f"opt_1_pi_load_{j}.pt"))
        torch.save(self.q.state_dict(), os.path.join(self.loadpath,f"q_1_load_{j}.pt"))
        torch.save(self.optimizer_q.state_dict(), os.path.join(self.optpath,f"opt_1_pi_load_{j}.pt"))
class Cat(Tool):
    def __init__(self,
                env:grid,
                lr_pi:float=.3,
                lr_q:float=.3,
                epsilon:float=.1,
                tau:float=0.01,
                loadpath:str = "loads",
                optpath:str ="opt",
                buffer_size:int=10000,
                K:int=1
                ):
        super(Cat,self).__init__(env)
        self.k = K
        self.p = policy(env)
        self.q = Q(env)
        self.q_target = Q(env)
        for (name_q, param_q),(name_q_target,param_q_target) in zip(self.q.state_dict().items(),self.q_target.state_dict().items()):
            param_q_target.copy_(param_q)
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
