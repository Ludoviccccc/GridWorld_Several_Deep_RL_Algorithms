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
class Tool:
    def __init__(self,env):
        self.rep_ac = Representation_action(env.Na)
        self.rep_cl = Representation(env.Nx,env.Ny)
class Mouse(Tool):
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
        super(Mouse,self).__init__(env)
        self.p = policy2(env)
        self.q = Q2(env)
        self.q_target = Q2(env)
        self.optimizerpi = optim.Adam(self.p.parameters(),lr=   lr_pi)
        self.optimizer_q = optim.Adam(self.q.parameters(), lr = lr_q)
        self.epsilon = epsilon
        self.tau = tau
        self.Na = env.Na
        self.loadpath = loadpath
        self.optpath = optpath
        self.buffer = Buffer(maxsize=buffer_size)
    def Qf(self,state:np.ndarray,action:list[int]):
        if state.ndim ==1:
            return self.q(torch.Tensor(state).unsqueeze(0),self.rep_ac(action))
        else:
            return self.q(torch.Tensor(state),self.rep_ac(action))
    def Qf_target(self,state:np.ndarray,action:int):
        print('action', action)
        if state.ndim ==1:
            return self.q_target(torch.Tensor(state).unsqueeze(0),self.rep_ac([action])).detach()
        else:
            return self.q_target(torch.Tensor(state),self.rep_ac(action)).detach()

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
                api:int,
                 logpi,
                 s):
        self.optimizerpi.zero_grad()
        advantage = self.q(s,self.rep_ac(api)).squeeze().detach()
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
        loss = F.mse_loss(self.Qf(states1,actions1).squeeze(),targets.squeeze())
        loss.backward()
        self.optimizer_q.step()
        return loss
    def epsilon_greedy_policy(self,
                            state:int):
        if np.random.binomial(1,self.epsilon):
            out = np.random.randint(0,self.Na)
        else:
            out = int(self.p(state).detach().item())
        return out 
    def __call__(self,state:np.ndarray):
        state = torch.Tensor(state).unsqueeze(0)
        return self.epsilon_greedy_policy(state)
class Cat(Tool):
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
        super(Cat,self).__init__(env)
        self.p = policy2(env)
        self.q = Q(env)
        self.q_target = Q(env)
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
    def Qf(self,state:np.ndarray,actions:list[np.ndarray]):
        if state.ndim ==1:
            return self.q(torch.Tensor(state).unsqueeze(0),self.rep_ac([actions[0]]),self.rep_ac([actions[1]]))
        else:
            return self.q(torch.Tensor(state),self.rep_ac(actions[0]),self.rep_ac(actions[1]))
    def Qf_target(self,state:np.ndarray,actions:int):
        print('actions', actions)
        if state.ndim ==1:
            return self.q_target(torch.Tensor(state).unsqueeze(0),self.rep_ac(actions[0]),self.rep_ac(actions[1])).detach()
        else:
            return self.q_target(torch.Tensor(state),self.rep_ac(actions[0]),self.rep_ac(actions[1])).detach()

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
                 state):
        self.optimizerpi.zero_grad()
        advantage = self.Qf(state,[api0,api1]).squeeze()
        advantage = advantage.detach()
        NegativPseudoLoss = torch.mean(torch.mul(logpi.squeeze(),advantage)) 
        NegativPseudoLoss.backward()
        self.optimizerpi.step()
        return NegativPseudoLoss
    def updateQ(self,
                states0,
                actions0, 
                actions1, 
                targets):  
        self.optimizer_q.zero_grad()
        loss = F.mse_loss(self.q(states0,actions0,actions1).squeeze(),targets.squeeze())
        loss.backward()
        self.optimizer_q.step()
        return loss
    def epsilon_greedy_policy(self,state:np.ndarray):
        if np.random.binomial(1,self.epsilon):
            out = np.random.randint(0,self.Na)
        else:
            print("state", state)
            out = self.p(state)
            print("out", out)
            out = int(out.detach().item())
        return out 
    def __call__(self,state:np.ndarray):
        state = torch.Tensor(state).unsqueeze(0)
        return self.epsilon_greedy_policy(state)
