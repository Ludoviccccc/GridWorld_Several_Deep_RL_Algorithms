from policy import policy, policy2
from Qfunc import Q,Q2
import torch.optim as optim
import torch.nn.functional as F
import torch
from rep import Representation_action, Representation
class mouse:
    def __init__(self,
                policy:policy2,
                q:Q2,
                lr_pi=.3,
                lr_q=.3,
                epislon=.1,
                Na = 8):
        self.p = policy
        self.q = q
        self.optimizerpi = optim.Adam(self.policy.parameters(),lr=lr_pi)
        self.optimizer_q = optim.Adam(self.q.parameters(), lr = lr_q)
        self.rep_ac = Representation_action()
        self.rep_cl = Representation()
        self.epsilon
        self.Na = Na
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
    def epsilon_greedy_policy(self,
                            state_vec:list,
                            rep_cl:Representation):
        if np.random.binomial(1,self.epsilon):
            out = np.random.randint(0,self.Na)
        else:
            out = int(self.p(self.rep_cl([state_vec[0]]),self.rep_cl([state_vec[1]])).detach().item())
        return out 
class cat:
    def __init__(self,
                p:policy,
                q:Q,
                lr_pi=.3,
                lr_q=.3,
                epislon=.1,
                Na = 8):
        self.p = p
        self.q = q
        self.optimizerpi = optim.Adam(self.policy.parameters(),lr=lr_pi)
        self.optimizer_q = optim.Adam(self.q.parameters(), lr = lr_q)
        self.rep_ac = Representation_action()
        self.rep_cl = Representation()
        self.epsilon
        self.Na = Na
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
    def epsilon_greedy_policy(self,
                            state_vec:list,
                            rep_cl:Representation):
        if np.random.binomial(1,self.epsilon):
            out = np.random.randint(0,self.Na)
        else:
            out = int(self.p(self.rep_cl([state_vec[0]]),self.rep_cl([state_vec[1]])).detach().item())
        return out 
