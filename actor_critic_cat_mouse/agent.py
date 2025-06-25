from policy import policy, policy2
from Qfunc import Q,Q2
import torch.optim as optim

class mouse:
    def __init__(policy:policy2,q:Q2,lr_pi=.3,lr_q=.3):
        self.policy = policy
        self.q = q
        self.optimizerpi = optim.Adam(self.policy.parameters(),lr=lr_pi)
        self.optimizer_q = optim.Adam(self.q.parameters(), lr = lr_q)
    def updatePi(api0:int,
                 api1:int,
                 logpi,
                 s_tab:list):
        self.optimizerpi.zero_grad()
        advantage = self.q(s_tab[0],s_tab[1],rep_ac(api0),rep_ac(api1)).squeeze().detach()
        advantage = advantage.detach()
        NegativPseudoLoss = torch.mean(torch.mul(logpi.squeeze(),advantage)) 
        NegativPseudoLoss.backward()
        self.optimizerpi.step()
        return NegativPseudoLoss
    def updateQ(states0,
                states1,
                actions0, 
                actions1, 
                targets):  
        self.optimizerQ.zero_grad()
        loss = F.mse_loss(self.q(states0,states1,actions0,actions1).squeeze(),targets.squeeze())
        loss.backward()
        optimizerQ.step()
        return loss
