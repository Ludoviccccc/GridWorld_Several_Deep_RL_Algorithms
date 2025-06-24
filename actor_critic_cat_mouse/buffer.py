import torch
class Buffer:
    def __init__(self):
        self.memory_state = []
        self.memory_action = []
        self.memory_newstate = []
        self.memory_reward = []
    def store(self,pair):
        for j in range(len(pair["state"])):
            self.memory_state.append(pair["state"][j])
            self.memory_action.append(pair["action"][j])
            self.memory_newstate.append(pair["new_state"][j])
            self.memory_reward.append(pair["reward"][j])
    def sample(self,N):
        assert(type(N)==int and N>0 and N<=len(self.memory_state))
        selection = torch.randint(0,len(self.memory_state),(N,))
        state = torch.stack([torch.Tensor(self.memory_state[j]) for j in selection])    
        action = torch.stack([torch.Tensor(self.memory_action[j]) for j in selection])
        newstate = torch.stack([torch.Tensor(self.memory_newstate[j]) for j in selection])
        reward = torch.Tensor([self.memory_reward[j] for j in selection])
        #renvoie un tuple de 4 tenseurs (s,a,s',r)
        return {"state":state, "action":action, "new_state":newstate,"reward": reward}

