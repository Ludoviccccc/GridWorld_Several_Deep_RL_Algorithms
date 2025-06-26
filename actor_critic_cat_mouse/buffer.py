import torch
class Buffer:
    def __init__(self,maxsize=150):
        self.memory_state = {"cat":[],"mouse":[]}
        self.memory_action = {"cat":[],"mouse":[]}
        self.memory_newstate = {"cat":[],"mouse":[]}
        self.memory_reward = []
        self.maxsize = maxsize
    def store(self,pair):
        self.memory_state["cat"].append(pair["state"]["cat"])
        self.memory_state["mouse"].append(pair["state"]["mouse"])
        self.memory_action["cat"].append(pair["action"]["cat"])
        self.memory_action["mouse"].append(pair["action"]["mouse"])
        self.memory_newstate["cat"].append(pair["new_state"]["cat"])
        self.memory_newstate["mouse"].append(pair["new_state"]["mouse"])
        self.memory_reward.append(pair["reward"])
        self.eviction()
    def sample(self,N):
        """
        returns dictionnary of state, action, new_state and reward
        """
        assert(type(N)==int and N>0 and N<=len(self.memory_state))
        selection = torch.randint(0,len(self.memory_state["mouse"]),(N,))
        state = {"cat":torch.Tensor([self.memory_state["cat"][j] for j in selection]),
                "mouse":torch.Tensor([self.memory_state["mouse"][j] for j in selection])}
        action = {"cat":torch.Tensor([self.memory_action["cat"][j] for j in selection]),
                "mouse":torch.Tensor([self.memory_action["mouse"][j] for j in selection])}
        new_state = {"cat":torch.Tensor([self.memory_newstate["cat"][j] for j in selection]),
                "mouse":torch.Tensor([self.memory_newstate["mouse"][j] for j in selection])}
        reward = torch.Tensor([self.memory_reward[j] for j in selection])
        return {"state":state, "action":action, "new_state":new_state,"reward": reward}


    def eviction(self):
        if len(self.memory_state)>self.maxsize:
            self.memory_state["cat"] = self.memory_state["cat"][-self.maxsize:]
            self.memory_state["mouse"] = self.memory_state["mouse"][-self.maxsize:]
            self.memory_action["cat"] = self.memory_action["cat"][-self.maxsize:]
            self.memory_action["mouse"] = self.memory_action["mouse"][-self.maxsize:]
            self.memory_new_state["cat"] = self.memory_new_state["cat"][-self.maxsize:]
            self.memory_new_state["mouse"] = self.memory_new_state["mouse"][-self.maxsize:]
            self.memory_reward = self.memory_reward[-self.maxsize:]
