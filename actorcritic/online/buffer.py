import torch
class Buffer:
    def __init__(self, maxsize = 10000):
        self.memory_state = []
        self.memory_action = []
        self.memory_newstate = []
        self.memory_reward = []
        self.maxsize = maxsize
    def store(self,sample):
        for j in range(len(sample["state"])):
            self.memory_state.append(sample["state"][j])
            self.memory_action.append(sample["action"][j])
            self.memory_newstate.append(sample["new_state"][j])
            self.memory_reward.append(sample["reward"][j])
        self.eviction()
    def eviction(self):
        if len(self.memory_state)>self.maxsize:
            self.memory_state = self.memory_state[-self.maxsize:]
            self.memory_action = self.memory_action[-self.maxsize:]
            self.memory_newstate = self.memory_newstate[-self.maxsize:]
            self.memory_reward = self.memory_reward[-self.maxsize:]

    def sample(self,N):
        assert(type(N)==int and N>0 and N<=len(self.memory_state))
        selection = torch.randint(0,len(self.memory_state),(N,))
        #print("selection", selection)
        state = torch.Tensor([int(self.memory_state[j]) for j in selection])    
        action = torch.Tensor([int(self.memory_action[j]) for j in selection])                                                           
        newstate = torch.Tensor([int(self.memory_newstate[j]) for j in selection])
        reward = torch.Tensor([int(self.memory_reward[j]) for j in selection])
        #renvoie un tuple de 4 tenseurs (s,a,s',r)
        sample = {"state": state,
                  "action": action,
                  "new_state": newstate,
                  "reward": reward
                  }
        #return state, action, newstate, reward
        return sample
    def print (self):

        print("self.memory_state    "      ,self.memory_state )
        print("self.memory_action   "      ,self.memory_action )
        print("self.memory_newstate "      ,self.memory_newstate) 
        print("self.memory_reward   "      ,self.memory_reward )

