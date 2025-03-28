import torch
class Buffer:
    def __init__(self):
        self.memory_state = []
        self.memory_action = []
        self.memory_newstate = []
        self.memory_reward = []
    def store(self,states,actions,new_state,reward):
        for j in range(len(states)):
            self.memory_state.append(states[j])
            self.memory_action.append(actions[j])
            self.memory_newstate.append(new_state[j])
            self.memory_reward.append(reward[j])
    def sample(self,N):
        assert(type(N)==int and N>0 and N<=len(self.memory_state))
        selection = torch.randint(0,len(self.memory_state),(N,))
        state = torch.stack([self.memory_state[j] for j in selection])    
        action = torch.stack([self.memory_action[j] for j in selection])                                                           
        newstate = torch.stack([self.memory_newstate[j] for j in selection])
        reward = torch.stack([self.memory_reward[j] for j in selection])
        #renvoie un tuple de 4 tenseurs (s,a,s',r)
        return state, action, newstate, reward

