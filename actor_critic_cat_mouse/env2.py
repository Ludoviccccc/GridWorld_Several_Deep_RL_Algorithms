import numpy as np

class grid:
    """
    Envirnonement grille sur laquelle se deplace l'agent jusqu'à atteindre le point G
    """
    def __init__(self,Nx,Ny,max_steps = 100):
        self.actions = [(0,1), (0, -1), (1, 0), (-1, 0),(1,1),(1,-1),(-1,-1),(-1,1)]
        self.Na = len(self.actions)
        self.Nx = Nx
        self.Ny = Ny
        self.cat_pos = (np.random.randint(0,self.Nx), np.random.randint(0,self.Ny))
        self.mouse_pos = (np.random.randint(0,self.Nx), np.random.randint(0,self.Ny))
        self.target_idx = 10
        self.target_mouse = (self.target_idx//self.Ny,self.target_idx%self.Ny)
        self.max_steps = max_steps
    def reset(self):
        self.count = 0
        self.catch = 0
        self.cat_pos = (np.random.randint(0,self.Nx), np.random.randint(0,self.Ny))
        self.mouse_pos = (np.random.randint(0,self.Nx), np.random.randint(0,self.Ny))
    def state_cat(self):
        state_cat = np.array([self.mouse_pos[0] - self.cat_pos[0],self.mouse_pos[1] - self.cat_pos[1]])
        if np.linalg.norm(state_cat)!=0:
            state_cat = state_cat/np.linalg.norm(state_cat)
        return state_cat
    def transition_cat(self,a:int):
        self.cat_previous = self.cat_pos
        self.cat_pos = self.transition_single_agent(self.cat_pos,a) 
        reward = self.reward_cat()
        state_cat = self.state_cat()
        self.catch+=1.0*(self.cat_pos==self.mouse_pos)
        return state_cat,reward
    def state_mouse(self):
        state_mouse = np.array([self.target_mouse[0] - self.mouse_pos[0],self.target_mouse[1] - self.mouse_pos[1]])
        if np.linalg.norm(state_mouse)!=0:
            state_mouse = state_mouse/np.linalg.norm(state_mouse)
        return state_mouse
    def transition_mouse(self,a:int):
        self.mouse_previous = self.mouse_pos
        self.mouse_pos = self.transition_single_agent(self.mouse_pos,a) 
        reward = self.reward_mouse()
        terminated = self.terminated()
        self.count+=1
        state_mouse = self.state_mouse()
        return state_mouse,reward
    def terminated(self):
        "says if the episode is teminated"
        cheese_reached = self.mouse_pos==self.target_mouse
        terminated = cheese_reached or self.catch==10
        return terminated  
    def truncated(self):
        return self.count>self.max_steps
    def transition_single_agent(self,agent_pos:tuple,a):
        assert(0<=a<=self.Na)
        d = self.actions[a]
        if self.Nx>agent_pos[0]+ d[0]>=0 and self.Ny>agent_pos[1]+d[1]>=0:
            sp = (agent_pos[0]+ d[0], agent_pos[1]+d[1])
            assert(0<=sp[0]*self.Ny+sp[1]<self.Nx*self.Ny)
            s_out = sp
        else:
            s_out = agent_pos
        return s_out
    def reward_cat(self):
        self.catch+=1*(self.cat_pos==self.mouse_pos)
        reward = (10.0)*(self.cat_pos==self.mouse_pos)  + (-1.0)*(self.cat_pos ==self.cat_previous) 
        return reward
    def reward_mouse(self):
        reward = (100.0)*(self.target_mouse==self.mouse_pos) + (-10.0)*(self.mouse_pos==self.mouse_previous)
        return reward
    def grid(self):
        s_mouse = self.mouse
        s_cat = self.cat
        T = np.zeros((self.Nx,self.Ny))
        T[s_mouse//self.Ny, s_mouse%self.Ny] = 1
        T[s_cat//self.Ny, s_cat%self.Ny] = -1
        T[self.target_mouse[0],self.target_mouse[1]] = 5
        print(T)
        return T
