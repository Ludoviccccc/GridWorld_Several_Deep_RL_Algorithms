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
        self.walls = [2,6,14,17,21,24,27,41,53]
        self.walls = []
        self.target_mouse = (self.target_idx//self.Ny,self.target_idx%self.Ny)
        self.max_steps = max_steps
    def reset(self):
        self.count = 0
        self.catch = 0
        self.cat_pos = (np.random.randint(0,self.Nx), np.random.randint(0,self.Ny))
        self.mouse_pos = (np.random.randint(0,self.Nx), np.random.randint(0,self.Ny))
        while self.terminated() or self._agent_in_wall(self.cat_pos) or self._agent_in_wall(self.mouse_pos):
            self.cat_pos = (np.random.randint(0,self.Nx), np.random.randint(0,self.Ny))
            self.mouse_pos = (np.random.randint(0,self.Nx), np.random.randint(0,self.Ny))
    def state_cat(self):
        return  self.cat_pos[0]*self.Ny+self.cat_pos[1]
    def transition_cat(self,a:int):
        self.cat_previous = self.cat_pos
        self.cat_pos = self.transition_single_agent(self.cat_pos,a) 
    def transition(self,a):
        """
        Performs state transition for bith agents
        """
        self.transition_cat(a["cat"])
        self.transition_mouse(a["mouse"])
        reward = {"cat":self.reward_cat(),"mouse":self.reward_mouse()}
        s_tab = {"cat":self.state_cat(),"mouse":self.state_mouse()}
        self.count+=1
        return s_tab,reward
    def state_mouse(self):
        return  self.mouse_pos[0]*self.Ny+self.mouse_pos[1]
    def transition_mouse(self,a:int):
        self.mouse_previous = self.mouse_pos
        self.mouse_pos = self.transition_single_agent(self.mouse_pos,a) 
    def terminated(self):
        "says if the episode is teminated"
        terminated = self.mouse_pos==self.target_mouse  or self.cat_pos==self.mouse_pos
        return terminated  
    def truncated(self):
        return self.count>self.max_steps
    def _agent_in_wall(self,agent_pos):
        return agent_pos[0]*self.Ny+agent_pos[1] in self.walls
    def transition_single_agent(self,agent_pos:tuple,a):
        assert(0<=a<=self.Na)
        d = self.actions[a]
        if self.Nx>agent_pos[0]+ d[0]>=0 and self.Ny>agent_pos[1]+d[1]>=0 and not self._agent_in_wall((agent_pos[0]+d[0],agent_pos[1]+d[1])):
            sp = (agent_pos[0]+ d[0], agent_pos[1]+d[1])
            assert(0<=sp[0]*self.Ny+sp[1]<self.Nx*self.Ny)
            s_out = sp
        else:
            #agent stays at the same position
            s_out = agent_pos
        return s_out
    def reward_cat(self):
        reward = (-1.0)*((self.mouse_pos[0] - self.cat_pos[0])**2 + (self.mouse_pos[1] - self.cat_pos[1])**2) + (-1.0)*(self.cat_pos==self.cat_previous)
        #reward =10.0*(self.mouse_pos==self.cat_pos) 
        return reward
    def reward_mouse(self):
        reward = (2.0)*(self.target_mouse==self.mouse_pos) + -10.0*(self.mouse_pos==self.cat_pos) + (-1.0)*(self.mouse_pos==self.mouse_previous)
        return reward
    def grid(self):
        s_mouse = self.mouse_pos
        s_cat = self.cat_pos
        T = np.zeros((self.Nx,self.Ny))
        T[s_mouse[0], s_mouse[1]] = 1
        T[s_cat[0], s_cat[1]] = -1
        T[self.target_mouse[0],self.target_mouse[1]] = 5
        for w in self.walls:
            T[w//self.Ny,w%self.Ny] = -5
        if self.terminated():
            print(f"termintated, episode {self.count}")
        print("cat", self.state_cat())
        print("mouse", self.state_mouse())
        print(T)
        return T
