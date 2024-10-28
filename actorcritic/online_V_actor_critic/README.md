# Actor critic implementation with State value fonction
# What I will try next is a parralel implementation --> synchonized parralel actor-critic
In a loop:
1. Initialize s at random <br>

While s!=sp <br>
    2. Transition: choose action using policy $\pi_{\theta}$, change state $s$, collect reward $r(s)$ <br>
    3. V(s) $\longleftarrow + \gamma$ V(sp) <br>
    4. Compute $\nabla_{\theta}J$ <br>
    5. Update parameters $\theta$ 
    5. s $\longleftarrow$  sp <br>


