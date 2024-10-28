# Actor critic implementation with State value fonction
# What I will try next is a parralel implementation --> synchonized parralel actor-critic
In a loop:
1. Initialize s at random <br>

Tant que s!=sp <br>
    2. Transition: choose action using policy $\pi$, change state $s$, collect reward $r$ <br>
    3. V(s) <- r(s) + $\gamma$ V(sp) <br>
    4. Compute $\nabla J$ <br>
    5. s <-- sp <br>


