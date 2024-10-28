# Actor critic implementation with State value fonction
# What I will try next is a parralel implementation --> synchonized parralel actor-critic
In a loop:
1. Initialize s at random

Tant que s!=sp
    2. Transition: choose action using policy $\pi$, change state $s$, collect reward $r$
    3. V(s) <- r(s) + $\gamma$ V(sp)
    4. Compute $\nabla J$
    5. s <-- sp


