# off-policy Online Q actor critic with a buffer
1. take action $a \sim \pi_{\theta}(a|s)$, get $(s,a',s',r)$ and store in $R$

2. sample a batch $(s_{i}, a_{i},r_{i},s_{p})$ from buffer.

3. update ${\hat{Q}}({\Phi})$ using targets $y_{i} = r_{i} + \gamma\hat{Q}({\Phi})$ for each $s_{i}, a_{i}$

4. ${\nabla}_{\theta}J(\theta) = \frac{1}{N}{\Sigma}_{i}{\nabla}_{\theta}\pi_{\theta}({{a}_{i}}^{\pi}|s_{i})$
