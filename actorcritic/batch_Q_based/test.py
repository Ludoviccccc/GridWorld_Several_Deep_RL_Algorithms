
def test(p):
    #p is the policy
    i = 0
    s = S
    softm = nn.Softmax(dim=1)
    while True:
        clear_output(wait=True)
        print(i)
        state = tensor_state(s, Nx, Ny)
        out  = p(state)
        dist  = distributions.Categorical(softm(out))
        a = dist.sample([1]) 
        sp,R = transition(a,s, Nx, Ny, G)
        grid(sp, Nx, Ny)
        s = sp
        time.sleep(0.05)
        i+=1
        if s==G:
            break
    return i
