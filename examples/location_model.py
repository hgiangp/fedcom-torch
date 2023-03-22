import numpy as np 

seed = 42
rng = np.random.default_rng(seed=seed)
no_users = 10 
d = 20 
delta = 200

def init_location():
    xs = rng.normal(loc=0, scale=d, size=(no_users))
    ys = rng.normal(loc=0, scale=d, size=(no_users))
    dirs = rng.integers(low=0, high=4, size=(no_users)) 

    for i in range(no_users):
        if xs[i] > d: 
            dirs[i] = 0 
        elif (- xs[i] > d): 
            dirs[i] = 2 
        elif ys[i] > d: 
            dirs[i] = 1
        elif (-ys[i] > d): 
            dirs[i] = 3

    return xs, ys, dirs

def update_location(xs, ys, dirs): 
    delta_xs = np.zeros(no_users, dtype=int)
    delta_ys = np.zeros(no_users, dtype=int)

    for i in range(no_users): 
        if dirs[i] == 0:
            delta_xs[i] = -1
        elif dirs[i] == 1: 
            delta_ys[i] = -1
        elif dirs[i] == 2: 
            delta_xs[i]= 1
        else: # 3 
            delta_ys[i] = 1

    # Update location 
    xs_new = xs + delta_xs * delta
    ys_new = ys + delta_ys * delta

    # Change direction
    xsigns = xs_new * xs < 0
    ysigns = ys_new * ys < 0

    dir_changed = np.argwhere([x or y for x, y in zip(xsigns, ysigns)]) 
    dirs_new = rng.integers(low=0, high=4, size=(no_users))
    dirs[dir_changed] = dirs_new[dir_changed]

    return xs_new, ys_new, dirs 

def test(): 
    xs, ys, dirs = init_location()
    print("xs =", xs)
    print("ys =", ys)
    print("dirs =", dirs)
    xs, ys, dirs = update_location(xs, ys, dirs)
    print("xs =", xs)
    print("ys =", ys)
    print("dirs =", dirs) 

if __name__=='__main__': 
    test()