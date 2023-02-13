import numpy as np 

rng = np.random.default_rng()

no_users = 10 
d = 10 


def init():
    xs = rng.normal(loc=0, scale=d, size=(no_users))
    ys = rng.normal(loc=0, scale=d, size=(no_users))
    dirs = np.zeros(shape=(no_users))

    for i in range(no_users):
        if xs[i] > d: 
            dirs[i] = 0 
        elif (- xs[i] > d): 
            dirs[i] = 2 
        elif ys[i] > d: 
            dirs[i] = 1
        elif (-ys[i] < d): 
            dirs[i] = 4 
        else: 
            dirs[i] = rng.integers(low=0, high=4) 

    return (xs, ys, dirs)

def update(): 
    return 

def test(): 
    xs, ys, dirs = init()
    print("xs =", xs)
    print("ys =", ys)
    print("dirs =", dirs)


if __name__=='__main__': 
    test()