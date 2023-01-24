# Created date: 2022-01-24 
import numpy as np 
from tqdm import trange
import json

seed = 42
rng = np.random.default_rng(seed=seed)

NUM_USERS = 10 

def softmax(x): 
    ex = np.exp(x)
    sum_ex = np.sum(np.exp(x))
    return ex/sum_ex

def generate_synthetic(alpha, beta, is_iid): 

    dimension = 60 
    NUM_CLASSES = 10 

    samples_per_user = rng.lognormal(4, 2, (NUM_USERS)).astype(int) + 50 
    print(samples_per_user)

    X_split = [[] for _ in range(NUM_USERS)]
    y_split = [[] for _ in range(NUM_USERS)]

    ### define variables ### 
    mean_W = rng.normal(loc=0, scale=alpha, size=NUM_USERS)
    mean_b = mean_W
    B = rng.normal(loc=0, scale=beta, size=NUM_USERS)
    mean_x = np.zeros(shape=(NUM_USERS, dimension))

    diagonal = np.zeros(shape=dimension)
    for j in range(dimension): 
        diagonal[j] = np.power((j+1), -1.2)
    cov_x = np.diag(diagonal)

    for i in range(NUM_USERS): 
        if is_iid == 1: # generated from the unit distribution mean = 0, deviation = 1 
            mean_x[i] = np.ones(shape=dimension) * B[i] # all zeros 
        else: # generated from different normal distributions 
            mean_x[i] = rng.normal(loc=B[i], scale=1, size=dimension)
        print(mean_x[i])
    
    if is_iid == 1: 
        W_global = rng.normal(loc=0, scale=1, size=(dimension, NUM_CLASSES))
        b_global = rng.normal(loc=0, scale=1, size=NUM_CLASSES)
    
    for i in range(NUM_USERS): 
        W = rng.normal(loc=mean_W[i], scale=1, size=(dimension, NUM_CLASSES))
        b = rng.normal(loc=mean_b[i], scale=1, size=NUM_CLASSES)

        if is_iid: 
            W = W_global
            b = b_global

        xx = rng.multivariate_normal(mean=mean_x[i], cov=cov_x, size=samples_per_user[i])
        yy = np.zeros(shape=samples_per_user[i])

        for j in range(samples_per_user[i]): 
            tmp = np.dot(xx[j], W) + b 
            yy[j] = np.argmax(softmax(tmp))
        
        X_split[i] = xx.tolist()
        y_split[i] = yy.tolist()

        print("{}-th users has {} samples".format(i, len(y_split[i])))
    
    return X_split, y_split

def main(): 

    train_path = "data/train/mytrain.json"
    test_path = "data/train/mytest.json"

    X, y = generate_synthetic(alpha=0, beta=0, is_iid=0)

    # Create data structure 
    train_data = {'user': [], 'user_data': {}, 'num_samples': []}    
    test_data = {'user': [], 'user_data': {}, 'num_samples': []}

    for i in trange(NUM_USERS, ncols=120):
        uname = 'f_{0:05d}'.format(i)
        combined = list(zip(X[i], y[i]))
        rng.shuffle(combined)
        X[i][:], y[i][:] = zip(*combined)
        num_samples = len(X[i])
        train_len = int(0.9 * num_samples)
        test_len = num_samples - train_len

        train_data['user'].append(uname)
        train_data['user_data'][uname] = {'x': X[i][:train_len], 'y': y[i][:train_len]}
        train_data['num_samples'].append(train_len)
        
        test_data['user'].append(uname)
        test_data['user_data'][uname] = {'x': X[i][train_len:], 'y': y[i][train_len:]}
        test_data['num_samples'].append(test_len)

    with open(train_path, 'w+') as ouf: # '+' option: create if not exist
        json.dump(train_data, ouf)
    with open(test_path, 'w+') as ouf: 
        json.dump(test_data, ouf)

if __name__=='__main__': 
    main()
    
