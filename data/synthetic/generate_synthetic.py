### Created date: 2023-01-24
### Generated data for the synthetic data set 
### Details: 
#

import numpy as np 
from tqdm import trange
import json

seed = 1
rng = np.random.default_rng(seed)

num_users = 10 # TODO
dimension = 5 # TODO: 60 
num_classes = 3 # TODO: 10 

def softmax(x): 
    ex = np.exp(x)
    sum_ex = np.sum(np.exp(x))
    return ex/sum_ex

def generate_synthetic(alpha, beta, iid): 
    samples_per_user = rng.lognormal(mean=4, sigma=2, size=num_users).astype(int) + 50 # shape = (NUM_USERS,)
    print(samples_per_user)

    ### define variables ### 
    mean_W = rng.normal(loc=0, scale=alpha, size=num_users) # (NUM_USERS, )
    mean_b = mean_W # (NUM_USERS, )
    B = rng.normal(loc=0, scale=beta, size=(num_users, 1)) # (NUM_USERS, 1)

    diagonal = np.array([np.power((j+1), -1.2) for j in range(dimension)]) # (dimension, )
    cov_x = np.diag(diagonal) # (dimension, dimension)

    if iid: # all users' features follow the same distribution 
        mean_x = np.tile(B, reps=dimension)
    else: 
        mean_x = np.array([rng.normal(loc=B[i], scale=1, size=dimension) for i in range(num_users)]) 
    print("mean_x.shape = {}".format(mean_x.shape)) # (users, dimension)
        
    if iid: # params are generated from the unit distribution mean = 0, deviation = 1
        W_global = rng.normal(loc=0, scale=1, size=(dimension, num_classes))
        b_global = rng.normal(loc=0, scale=1, size=num_classes)
    
    X_split = [[] for _ in range(num_users)]
    y_split = [[] for _ in range(num_users)]
    for i in range(num_users): 
        if iid: # parameters are identical for all users 
            W = W_global
            b = b_global
        else: 
            W = rng.normal(loc=mean_W[i], scale=1, size=(dimension, num_classes)) # (dimension, classes)
            b = rng.normal(loc=mean_b[i], scale=1, size=num_classes) # (classes, )
        
        xx = rng.multivariate_normal(mean=mean_x[i], cov=cov_x, size=samples_per_user[i]) # (num_samples_i, dimension) 
        tmp = softmax(np.dot(xx, W) + b) # (samples, classes)
        yy = np.argmax(tmp, axis=1) # (samples, ) 
        
        X_split[i] = xx.tolist()
        y_split[i] = yy.tolist()

        print("{}-th users has {} samples".format(i, len(y_split[i])))
    
    return X_split, y_split

def main(): 

    train_path = "data/train/mytrain.json"
    test_path = "data/test/mytest.json"

    X, y = generate_synthetic(alpha=0, beta=0, iid=0)

    # Create data structure 
    train_data = {'user': [], 'user_data': {}, 'num_samples': []}    
    test_data = {'user': [], 'user_data': {}, 'num_samples': []}

    for i in trange(num_users, ncols=120):
        uname = 'f_{0:05d}'.format(i)
        combined = list(zip(X[i], y[i]))
        rng.shuffle(combined)
        X[i][:], y[i][:] = zip(*combined)
        num_samples = len(X[i])
        train_len = int(0.8 * num_samples)
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
    
