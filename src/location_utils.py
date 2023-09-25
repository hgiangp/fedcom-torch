import numpy as np 
seed = 1
rng = np.random.default_rng(seed=seed)

def read_data(file_name='location_data.txt'):
    r"""Read location of vehicles 
    Args: 
    Return: 
        xs: np [], # (num_users, )
        ys: np [], # (num_users, )
        dirs_1: 1st direction before intersection, np [], # (num_users, )
        dirs_2: 2nd direction after intersection, np [], # (num_users, )
    """
    xs, ys = [], []
    dirs_1, dirs_2 = [], []
    for line in open(file_name, 'r'):
        data = line.strip().split(',')
        if len(data) < 6:
            print('data corruptted!')

        data = [float(xi) for xi in data]
        
        xs.append(data[0])
        ys.append(data[1])
        dirs_1.append(data[2])
        dirs_2.append(data[3])
    
    # shuffle the location 
    combined = list(zip(xs, ys, dirs_1, dirs_2))
    # rng.shuffle(combined)
    xs, ys, dirs_1, dirs_2 = zip(*combined)

    # convert to numpy array 
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    dirs_1 = np.asarray(dirs_1) * np.pi / 180 # convert to radian 
    dirs_2 = np.asarray(dirs_2) * np.pi / 180 # convert to radian 
    
    return xs, ys, dirs_1, dirs_2

def test():
    xs, ys, dirs_1, dirs_2 = read_data()
    print(f"{xs}\n{ys}\n{dirs_1}\n{dirs_2}")
    print(f"type(xs) = {type(xs)}, type(ys) = {type(ys)}, type(dirs1) = {type(dirs_1)},type(dirs2) = {type(dirs_2)}")

if __name__=='__main__':
    test()
