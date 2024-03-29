import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import json

np.set_printoptions(precision=3, linewidth=np.inf)

seed=0
rng = np.random.default_rng(seed=seed)

num_users = 10
num_labels = 3

# Setup directory for train/test data 
train_file = 'data_niid_seed_0_train_8.json'
test_file = 'data_niid_seed_0_test_8.json'
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, 'data')
train_path = os.path.join(data_dir, 'train', train_file)
test_path = os.path.join(data_dir, 'test', test_file)
dir_path = os.path.dirname(train_path)
if not os.path.exists(dir_path): 
    os.makedirs(dir_path)
dir_path = os.path.dirname(test_path)
if not os.path.exists(dir_path): 
    os.makedirs(dir_path)

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

# Tranform data from Image to numpy array 
images = []
targets = []
for image, target in trainset: 
    images.append(np.asarray(image).reshape(-1))
    targets.append(target)

images = np.asarray(images)
targets = np.asarray(targets)
print(images.shape)
print(targets[:10])

# Normalize the image data
# mu = np.mean(images, axis=0) 
# sigma = np.std(images, axis=0)
# images = (images - mu)/(sigma + 0.001)
# print(mnist_data.shape, max(mnist_data[0]), min(mnist_data[0]))

# Categorize the image data
cifar_data = []
for i in range(10):
    idx = targets==i
    cifar_data.append(images[idx])

len_data = [len(v) for v in cifar_data]
print(f"Number of samples of each label: {len_data}\t{np.asarray(len_data).sum()}")

# CREATE USER DATA SPLIT 
# Assign 90 samples to each users, 3 labels, 20 samples each
sams_per_lab = 50
X = [[] for _ in range(num_users)]
y = [[] for _ in range(num_users)]
idx = np.zeros(10, dtype=int) # 10 labels 0 - 9

for user in range(num_users): 
    for j in range(num_labels): 
        l = (user + j)%10 
        X[user] += cifar_data[l][idx[l]:idx[l]+sams_per_lab].tolist()
        y[user] += (l*np.ones(sams_per_lab)).tolist()
        idx[l] += sams_per_lab
 
print(f"idx = {idx}") #  90 * 10 / 10 = 60 

# Assign remaining samples by power law 
allocated_samples = np.ceil(sams_per_lab*num_labels*10/num_users)

props = rng.lognormal(0, 2.0, (10, 35, 3)) # 10 classes, 10 users, 3 labels
props = np.array([[[len(v)-allocated_samples]] for v in cifar_data]) * props/np.sum(props, axis=(1, 2), keepdims=True)
# print(f"props = {props}")

for user in range(num_users): 
    for j in range(num_labels): 
        l = (user + j)%10
        n_samples = int(props[l, user, j])
        # print(n_samples)
        if idx[l] + n_samples < len(cifar_data[l]):   
            X[user] += cifar_data[l][idx[l]:idx[l]+n_samples].tolist()
            y[user] += (l*np.ones(n_samples)).tolist()
            idx[l] += n_samples

print(f"idx = {idx}") 

# Create data structure 
train_data = {'users': [], 'user_data': {}, 'num_samples': []}
test_data = {'users': [], 'user_data': {}, 'num_samples': []}

# Setup 10 users 
for i in range(num_users): 
    uname = 'f_{0:05d}'.format(i)

    combined = list(zip(X[i], y[i]))
    rng.shuffle(combined)
    X[i][:], y[i][:] = zip(*combined)
    num_samples = len(X[i])
    train_len = int(0.8*num_samples)
    test_len = num_samples - train_len

    train_data['users'].append(uname)
    train_data['user_data'][uname] = {'x': X[i][:train_len], 'y': y[i][:train_len]}
    train_data['num_samples'].append(train_len)
    test_data['users'].append(uname)
    test_data['user_data'][uname] = {'x': X[i][train_len:], 'y': y[i][train_len:]}
    test_data['num_samples'].append(test_len)

print(f"train_data['num_samples'] = {train_data['num_samples']}, sum = {sum(train_data['num_samples'])}")
print(f"test_data['num_samples'] = {test_data['num_samples']}, sum = {sum(test_data['num_samples'])}")

with open(train_path, 'w') as outfile: 
    json.dump(train_data, outfile)
with open(test_path, 'w') as outfile: 
    json.dump(test_data, outfile)