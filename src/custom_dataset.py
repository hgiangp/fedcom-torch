from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch 
import os 
import json

class CustomDataset(Dataset): 
    def __init__(self, input_dict, device):
        # self.input_dict = input_dict # dict {'x': [], 'y': []}
        self.data = torch.tensor(input_dict['x'], device=device)
        self.label = torch.tensor(input_dict['y'], dtype=int, device=device)
        self.device = device 
    
    def __len__(self):
        r""" Return number of samples in our dataset""" 
        return len(self.label)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        target = self.label[idx]
        return data, target 

def read_data(train_data_dir, test_data_dir):
    r""" Read json data to (clients: list, train_data: dict, test_data: dist)
    Args:
        train_data_dir, test_data_dir: train_data directory, test_data directory 
    Return: 
        clients: list []
        train_data:  {'client_name': {'x': [], 'y': []}}
        test_data:  {'client_name': {'x': [], 'y': []}}
    """
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json') ]
    for fname in train_files: 
        file_path = os.path.join(train_data_dir, fname)
        with open(file_path, 'r') as inf: 
            cdata = json.load(inf)
        train_data.update(cdata['user_data'])
    
    clients = list(sorted(train_data.keys()))
    print(clients)
    print(len(train_data))
    print(train_data[clients[0]].keys())

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]
    for fname in test_files: 
        file_path = os.path.join(test_data_dir, fname)
        with open(file_path, 'r') as inf: 
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])
    
    return clients, train_data, test_data

def load_dataloader(train_dict, test_dict, device, shuffle=True, drop_last=True): 
    batch_size = 32

    # Init CustomDataset 
    training_data = CustomDataset(train_dict, device)
    test_data = CustomDataset(test_dict, device)

    # Init DataLoader 
    traindata_loader = DataLoader(training_data, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    testdata_loader = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle, drop_last=False)
    return traindata_loader, testdata_loader

def test_load_data(user_id=1, dataset_name='synthetic'):
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_dir = os.path.join(parent_dir, 'data', dataset_name, 'data', 'train')
    test_dir = os.path.join(parent_dir, 'data', dataset_name, 'data', 'test')

    clients, train_dt, test_dt = read_data(train_dir, test_dir) # (clients, train_data, test_data)
    client_id = clients[user_id]
    train_data = train_dt[client_id] # dict{'x': [], 'y': []}
    test_data = test_dt[client_id] # dict{'x': [], 'y': []}
    print(f"len(train_data['y'] = {len(train_data['y'])}")
    print(f"len(test_data['y']) = {len(test_data['y'])}")
    return train_data, test_data

if __name__=='__main__': 
    test_load_data(user_id=0, dataset_name='mnist')