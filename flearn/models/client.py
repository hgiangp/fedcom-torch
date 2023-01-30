from torch.utils.data import DataLoader
class Client: 
    def __init__(self, id, model = None, train_data={'x': [], 'y': []}, test_data={'x': [], 'y':[]}, batch_size=1):
        self.id = id 
        self.model = model
        self.train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
        self.train_samples = len(train_data)
        self.test_samples = len(test_data)
    
    def train(self, num_epochs, surr_term): 
        pass
        