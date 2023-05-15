from src.custom_dataset import load_dataloader

class Client: 
    def __init__(self, id, train_data={'x':[],'y':[]}, test_data={'x':[],'y':[]}, model=None):
        self.id = id 
        self.model = model # Model
        self.train_loader, self.test_loader = load_dataloader(train_data, test_data)
        self.num_samples = len(self.train_loader.dataset)
        self.test_samples = len(self.test_loader.dataset)

        self.init_train() # for generating the initial grads
        print(f"id = {id}, model = {model}, num_samples = {self.num_samples}")
        
    def init_train(self, num_epochs=1): 
        r""" Train one epoch to update gradients for the first iteration"""
        self.model.train(num_epochs=num_epochs, train_loader=self.train_loader)

    def get_wparams(self): 
        r"""Get weighted model.parameters()"""
        return (self.num_samples, self.model.get_params())
    
    def get_params(self): 
        r"""Get model.parameters()"""
        return self.model.get_params()

    def set_params(self, params: dict):
        r"""Set initial params to model parameters()
        Args: params: dict {'state_dict': model.parameters().data} """
        self.model.set_params(params)
        # print(f"set_params arx_params = {self.arx_params}")
    
    def get_wgrads(self, ground: int): 
        r"""Get weighted model.parameters() gradients"""
        if ground == 0:
            self.init_train(num_epochs=1) 
        return (self.num_samples, self.model.get_grads())
    
    def get_grads(self): 
        r"""Get model.parameters() gradients"""
        return self.model.get_grads()
    
    def train(self, num_epochs: int, ggrads: dict):
        dataloader = self.train_loader
        self.model.train_fed(num_epochs, dataloader, ggrads)
        wsoln = self.get_wparams()
        return wsoln
    
    def test(self): 
        r"""Evaluate on the test dataset"""
        tot_correct, loss = self.model.test(self.test_loader)
        return self.test_samples, tot_correct, loss

    def train_error_and_loss(self): 
        r"""Evaluate on the train dataset"""
        tot_correct, loss = self.model.test(self.train_loader)
        return self.num_samples, tot_correct, loss 

def test():
    import torch
    from flearn.models.synthetic.mclr import Model
    from src.custom_dataset import test_load_data
    model = Model(5, 3)

    user_id = 1
    train_data, test_data = test_load_data(user_id)
    client = Client(user_id, train_data, test_data, model)
    
    lparams = client.model.get_params()
    ggrads = {k: torch.zeros_like(v) for k, v in lparams.items()}
    client.train(num_epochs = 30, ggrads=ggrads)

if __name__ == '__main__':
    test()