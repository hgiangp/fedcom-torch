from src.custom_dataset import load_dataloader
# from custom_dataset import load_dataloader
import torch 

class Client: 
    def __init__(self, id, model, params, train_data={'x':[],'y':[]}, test_data={'x':[],'y':[]}):        
        self.id = id 

        # check available device
        device = self.check_device()
        
        # load model, data to availble device 
        self.model = model(params['model_params'], params['learning_rate'], params['xi_factor'], device) 
        self.train_loader, self.test_loader = load_dataloader(train_data, test_data, device)

        self.num_samples = len(self.train_loader.dataset)
        self.test_samples = len(self.test_loader.dataset)
        print(f"id = {id}, model = {model}, device = {device}, num_samples = {self.num_samples}")

        self.global_params = None # save w(t)
        self.global_grad = None # save F_k (w(t))

    def check_device(self): 
        # setting device on GPU if available, else CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device
        
    def init_train(self, num_epochs=1): 
        r""" Train one epoch to update gradients for the first iteration"""
        self.model.train(num_epochs=num_epochs, train_loader=self.train_loader)

    def get_wparams(self): 
        r"""Get weighted model.parameters()"""
        return (self.num_samples, self.model.get_params())
    
    def get_params(self): 
        r"""Get model.parameters()"""
        return self.model.get_params()

    def set_params(self, params: dict, train_data=None):
        r"""Set initial params to model parameters()
        Args: params: dict {'state_dict': model.parameters().data} """
        # save global params 
        self.global_params = params # TODO: check detach memory between users 
        print(f"id = {self.id}, global_params = {calculate_model_norm(self.global_params)}")
        
        # set w_k(t) = w(t)
        self.model.set_params(params)
        
        # update F_k(w(t)) 
        self.init_train(num_epochs=1)
        self.global_grad = self.get_grads()
        if train_data is not None: 
            self.model.calculate_jacobian(train_data)

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
        # 1. receive f, p, w(t) from server -> set_params + save global_params 
        # 2. + calculate local gradient grad F_k (w(t))
        # 3. train local model in num_epochs
        self.model.train_fed(num_epochs, dataloader, ggrads)
        # 4. estimate L, gamma 
        # 5. 

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
    
    def estimate_second_gradient(self): 
        w = self.get_params()
        grad = self.get_grads()

        dgrad = calculate_model_diff(grad, self.global_grad)
        dw = calculate_model_diff(w, self.global_params)
        rs = calculate_model_norm(dgrad)/calculate_model_norm(dw)
        print("estimate_second_gradient = ", rs)
        return (self.num_samples, rs)
    
    def estimate_convex_factor(self):
        grad = self.get_grads()
        _, loss = self.model.test(self.train_loader)
        loss_final = 0.0 

        gamma = calculate_model_norm(grad)/2/(loss - loss_final)
        print("estimate_convex_factor: ", gamma)
        return (self.num_samples, gamma)
    
    def get_global_grad(self): 
        return (self.num_samples, self.global_grad)

def print_model(parameters: dict): 
    for name, params in parameters.items():
        print(name)
        for param in params: 
            print(param)

def calculate_model_norm(parameters: dict):
    total_norm = 0.0 
    for name, p in parameters.items():
        # print(name)
        param_norm = p.norm(2)
        total_norm += param_norm.item() ** 2
        # print(total_norm)
    total_norm = total_norm ** (1. / 2)
    return total_norm

def calculate_model_diff(parameters1: dict, parameters2: dict):
    print("parameters1", calculate_model_norm(parameters1))
    print("parameters2", calculate_model_norm(parameters2))
    diff = {k: torch.zeros_like(v) for k, v in parameters1.items()}
    for name, p1 in parameters1.items():
        diff[name] = p1 - parameters2[name]
    return diff

def test():
    import torch
    import os, sys 

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(parent_dir)
    sys.path.append(parent_dir)
    
    from flearn.models.mnist.mclr import Model
    from src.custom_dataset import test_load_data
    import importlib

    train_data, test_data = test_load_data(user_id=1, dataset_name='mnist')

    model_path = '%s.%s.%s.%s' % ('flearn', 'models', 'mnist', 'mclr')
    mod = importlib.import_module(model_path)
    model = getattr(mod, 'Model')

    params = {}
    params['model_params'] = (784, 10)
    params['learning_rate'] = 0.01
    params['xi_factor'] = 1 

    user_id = 1

    client = Client(user_id, model, params, train_data, test_data)

    for _ in range(200) :
        gw = client.get_params()
        print("gw", calculate_model_norm(gw))
        client.set_params(gw, train_data)
        client.train(5, None)
    
if __name__ == '__main__':
    test()