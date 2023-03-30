import torch.optim as optim
import torch.nn as nn 
import torch 
import copy

from custom_dataset import load_data_loader
class Client: 
    def __init__(self, id, train_data={'x':[],'y':[]}, test_data={'x':[],'y':[]}, model=None):
        self.id = id 
        self.model = model #CustomLogisticRegression()
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-2)
        self.loss_fn = nn.CrossEntropyLoss()
        
        self.train_loader, self.test_loader = load_data_loader(train_data, test_data)
        self.num_samples = len(self.train_loader.dataset)
        self.test_samples = len(self.test_loader.dataset)
        
        self.xi_factor = 0.1 # TODO
        self.diff_grads = {} 

        self.initial_train() # for generating the initial grads (train without global grads)
        print(f"id = {id}, model = {model}, num_samples = {self.num_samples}")
    
    def initial_train(self): 
        r""" Train one epoch to update gradients for the first iteration"""
        num_epochs = 5
        for _ in range(num_epochs): 
            for _, (X, y) in enumerate(self.train_loader):
                output = self.model(X)
                loss = self.loss_fn(output, y)
                # Back propagation
                self.optimizer.zero_grad()
                loss.backward()  # update gradients
                self.optimizer.step()  # update model parameters
    
    def get_params(self): 
        r"""Get model.parameters()
        Return: 
            dict {'state_dict': tensor(param.data)}
        """
        return (self.num_samples, self.model.get_params())

    def set_params(self, params_dict=None):
        r"""Set initial params to model parameters()
        Args: 
            model_params: dict {'state_dict': model.parameters().data}
        Return: None 
            set model.parameters = model_params 
        Hint: https://medium.com/@mrityu.jha/understanding-the-grad-of-autograd-fc8d266fd6cf
        """
        self.model.set_params(params_dict)
    
    def get_grads(self): 
        r"""Get model.parameters() gradients 
        Return: 
            dict {'state_dict': tensor(param.grad)}
        """
        return (self.num_samples, self.model.get_grads())
    
    def dot_diff_grads(self, diff_gards): 
        r"""diff_grads: dict {'state_dict': grad}"""
        sum = 0.0
        local_params = self.get_params()[1]

        for k in local_params.keys():
            sum += (local_params[k] * diff_gards[k]).sum()
        
        return sum
    
    def train(self, num_epochs):
        # Calculate diff_grads 
        size = len(self.train_loader.dataset)

        for t in range(num_epochs): 
            # print(f"Epoch {t+1}\n-------------------------------")
            for batch, (X, y) in enumerate(self.train_loader):
                output = self.model(X)
                loss = self.loss_fn(output, y)

                # Calculate surrogate term and update the loss
                # Hint: https://discuss.pytorch.org/t/how-to-add-a-loss-term-from-hidden-layers/47804
                surr_term = self.dot_diff_grads(self.diff_grads)
                loss = loss + surr_term

                # Back propagation
                self.optimizer.zero_grad()
                loss.backward()  # update gradients
                self.optimizer.step()  # update model parameters

                # print log
                # loss, current = loss.item(), batch * len(X)
                # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                # print(f"model.parameters() =\n{self.model.get_params()}")
    
        # print("Done!")
        wsoln = self.get_params()
        return wsoln
    
    def calc_diff_grads(self, glob_grads): 
        local_grads = self.get_grads()[1] # the second arg

        for k in local_grads.keys(): 
            self.diff_grads[k] = self.xi_factor * glob_grads[k] - local_grads[k]
    

    def common_test(self, dataloader): 
        size = len(dataloader.dataset) # number of samples of train set or test set 
        num_batches = len(dataloader)
        test_loss, correct = 0, 0 
        
        with torch.no_grad(): 
            for X, y in dataloader: 
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        
        test_loss /= num_batches
        # correct /= size 
        
        # print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

        return  correct, test_loss
    
    def test(self): 
        r"""Evaluate on the test dataset"""
        tot_correct, _ = self.common_test(self.test_loader)
        return self.test_samples, tot_correct

    def train_error_and_loss(self): 
        r"""Evaluate on the train dataset"""
        tot_correct, loss = self.common_test(self.train_loader)
        return self.num_samples, tot_correct, loss 
    
    def fake_set_grads(self): 
        params = copy.deepcopy(self.get_params()[1])
        fake_global_grads = {k: torch.randn_like(v) for k, v in zip(params.keys(), params.values())}
        self.calc_diff_grads(glob_grads=fake_global_grads)

def test_train(): 
    print("test_train")
    from custom_model import CustomLogisticRegression
    from custom_dataset import test_load_data 

    model = CustomLogisticRegression(input_dim=5, output_dim=3)
    
    # FRIST USER
    user_id = 0
    train_data1, test_data1 = test_load_data(user_id)
    model1 = copy.deepcopy(model)
    client = Client(user_id, train_data1, test_data1, model1)
    
    client.fake_set_grads()
    num_epochs = 5 
    
    client.train(num_epochs)
    posttrain_params = client.get_params()
    
    print(f"posttrain_params =\n{posttrain_params}")

    print("##########################\n##########################")
    # SECOND USER 
    user_id = 1
    train_data2, test_data2 = test_load_data(user_id)
    model2 = copy.deepcopy(model)
    client2 = Client(user_id, train_data2, test_data2, model2)
    pretrain_params2 = copy.deepcopy(client2.get_params())
    print(f"pretrain_params2 =\n{pretrain_params2}")
    
    client2.fake_set_grads()
    client2.train(num_epochs)
    
    posttrain_params2 = client2.get_params()
    print(f"posttrain_params2 =\n{posttrain_params2}")
    print(f"client.get_params() =\n{client.get_params()}")

def test_test(): 
    print("test_test()")
    from custom_model import CustomLogisticRegression
    from custom_dataset import test_load_data 

    model = CustomLogisticRegression(input_dim=5, output_dim=3)

    user_id = 4
    train_data, test_data = test_load_data(user_id)
    client = Client(user_id, train_data, test_data, model)
    num_epochs = 30

    # Receive global grads 
    client.fake_set_grads()
    ## Train 
    client.train(num_epochs)

    ## Test
    print("Test Loss\n-------------------------------")
    client.test()

def test_diff_grads(): 
    print("test_diff_grads()")
    from custom_model import CustomLogisticRegression
    from custom_dataset import test_load_data 

    model = CustomLogisticRegression(input_dim=5, output_dim=3)

    user_id = 1
    train_data, test_data = test_load_data(user_id)
    client = Client(user_id, train_data, test_data, model)
    pretrain_params = copy.deepcopy(client.get_params()[1])
    print(f"pretrain_params =\n{pretrain_params}")

    diff_dict = {k: torch.randn_like(v) for k, v in zip(pretrain_params.keys(), pretrain_params.values())}
    global_grads = {k: torch.ones_like(v) for k, v in zip(pretrain_params.keys(), pretrain_params.values())}

    num_epochs = 30
    ## Train 
    client.train(num_epochs, diff_dict)

    ## Test diff grads
    diff_grads = client.calc_diff_grads(glob_grads=global_grads)
    print(diff_grads)

def test_initial_grads(): 
    from custom_model import CustomLogisticRegression
    from custom_dataset import test_load_data 

    model = CustomLogisticRegression(input_dim=5, output_dim=3)
    user_id = 1
    train_data, test_data = test_load_data(user_id)
    client = Client(user_id, train_data, test_data, model)
    print(client.get_grads())

if __name__ == '__main__':
    # test_train() 
    test_test()
    # test_diff_grads()
    # test_initial_grads()