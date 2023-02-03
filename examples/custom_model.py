import torch.optim as optim
import torch.nn as nn 
import torch 

from model_test import CustomLogisticRegression

from custom_dataset import load_data 
from torch_utils import add_params_grads 


class CustomModel(object): 
    def __init__(self):
        self.model = CustomLogisticRegression()
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-2)
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_loader, self.test_loader = load_data()
    
    def get_params(self): 
        r"""Get model.parameters()
        Return: 
            dict {'state_dict': tensor(param.data)}
        """
        return {k: v.data for k, v in zip(self.model.state_dict(), self.model.parameters())}

    def set_params(self, model_params=None):
        r"""Set initial params to model parameters()
        Args: 
            model_params: dict {'state_dict': model.parameters().data}
        Return: None 
            set model.parameters = model_params 
        Hint: https://medium.com/@mrityu.jha/understanding-the-grad-of-autograd-fc8d266fd6cf
        """
        model = self.model  
        with torch.no_grad(): 
            for name, param in zip(model.state_dict(), model.parameters()): 
                print(model_params[name])
                param.data = model_params[name] 
        print(self.get_params())
    
    def get_gradients(self): 
        r"""Get model.parameters() gradients 
        Return: 
            dict {'state_dict': tensor(param.grad)}
        """
        return {k: v.grad for k, v in zip(self.model.state_dict(), self.model.parameters())}
    
    def train_loop(self, diff_dict): 
        dataloader = self.train_loader
        model = self.model 
        loss_fn = self.loss_fn

        size = len(dataloader.dataset)

        for batch, (X, y) in enumerate(dataloader):   
            # print(f'X = {X} \ny = {y}')
            output = model(X)
            loss = loss_fn(output, y)
                
            # Calculate surrogate term and update the loss 
            # Hint: https://discuss.pytorch.org/t/how-to-add-a-loss-term-from-hidden-layers/47804
            surr_term = add_params_grads(self.get_params(), diff_dict)
            loss = loss + surr_term

            # Back propagation 
            self.optimizer.zero_grad()
            loss.backward() # update gradients 
            self.optimizer.step()  # update model parameters 
            
            # print log  
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    def train(self, diff_dict): 
        epoches = 20 
        for t in range(epoches): 
            print(f"Epoch {t+1}\n-------------------------------")
            self.train_loop(diff_dict)
        print("Done!")


def test_train(): 
    print("test_train ")
    csmodel = CustomModel()
    test_param = csmodel.get_params()
    # print(test_param)
    diff_dict = {k: torch.randn_like(v) for k, v in zip(test_param.keys(), test_param.values())}
    csmodel.train(diff_dict)

if __name__ == '__main__':
    test_train() 
