import torch.optim as optim
import torch.nn as nn 
import torch 

from model_test import CustomLogisticRegression

from custom_dataset import load_data 
from torch_utils import * 

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
        return get_params(self.model)

    def set_params(self, model_params=None):
        r"""Set initial params to model parameters()
        Args: 
            model_params: dict {'state_dict': model.parameters().data}
        Return: None 
            set model.parameters = model_params 
        Hint: https://medium.com/@mrityu.jha/understanding-the-grad-of-autograd-fc8d266fd6cf
        """
        set_params(self.model, model_params)
    
    def get_gradients(self): 
        r"""Get model.parameters() gradients 
        Return: 
            dict {'state_dict': tensor(param.grad)}
        """
        return get_gradients(self.model)
    
def test_train(): 
    print("test_train ")
    csmodel = CustomModel()
    test_param = csmodel.get_params()
    # print(test_param)
    diff_dict = {k: torch.randn_like(v) for k, v in zip(test_param.keys(), test_param.values())}
    
    epoches = 100 
    for t in range(epoches): 
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(csmodel.train_loader, csmodel.model, csmodel.loss_fn, csmodel.optimizer, diff_dict)
        test_loop(csmodel.test_loader, csmodel.model, csmodel.loss_fn)
    print("Done!")

if __name__ == '__main__':
    test_train() 
