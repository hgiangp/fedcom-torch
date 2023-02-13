import torch.nn as nn 
import torch 
class LinearRegression(nn.Module): 
    def __init__(self, input_dim=2, output_dim=3):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x): 
        # x = torch.flatten(x, 1)
        outputs = self.linear(x)
        return outputs

class CustomLogisticRegression(nn.Module): 
    r""" Custom Logistic Regression: 1 Linear Layer 
        Loss function: Cross Entropy
        https://aaronkub.com/2020/02/12/logistic-regression-with-pytorch.html
        https://discuss.pytorch.org/t/multi-class-cross-entropy-loss-and-softmax-in-pytorch/24920/9
    """
    def __init__(self, input_dim=2, output_dim=3): 
        super(CustomLogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x): 
        outputs = self.linear(x)
        return outputs
    
    def get_params(self): 
        return {k: v.data for k, v in zip(self.state_dict(), self.parameters())}

    def set_params(self, params_dict): 
        self.load_state_dict(params_dict)
    
    def get_grads(self): 
        return {k: v.grad for k, v in zip(self.state_dict(), self.parameters())} 

    def get_model_size(self):  
        torch_float = 32 
        msize = sum(param.numel() for param in self.parameters()) * torch_float
        return msize

def test(): 
    model = CustomLogisticRegression()
    test_param = model.get_params()

    test_load_dict = {k: torch.randn_like(v) for k, v in zip(test_param.keys(), test_param.values())}
    print(f"test_load_dict =\n{test_load_dict} ")
    model.set_params(test_load_dict)
    print(f"model.get_params()=\n{model.get_params()}")

    print(f"model.get_grads() = {model.get_grads()}")

if __name__=="__main__": 
    test()
