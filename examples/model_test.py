import torch.nn as nn 

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
    def __init__(self, input_dim=60, output_dim=10): 
        super(CustomLogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x): 
        outputs = self.linear(x)
        return outputs
