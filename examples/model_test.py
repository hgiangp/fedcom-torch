import torch
import torch.nn as nn 
import torch.optim as optim 

class LinearRegression(nn.Module): 
    def __init__(self, input_dim=2, output_dim=3):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x): 
        # x = torch.flatten(x, 1)
        outputs = self.linear(x)
        return outputs

class CustomLogisticRegression(nn.Module): 
    def __init__(self, input_dim=60, output_dim=10): 
        super(CustomLogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x): 
        outputs = self.linear(x)
        return outputs


def test_linear_regression(): 
    model = LinearRegression()
    optimizer = optim.SGD(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()

    x = torch.randn(4, 2)
    y = torch.randn(4, 3)

    print(model)

    diff_dict = {k: torch.randn_like(v.data) for k, v in zip(model.state_dict(), model.parameters())}

    print(diff_dict)

def get_params(model): 
    return {k: v.data for k, v in zip(model.state_dict(), model.parameters())}

def add_params_grads(params_dict, grads_dict): 
    r""" Calculate surrogate term by dot product of 
    the previous diff_gradients and current local parameters
    Args: 
        params_dict: {'state_dict': param.data} current model parameters value
        grads_dict: {'state_dict': diff_grads} differentiate gradients between 
                    the previous global model (weighted by \eta) and local model parameters 
                    = dot ((\eta * \delta F (w^{t-1}) - F_n (w^{t-1}), w^t)
    Return: 
        Tensor()
    """
    sum = 0.0
    for k in params_dict.keys(): 
        sum += (params_dict[k] * grads_dict[k]).sum()

    return sum

def set_params(model_params=None):
    r"""Set initial params to model parameters()
    Args: 
        model_params: dict {'state_dict': model.parameters().data}
    Return: None 
        set model.parameters = model_params 
    Hint: https://medium.com/@mrityu.jha/understanding-the-grad-of-autograd-fc8d266fd6cf
    """
    with torch.no_grad(): 
        for name, param in zip(model.state_dict(), model.parameters()): 
            print(model_params[name])
            param.data = model_params[name] 
    print(get_params())

def get_params(model): 
    r"""Get model.parameters()
    Return: 
        dict {'state_dict': tensor(param.data)}
    """
    return {k: v.data for k, v in zip(model.state_dict(), model.parameters())}

def get_gradients(model): 
    r"""Get model.parameters() gradients 
    Return: 
        dict {'state_dict': tensor(param.grad)}
    """
    return {k: v.grad for k, v in zip(model.state_dict(), model.parameters())}

def train(model, loss_fn, x, y, diff_dict, optimizer): 
    r""" Solve local surrogate function"""
    # model.train() # switch between train_mode and eval_mode: batch_norm, dropout, ... different
    for epoch in range(20):
        # Compute prediction and loss
        output = model(x)
        loss = loss_fn(output, y)
        
        # Calculate surrogate term and update the loss 
        # Hint: https://discuss.pytorch.org/t/how-to-add-a-loss-term-from-hidden-layers/47804
        surr_term = add_params_grads(get_params(), diff_dict)
        loss = loss + surr_term

        # Back propagation 
        optimizer.zero_grad()
        loss.backward() # update gradients 
        optimizer.step()  # update model parameters 
        print('Epoch {}: loss {}'.format(epoch, loss.item()))
    
def test(model): 
    r""" Test test_data, evaluate model 
    Hint: https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch
    """
    model.eval()
    with torch.no_grad(): 
        pass 
    pass 

def test_set_params(): 
    print('test_set_params(): start')
    # first train 
    train()
    print(get_params())

    # save optimal param to dict 
    pretrained_params_dict = get_params()

    # init test_params copy dictionary 
    test_params_dict = {k: torch.zeros_like(v.data) for k, v in zip(model.state_dict(), model.parameters())}
    print(f'test_params_dict dtype = {type(test_params_dict)}')
    print(test_params_dict)

    # set all model.parameters() to zeros 
    print('='*20)
    set_params(test_params_dict)
    print('='*20)
    print(get_params()) 

    # load the pretrained params to model.parameters 
    set_params(pretrained_params_dict)

    # continue training 
    train()

    print(get_params()) 
    
    print('test_set_params(): done')

if __name__=='__main__': 
    test_set_params()
    params = get_params()
    grads = get_gradients()
    print(grads)