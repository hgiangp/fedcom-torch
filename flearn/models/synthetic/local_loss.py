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

model = LinearRegression()
optimizer = optim.SGD(model.parameters(), lr=1e-2)
loss_fn = nn.MSELoss()

x = torch.randn(4, 2)
y = torch.randn(4, 3)

print(model)

diff_dict = {k: torch.randn_like(v.data) for k, v in zip(model.state_dict(), model.parameters())}

print(diff_dict)

def serialize_params(model): 
    return {k: v.data for k, v in zip(model.state_dict(), model.parameters())}

def serialize_gards(model): 
    return {k: v.grad for k, v in zip(model.state_dict(), model.parameters())}

def add_params_grads(params_dict, grads_dict): 
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
    """
    with torch.no_grad(): 
        for name, param in zip(model.state_dict(), model.parameters()): 
            print(model_params[name])
            param.data = model_params[name] 
    print(serialize_params(model))

def train(): 
    for epoch in range(20): 
        optimizer.zero_grad()
        output = model(x)

        loss = loss_fn(output, y)
        surr_term = add_params_grads(serialize_params(model), diff_dict)
        loss = loss + surr_term
        
        loss.backward()
        optimizer.step()
        print('Epoch {}: loss {}'.format(epoch, loss.item()))
    # https://discuss.pytorch.org/t/how-to-add-a-loss-term-from-hidden-layers/47804

def test_set_params(): 
    print('test_set_params(): start')
    # first train 
    train()
    print(serialize_params(model))

    # save optimal param to dict 
    pretrained_params_dict = serialize_params(model)

    # init test_params copy dictionary 
    test_params_dict = {k: torch.zeros_like(v.data) for k, v in zip(model.state_dict(), model.parameters())}
    print(f'test_params_dict dtype = {type(test_params_dict)}')
    print(test_params_dict)

    # set all model.parameters() to zeros 
    print('='*20)
    set_params(test_params_dict)
    print('='*20)
    print(serialize_params(model)) 

    # load the pretrained params to model.parameters 
    set_params(pretrained_params_dict)

    # continue training 
    train()

    print(serialize_params(model)) 
    
    print('test_set_params(): done')

if __name__=='__main__': 
    test_set_params()