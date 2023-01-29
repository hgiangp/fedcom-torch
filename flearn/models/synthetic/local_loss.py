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
criterion = nn.MSELoss()

x = torch.randn(4, 2)
y = torch.randn(4, 3)

print(model)

diff_dict = {k: torch.randn_like(v.data) for k, v in zip(model.state_dict(), model.parameters())}

print(diff_dict)

def serialize_params(model): 
    return {k: v.data for k, v in zip(model.state_dict(), model.parameters())}

def serialize_gards(model): 
    return {k: v.grad for k, v in zip(model.state_dict(), model.parameters())}

def grads_params_add(grads_dict, params_dict): 
    sum = 0.0
    for k in params_dict.keys(): 
        sum += (params_dict[k] * grads_dict[k]).sum()

    return sum 

# grads_params_add(diff_dict, serialize_params(model))
# https://discuss.pytorch.org/t/how-to-add-a-loss-term-from-hidden-layers/47804
for epoch in range(100): 
    optimizer.zero_grad()
    output = model(x)

    loss = criterion(output, y)
    surr_term = grads_params_add(diff_dict, serialize_params(model))
    loss = loss + surr_term
    
    loss.backward()
    optimizer.step()
    print('Epoch {}: loss {}'.format(epoch, loss.item()))
# https://discuss.pytorch.org/t/how-to-add-a-loss-term-from-hidden-layers/47804

print(serialize_params(model))



