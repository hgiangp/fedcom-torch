import torch 

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

def train_loop(dataloader, model, loss_fn, optimizer, diff_dict):
    size = len(dataloader.dataset)

    for batch, (X, y) in enumerate(dataloader):
        # print(f'X = {X} \ny = {y}')
        output = model(X)
        loss = loss_fn(output, y)

        # Calculate surrogate term and update the loss
        # Hint: https://discuss.pytorch.org/t/how-to-add-a-loss-term-from-hidden-layers/47804
        surr_term = add_params_grads(get_params(model), diff_dict)
        loss = loss + surr_term

        # Back propagation
        optimizer.zero_grad()
        loss.backward()  # update gradients
        optimizer.step()  # update model parameters

        # print log
        loss, current = loss.item(), batch * len(X)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
def get_params(model): 
    r"""Get model.parameters()
    Return: 
        dict {'state_dict': tensor(param.data)}
    """
    return {k: v.data for k, v in zip(model.state_dict(), model.parameters())}

def set_params(model, model_params=None):
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
    # print(get_params(model))

def get_gradients(model): 
    r"""Get model.parameters() gradients 
    Return: 
        dict {'state_dict': tensor(param.grad)}
    """
    return {k: v.grad for k, v in zip(model.state_dict(), model.parameters())}

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0 
    
    with torch.no_grad(): 
        for X, y in dataloader: 
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= num_batches
    correct /= size 
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")