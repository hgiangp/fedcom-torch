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