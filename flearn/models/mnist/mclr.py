import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import functorch
from torch.nn.utils import _stateless

class Net(nn.Module): 
    def __init__(self, model_dim=(5, 3)): 
        super(Net, self).__init__()
        self.linear = nn.Linear(model_dim[0], model_dim[1])

    def forward(self, x): 
        outputs = self.linear(x)
        return outputs
    
    def get_params(self): 
        # https://discuss.pytorch.org/t/difference-between-detach-clone-and-clone-detach/34173
        return {k: v.data.detach().clone() for k, v in zip(self.state_dict(), self.parameters())}

    def set_params(self, params: dict): 
        r"https://medium.com/@mrityu.jha/understanding-the-grad-of-autograd-fc8d266fd6cf"
        self.load_state_dict(params)
    
    def get_grads(self): 
        return {k: v.grad.detach().clone() for k, v in zip(self.state_dict(), self.parameters())}
    
    def get_model_size(self):  
        torch_float = 32 
        msize = sum(param.numel() for param in self.parameters()) * torch_float
        return msize

class Model: 
    r""" Multi-class Logistic Regression: 1 Linear Layer 
        Loss function: Cross Entropy
        https://aaronkub.com/2020/02/12/logistic-regression-with-pytorch.html
        https://discuss.pytorch.org/t/multi-class-cross-entropy-loss-and-softmax-in-pytorch/24920/9
    """
    def __init__(self, model_dim=(5, 3), lr=1e-3, xi=1, device=torch.device('cpu')):
        self.model = Net(model_dim)
        self.model.to(device) # move model to device

        # init optimizer and loss function 
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr) # TODO
        self.loss_fn = nn.CrossEntropyLoss()
        self.xi = xi # TODO
    
    
    def get_params(self): 
        return self.model.get_params()

    def set_params(self, params): 
        self.model.set_params(params)
    
    def get_grads(self): 
        return self.model.get_grads()
    
    def calc_surrogate_term(self, delta_grads: dict): 
        sum = 0.0
        lparams = self.model.get_params()
        for k in lparams.keys(): 
            sum += (lparams[k] * delta_grads[k]).sum()
        return sum

    def train(self, num_epochs: int, train_loader: DataLoader): 
        r"""Train model"""
        # print('train is_cuda: ', next(self.model.parameters()).is_cuda)
        for t in range(num_epochs): 
            # print(f"Epoch {t+1}\n-------------------------------")
            for batch, (X, y) in enumerate(train_loader):
                output = self.model(X)
                loss = self.loss_fn(output, y)

                # Back propagation
                self.optimizer.zero_grad()
                loss.backward()  # update gradients
                # self.optimizer.step()  # update model parameters


    def train_fed(self, num_epochs: int, train_loader: DataLoader, ggrads: dict):
        r""" Train local model for fedearated learning 
        Args: 
            num_epochs: int: number of local rounds 
            train_loader: DataLoader of the training set
            ggrads: global gradient of the previous global round
        Return:
            soln: dict: model.parameters()
        """
        # print('train_fed is_cuda: ', next(self.model.parameters()).is_cuda)
        # Calculate delta grads at the begining of ground 
        if ggrads is not None: 
            delta_grads = {}
            lgrads = self.model.get_grads() # curent local grads 
            for k in lgrads.keys(): 
                delta_grads[k] = self.xi * ggrads[k] - lgrads[k]

        # Train the model 
        size = len(train_loader.dataset)
        for t in range(num_epochs): 
            # print(f"Epoch {t+1}\n-------------------------------")
            for batch, (X, y) in enumerate(train_loader):
                output = self.model(X)
                loss = self.loss_fn(output, y)

                if ggrads is not None: 
                    # Calculate surrogate term and update the loss 
                    # https://discuss.pytorch.org/t/how-to-add-a-loss-term-from-hidden-layers/47804
                    surr_term = self.calc_surrogate_term(delta_grads)
                    loss = loss + surr_term

                # Back propagation
                self.optimizer.zero_grad()
                loss.backward()  # update gradients
                self.optimizer.step()  # update model parameters

                # print log
                loss, current = loss.item(), (batch+1)*len(X)
                # print(f"train() {t}-{batch} loss: {loss:>7f}  [{current:>5d}/{size:>5d}]  surr: {surr_term:>7f}")
    
        # print("Done!")
        # soln = self.get_params()
        # return soln 
        
    def calculate_hessian(self, dataloader: DataLoader): 
        import functorch
        from torch.nn.utils import _stateless
        import time

        start = time.time()

        model = self.model 
        hmin = 0.0; hmax = 0.0

        for batch, (X, y) in enumerate(dataloader):
            def loss(params):
                out: torch.Tensor = _stateless.functional_call(model, {n: p for n, p in zip(names, params)}, X)
                return self.loss_fn(out, y)

            names = list(n for n, _ in model.named_parameters())
            hess = functorch.hessian(loss)(tuple(model.parameters())) 
            # print(hess)
            hess = torch.cat([e.flatten() for h in hess for e in h]) # flatten
            print(f"torch.max(hess), torch.min(hess) = {torch.max(hess)} {torch.min(hess)}")
            hmin += torch.min(hess)
            hmax += torch.max(hess)
            # break 
        hmin = hmin/len(dataloader)
        hmax = hmax/len(dataloader)
        print(f"max(hess), min(hess) = {hmax} {hmin}")
        end = time.time()
        print(end - start)
    
    def calculate_jacobian(self, train_data): 
        model = self.model 
        y = torch.tensor(train_data['y'], dtype=torch.long) 
        X = torch.tensor(train_data['x'])

        names = list(n for n, _ in model.named_parameters())

        def loss(params): 
            out: torch.Tensor = _stateless.functional_call(model, {n: p for n, p in zip(names, params)}, X)
            return self.loss_fn(out, y)
        
        J = torch.autograd.grad(loss(model.parameters()), tuple(model.parameters()))
        J = torch.cat([e.flatten() for e in J]) # flatten
        
        print("J.norm(2) = ", J.norm(2).item())

    def calculate_hessian2(self, train_data): 
        import functorch
        from torch.nn.utils import _stateless
        import time

        start = time.time()

        model = self.model 
        hmin = 0.0; hmax = 0.0
        y = torch.tensor(train_data['y'], dtype=torch.long) 
        X = torch.tensor(train_data['x'])

        def loss(params):
            out: torch.Tensor = _stateless.functional_call(model, {n: p for n, p in zip(names, params)}, X)
            return self.loss_fn(out, y)

        names = list(n for n, _ in model.named_parameters())
        hess = functorch.hessian(loss)(tuple(model.parameters())) 
        # print(hess)
        hess = torch.cat([e.flatten() for h in hess for e in h]) # flatten
        print(f"torch.max(hess), torch.min(hess) = {torch.max(hess)} {torch.min(hess)}")
        end = time.time()
        print(end - start)    

    def test(self, dataloader: DataLoader, debug=False): 
        # print('test is_cuda: ', next(self.model.parameters()).is_cuda)
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
        
        if debug: 
            print(f"Test Error: Accuracy: {correct:>0.1f}, Avg loss: {test_loss:>8f}")
        
        return  correct, test_loss

    def predict(self, dataloader: DataLoader, debug=False): 
        size = len(dataloader.dataset) # number of samples of train set or test set 
        num_batches = len(dataloader)
        y_pred = []
        
        with torch.no_grad(): 
            for X, y in dataloader: 
                preds = self.model(X).argmax(1).tolist()
                y_pred.append(preds)
    
        # Flatten the list 
        y_pred_flat = [item for sublist in y_pred for item in sublist]

        if debug:
            print(f"y_pred = {y_pred_flat}")
            print(f"predict() len(y_pred) = {len(y_pred_flat)}")
        
        return y_pred_flat
    
    def save(self, save_dir):
        torch.save(self.model.state_dict(), f'{save_dir}/model_weights.pth') 

def print_model(parameters: dict): 
    for name, params in parameters.items():
        print(name)
        for param in params: 
            print(param)

def calculate_model_norm(parameters: dict):
    total_norm = 0.0 
    for name, p in parameters.items():
        # print(name)
        param_norm = p.norm(2)
        total_norm += param_norm.item() ** 2
        # print(total_norm)
    total_norm = total_norm ** (1. / 2)
    return total_norm

def calculate_model_diff(parameters1: dict, parameters2: dict):
    print("parameters1", calculate_model_norm(parameters1))
    print("parameters2", calculate_model_norm(parameters2))
    diff = {k: torch.zeros_like(v) for k, v in parameters1.items()}
    for name, p1 in parameters1.items():
        diff[name] = p1 - parameters2[name]
    return diff 

def test():
    import torch
    import os, sys 

    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.path.append(parent_dir)
    from src.custom_dataset import test_load_data, load_dataloader
    
    train_data, test_data = test_load_data(user_id=1, dataset_name='mnist')
    train_loader, test_loader = load_dataloader(train_data, test_data)
    
    model = Model(model_dim=(784, 10), lr=0.1)
    lparams = model.get_params()
    ggrads = {k: torch.zeros_like(v) for k, v in lparams.items()}
    
    model.train(10, train_loader)

    grad = model.get_grads()
    param = model.get_params()
    for rounds in [10, 50, 100, 100, 100]: 
        print("param before train", calculate_model_norm(param))
        model.train(rounds, train_loader)
        print("param after train", calculate_model_norm(param))
        grad_new = model.get_grads()
        param_new = model.get_params()
        print("param after get param_new", calculate_model_norm(param))
        dparam = calculate_model_diff(param, param_new)
        dgrad = calculate_model_diff(grad, grad_new)

        param_norm = calculate_model_norm(dparam)
        print("param_norm", param_norm)
        grad_norm = calculate_model_norm(dgrad)
        print("grad_norm", grad_norm)
        grad = grad_new
        param = param_new

    model.calculate_hessian(train_loader)
    model.calculate_hessian2(train_data)
    # model.test(train_loader) # train error and loss 
    # model.test(test_loader)  # test error and loss

if __name__=='__main__': 
    test()