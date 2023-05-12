import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class Net(nn.Module): 
    def __init__(self, model_dim=(5, 3)): 
        super(Net, self).__init__()
        self.linear = nn.Linear(model_dim[0], model_dim[1])

    def forward(self, x): 
        outputs = self.linear(x)
        return outputs
    
    def get_params(self): 
        return {k: v.data for k, v in zip(self.state_dict(), self.parameters())}

    def set_params(self, params: dict): 
        r"https://medium.com/@mrityu.jha/understanding-the-grad-of-autograd-fc8d266fd6cf"
        self.load_state_dict(params)
    
    def get_grads(self): 
        return {k: v.grad for k, v in zip(self.state_dict(), self.parameters())}
    
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
    def __init__(self, model_dim=(5, 3)):
        self.model = Net(model_dim)
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-3) # TODO
        self.loss_fn = nn.CrossEntropyLoss()
        self.xi = 1 # TODO
    
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
        for t in range(num_epochs): 
            # print(f"Epoch {t+1}\n-------------------------------")
            for batch, (X, y) in enumerate(train_loader):
                output = self.model(X)
                loss = self.loss_fn(output, y)

                # Back propagation
                self.optimizer.zero_grad()
                loss.backward()  # update gradients
                self.optimizer.step()  # update model parameters


    def train_fed(self, num_epochs: int, train_loader: DataLoader, ggrads: dict):
        r""" Train local model for fedearated learning 
        Args: 
            num_epochs: int: number of local rounds 
            train_loader: DataLoader of the training set
            ggrads: global gradient of the previous global round
        Return:
            soln: dict: model.parameters()
        """
        # Calculate delta grads at the begining of ground 
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

                # Calculate surrogate term and update the loss https://discuss.pytorch.org/t/how-to-add-a-loss-term-from-hidden-layers/47804
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
    
    def test(self, dataloader: DataLoader, debug=False): 
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

def test():
    import torch
    import os, sys 

    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.path.append(parent_dir)
    from src.custom_dataset import test_load_data, load_dataloader
    
    train_data, test_data = test_load_data(user_id=1)
    train_loader, test_loader = load_dataloader(train_data, test_data)
    
    model = Model(5, 3)
    lparams = model.get_params()
    ggrads = {k: torch.zeros_like(v) for k, v in lparams.items()}
    # pretrain 
    model.train(5, train_loader)
    model.train_fed(30, train_loader, ggrads)
    # train error and loss 
    model.test(train_loader)
    # test error and loss
    model.test(test_loader)

if __name__=='__main__': 
    test()