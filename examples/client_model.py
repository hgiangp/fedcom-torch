import torch.optim as optim
import torch.nn as nn 
import torch 

import copy 

from custom_model import CustomLogisticRegression
from custom_dataset import load_data, test_load_data
from torch_utils import add_params_grads

class Client(object): 
    def __init__(self, id, train_data={'x':[],'y':[]}, test_data={'x':[],'y':[]}, model=None):
        self.id = id 
        self.model = model #CustomLogisticRegression()
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-2)
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_loader, self.test_loader = load_data(train_data, test_data)
        print(f"id = {id}, model = {model}")
    
    def get_params(self): 
        r"""Get model.parameters()
        Return: 
            dict {'state_dict': tensor(param.data)}
        """
        return self.model.get_params()

    def set_params(self, params_dict=None):
        r"""Set initial params to model parameters()
        Args: 
            model_params: dict {'state_dict': model.parameters().data}
        Return: None 
            set model.parameters = model_params 
        Hint: https://medium.com/@mrityu.jha/understanding-the-grad-of-autograd-fc8d266fd6cf
        """
        self.model.set_params(params_dict)
    
    def get_grads(self): 
        r"""Get model.parameters() gradients 
        Return: 
            dict {'state_dict': tensor(param.grad)}
        """
        return self.model.get_grads()
    
    def train(self, num_epochs, diff_dict):
        size = len(self.train_loader.dataset)

        for t in range(num_epochs): 
            print(f"Epoch {t+1}\n-------------------------------")
            for batch, (X, y) in enumerate(self.train_loader):
                output = self.model(X)
                loss = self.loss_fn(output, y)

                # Calculate surrogate term and update the loss
                # Hint: https://discuss.pytorch.org/t/how-to-add-a-loss-term-from-hidden-layers/47804
                surr_term = add_params_grads(self.get_params(), diff_dict)
                loss = loss + surr_term

                # Back propagation
                self.optimizer.zero_grad()
                loss.backward()  # update gradients
                self.optimizer.step()  # update model parameters

                # print log
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                # print(f"model.parameters() =\n{self.model.get_params()}")
    
        print("Done!")
        # print(f"In train():\nself.get_params() =\n{self.get_params()}\nmodel.parameters() =\n{self.model.get_params()}")

        # soln = get_params(model=model)
        # return (size, soln)

def test_train(): 
    print("test_train")

    model = CustomLogisticRegression(input_dim=5, output_dim=3)
    
    # FRIST USER
    user_id = 0
    train_data1, test_data1 = test_load_data(user_id)
    model1 = copy.deepcopy(model)
    client = Client(user_id, train_data1, test_data1, model1)
    pretrain_params = copy.deepcopy(client.get_params())
    print(f"pretrain_params =\n{pretrain_params}")
    diff_dict = {k: torch.randn_like(v) for k, v in zip(pretrain_params.keys(), pretrain_params.values())}
    num_epochs = 5 
    client.train(num_epochs, diff_dict)
    posttrain_params = client.get_params()
    
    print(f"pretrain_params =\n{pretrain_params}")
    print(f"posttrain_params =\n{posttrain_params}")

    print("##########################\n##########################")
    # SECOND USER 
    user_id = 1
    train_data2, test_data2 = test_load_data(user_id)
    model2 = copy.deepcopy(model)
    client2 = Client(user_id, train_data2, test_data2, model2)
    pretrain_params2 = copy.deepcopy(client2.get_params())
    print(f"pretrain_params2 =\n{pretrain_params2}")
    client2.train(num_epochs, diff_dict)
    posttrain_params2 = client2.get_params()
    print(f"posttrain_params2 =\n{posttrain_params2}")
    print(f"client.get_params() =\n{client.get_params()}")
    
    # epoches = 100 
    # for t in range(epoches): 
    #     print(f"Epoch {t+1}\n-------------------------------")
    #     train_loop(csmodel.train_loader, csmodel.model, csmodel.loss_fn, csmodel.optimizer, diff_dict)
    #     test_loop(csmodel.test_loader, csmodel.model, csmodel.loss_fn)
    # print("Done!")


if __name__ == '__main__':
    test_train() 