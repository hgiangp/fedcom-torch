import importlib
import os 
import copy 
import torch 

from custom_dataset import read_data 
from client_model import Client

class BaseFederated(object): 
    def __init__(self, model, dataset):
        self.client_model = model(input_dim=5, output_dim=3)
        self.clients = self.setup_clients(self.client_model, dataset)
        self.latest_model = self.client_model.get_params()
        print(f"self.latest_model =\n{self.latest_model}")
        print("BaseFederated generated!")
    
    def setup_clients(self, model, dataset): 
        clients, train_data, test_data = dataset
        all_clients = [Client(id, train_data[id], test_data[id], copy.deepcopy(model)) for id in clients]
        return all_clients
    
    def aggregate(self, wsolns):
        r""" 
        Args: wsolns: list of tuple (weight, params_dict), in which 
            weight: number of samples 
            params_dict: dictionary of model.parameters() or model gradients
        Return: 
            averaged soln
        """ 
        total_weight = 0.0
        base = {k: torch.zeros_like(v) for k, v in wsolns[0][1].items()}
        for (w, soln) in wsolns:
            total_weight += w 
            for k, v in soln.items(): 
                base[k] += w * v 
        
        averaged_soln = {k: v / total_weight for k, v in base.items()}
        
        return averaged_soln
    
    def train(self):
        num_rounds = 1
        for t in range(num_rounds): 
            print(f"Round {t+1}\n-------------------------------")
            # collect num_samples, grads from clients 
            wgrads = []
            for c in self.clients: 
                wgrads.append(c.get_grads())

            # aggregate the clients grads
            agrads = self.aggregate(wgrads)

            # broadcast the global params and difference grads 
            for c in self.clients:
                c.calc_diff_grads(agrads)

            # clients train the local surrogate models
            num_epochs = 20 # TODO: network_opt
            wsolns = [] # buffer for receiving clients' solution
            for c in self.clients:
                wsolns.append(c.train(num_epochs)) 
                
            # aggregate the global parameters and broadcast to all uses 
            aparams = self.aggregate(wsolns)
            for c in self.clients: 
                c.set_params(aparams)
        
        print("Done!")

    def test_aggregate(self):
        soln1 = {k: torch.ones_like(v) for k, v in self.latest_model.items()} 
        soln2 = {k: torch.ones_like(v)*2 for k, v in self.latest_model.items()} 
        soln3 = {k: torch.ones_like(v)*3 for k, v in self.latest_model.items()} 
        wsolns = [(1, soln1), (2, soln2), (3, soln3)]
        soln = self.aggregate(wsolns)
        print(f"soln=\n{soln}")
    
def test(): 

    # load the client model 
    model_path = '%s' % ('custom_model')
    mod = importlib.import_module(model_path)
    model = getattr(mod, 'CustomLogisticRegression')

    # load the dataset 
    dataset_name = 'synthetic'
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_dir = os.path.join(parent_dir, 'data', dataset_name, 'data', 'train')
    test_dir = os.path.join(parent_dir, 'data', dataset_name, 'data', 'test')
    dataset = read_data(train_dir, test_dir)

    # TODO: check params 
    t = BaseFederated(model, dataset)
    # t.test_aggregate()

    t.train()

if __name__=="__main__": 
    test()