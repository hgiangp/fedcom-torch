import copy 
import torch
import numpy as np 

from client_model import Client
from network_params import s_n

class BaseFederated: 
    def __init__(self, model, model_dim, dataset):
        r"""Federated Learning Model
        Args: dim = (in_dim, out_dim) # input, output dimension
        """
        self.client_model = model(model_dim) # (5, 3) # (784, 10)
        self.clients = self.setup_clients(self.client_model, dataset)
        self.latest_model = self.client_model.get_params() # TODO: latest_model updated 
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
    
    def calc_dissimilarity(self, agrads, wgrads):
        difference = 0
        for _, grads in wgrads: 
            for k in grads.keys(): 
                # difference += torch.square(agrads[k] - grads[k]).sum()
                difference += (agrads[k] - grads[k]).sum()
        difference = difference * 1.0 / len(self.clients)
        
        return difference
    
    def train(self, num_epochs=20, ground=0):
        r"""
        Args: 
            num_epochs: number of local rounds # network opt 
            num_rounds: number of global rounds 
        """
        # collect num_samples, grads from clients 
        wgrads = []
        for c in self.clients: 
            wgrads.append(c.get_wgrads())

        # aggregate the clients grads
        agrads = self.aggregate(wgrads)

        difference = self.calc_dissimilarity(agrads=agrads, wgrads=wgrads)
        print('gradient difference: {}'.format(difference))

        # clients train the local surrogate models
        wsolns = [] # buffer for receiving clients' solution
        for c in self.clients:
            wsolns.append(c.train(num_epochs, agrads)) 

        # aggregate the global parameters and broadcast to all uses 
        self.latest_model = self.aggregate(wsolns)
        for c in self.clients: 
            c.set_params(self.latest_model)

        # Test model
        stats = self.test() # (list num_samples, list total_correct)  
        stats_train = self.train_error_and_loss() # (list num_samples, list total_correct, list losses)
        print("At round {} accuracy: {}".format(ground, np.sum(stats[1])*1.0/np.sum(stats[0])))
        print("At round {} training accuracy: {}".format(ground, np.sum(stats_train[1])*1.0/np.sum(stats_train[0])))
        print("At round {} training loss: {}".format(ground, np.dot(stats_train[2], stats_train[0])*1.0/np.sum(stats_train[0])))
        print("At round {} test loss: {}".format(ground, np.dot(stats[2], stats[0])*1.0/np.sum(stats[0])))
        
    
    def test(self):
        num_samples = []
        tot_correct = []
        losses = []

        for c in self.clients: 
            ns, ct, cl = c.test()
            num_samples.append(ns)
            tot_correct.append(ct*1.0)
            losses.append(cl*1.0)

        return num_samples, tot_correct, losses
    
    def train_error_and_loss(self): 
        num_samples = []
        tot_correct = []
        losses = []

        for c in self.clients: 
            ns, ct, cl = c.train_error_and_loss()
            num_samples.append(ns)
            tot_correct.append(ct*1.0)
            losses.append(cl*1.0)
        
        return num_samples, tot_correct, losses
    
    def get_num_samples(self): 
        num_samples = np.array([client.num_samples for client in self.clients])
        print(f"num_samples = {num_samples}")
        return num_samples 
    
    def get_smodel(self): 
        r""" Get model size"""
        msize = s_n # msize = self.client_model.get_model_size()
        print(f"msize = {msize}")
        return msize
    
def test_aggregate(server):
    soln1 = {k: torch.ones_like(v) for k, v in server.latest_model.items()} 
    soln2 = {k: torch.ones_like(v)*2 for k, v in server.latest_model.items()} 
    soln3 = {k: torch.ones_like(v)*3 for k, v in server.latest_model.items()} 
    wsolns = [(1, soln1), (2, soln2), (3, soln3)]
    soln = server.aggregate(wsolns)
    print(f"soln=\n{soln}")

def test_calc_msize(server: BaseFederated): 
    print(server.latest_model)
    return server.get_smodel()
    
def test(model_dim=(5, 3), dataset_name='synthetic'):
    model_name = 'CustomLogisticRegression'
    from system_utils import load_model, load_data 
    model = load_model(model_name)
    dataset = load_data(dataset_name)
    
    t = BaseFederated(model, model_dim, dataset)
    # test_aggregate(t)
    # t.train(num_rounds=10)
    # t.get_num_samples()
    # t.get_mod_size()
    # print("test_calc_msize()", test_calc_msize(t))
    num_rounds = 250 
    for i in range(num_rounds):
        t.train(num_epochs=10, ground=i)
    

if __name__=="__main__": 
    test() # test synthetic dataset 
    # test(model_dim=(784, 19), dataset_name='mnist') # test mnist dataset 