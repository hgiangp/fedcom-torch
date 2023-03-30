import numpy as np 

from server_model import BaseFederated 
from network_optim import NetworkOptim
from system_utils import * 

class SystemModel: 
    def __init__(self, mod_name='CustomLogisticRegression', mod_dim=(5, 3), dataset_name='synthetic', num_users=10):
        self.fed_model = self.init_federated(mod_name, mod_dim, dataset_name)
        self.net_optim = self.init_netoptim(num_users, self.fed_model)
        print("SystemModel __init__!")

    def init_federated(self, mod_name, mod_dim, dataset_name):
        r""" TODO
        """
        model = load_model(mod_name)
        dataset = load_data(dataset_name)
        fed_model = BaseFederated(model, mod_dim, dataset)
        return fed_model 
    
    def init_netoptim(self, num_users, fed_model): 
        r""" Network Optimization Model"""  
        num_samples = fed_model.get_num_samples()
        msize = fed_model.get_mod_size()
        data_size = np.array([msize for _ in range(num_users)])
        
        net_optim = NetworkOptim(num_users, num_samples, data_size, updated_dist=10)
        return net_optim
    
    def train(self): 
        num_rounds = 30 # TODO: check num_global_rounds 
        for t in range(num_rounds):
            print(f"Round {t+1}\n-------------------------------") 
            num_local_rounds, num_global_rounds = self.net_optim.optimize_network_fake()
            print(f"iter = {t}\tnum_local_rounds = {num_local_rounds}\tnum_global_rounds = {num_global_rounds}")
            self.fed_model.train(num_epochs=int(num_local_rounds), curr_round=t)
            
            print("update_location") # for the next global round
            self.net_optim.update_channel_gains()
        
        print("Done!")

def test(): 
    sm = SystemModel()
    sm.train()

if __name__=="__main__": 
    test()