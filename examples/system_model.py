import numpy as np 

from server_model import BaseFederated 
from network_optim import NetworkOptim
from system_utils import * 

class SystemModel: 
    def __init__(self, mod_name='CustomLogisticRegression', mod_dim=(5, 3), dataset_name='synthetic', num_users=10, updated_dist=10):
        self.fed_model = self.init_federated(mod_name, mod_dim, dataset_name)
        self.net_optim = self.init_netoptim(num_users, updated_dist, self.fed_model)
        print("SystemModel __init__!")

    def init_federated(self, mod_name, mod_dim, dataset_name):
        r""" TODO
        """
        model = load_model(mod_name)
        dataset = load_data(dataset_name)
        fed_model = BaseFederated(model, mod_dim, dataset)
        return fed_model 
    
    def init_netoptim(self, num_users, updated_dist, fed_model): 
        r""" Network Optimization Model"""  
        num_samples = fed_model.get_num_samples()
        msize = fed_model.get_mod_size()
        data_size = np.array([msize for _ in range(num_users)])
        
        net_optim = NetworkOptim(num_users, num_samples, data_size, updated_dist)
        return net_optim
    
    def train(self): 
        t_min, decs = self.net_optim.initialize_feasible_solution() # eta = 0.317, t_min = 66.823
        
        tau = int(3 * t_min) # > t_min (= t_min + const) e.g t_min + t_min/10 TODO 
        t0 = t_min / 200 # TODO: set value of t0 
        
        print(f"system_model train() tau = {tau}\tt0 = {t0}\tt_min = {t_min}")
        iter = 0 # TODO: printing only 
        while 1: 
            print(f"Round {iter}\n-------------------------------") 
            a_n, num_local_rounds, num_global_rounds = self.net_optim.optimize_network_fake(tau, decs, cround=iter)
            print("At round {} local rounds: {}".format(iter, num_local_rounds))
            print("At round {} global rounds: {}".format(iter, num_global_rounds))
            
            # TODO: view number of global rounds
            self.fed_model.train(num_epochs=int(num_local_rounds), cround=iter)

            # check stop condition 
            if a_n < 0: 
                break 

            # not stop, update location for the next global round 
            print("update_location") 
            self.net_optim.update_channel_gains()
            tau = tau - t0
            iter += 1

        print("Done!")
    
    def train_fixedi(self):
        t_min, decs = self.net_optim.initialize_feasible_solution() # eta = 0.317, t_min = 66.823
        
        tau = int(3 * t_min) # > t_min (= t_min + const) e.g t_min + t_min/10 TODO 
        t0 = t_min / 200 # TODO: set value of t0
        print(f"system_model train() tau = {tau}\tt0 = {t0}\tt_min = {t_min}")
        
        # Optimize network at the first iteration 
        iter = 0 # TODO: printing only 

        a_n, num_local_rounds, num_global_rounds = self.net_optim.optimize_network_fake(tau, decs, cround=0)
        print("At round {} local rounds: {}".format(iter, num_local_rounds))
        print("At round {} global rounds: {}".format(iter, num_global_rounds))

        max_round = int(num_global_rounds) - 1 
        # num_grounds = int(num_global_rounds)

        # FL training 
        while 1: 
            print(f"Round {iter}\n-------------------------------")             
            # TODO: view number of global rounds
            self.fed_model.train(num_epochs=int(num_local_rounds), cround=iter)

            # check stop condition 
            if iter == max_round: 
                break 

            # not stop, update location for the next global round 
            print("update_location") 
            self.net_optim.update_channel_gains()
            iter += 1

            # Calculate energy consumption in the next iteration 
            obj = self.net_optim.calc_total_energy_fixedi(int(num_local_rounds), 1).sum()
            print("At round {} energy consumption: {}".format(iter, obj))
            
        print("Done!")

def test(): 
    sm = SystemModel(updated_dist=5)
    sm.train()

def test_fixedi(): 
    sm = SystemModel(updated_dist=5)
    sm.train_fixedi()

if __name__=="__main__": 
    test()
    # test_fixedi()