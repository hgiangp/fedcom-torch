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
        # t_min, decs = self.net_optim.initialize_feasible_solution() # eta = 0.317, t_min = 66.823        
        # tau = int(4 * t_min) # > t_min (= t_min + const) e.g t_min + t_min/10 TODO 
        # t0 = t_min / 250 # TODO: set value of t0 

        tau, t0 = 40, 0.2
        print(f"system_model train() tau = {tau}\tt0 = {t0}")
        decs = np.random.randint(low=0, high=2, size=10)

        iter = 0 # TODO: printing only 
        while 1: 
            print(f"Round {iter}\n-------------------------------") 
            a_n, num_lrounds, num_grounds = self.net_optim.optimize_network_fake(tau, decs, ground=iter)
            print("At round {} local rounds: {}".format(iter, num_lrounds))
            print("At round {} global rounds: {}".format(iter, num_grounds))
            print("At round {} tau: {}".format(iter, num_grounds))
            
            # TODO: view number of global rounds
            self.fed_model.train(num_epochs=int(num_lrounds), ground=iter)

            # check stop condition
            tau = tau - t0
            if a_n < 1 or tau < 0: 
                break 

            # not stop, update location for the next global round 
            print("update_location") 
            self.net_optim.update_channel_gains()
            iter += 1

        print("Done!")
    
    def train_fixedi(self):
        t_min, decs = self.net_optim.initialize_feasible_solution() # eta = 0.317, t_min = 66.823
        
        tau = int(6 * t_min) # > t_min (= t_min + const) e.g t_min + t_min/10 TODO 
        t0 = t_min / 600 # TODO: set value of t0
        print(f"system_model train() tau = {tau}\tt0 = {t0}\tt_min = {t_min}")
        
        # Optimize network at the first iteration 
        iter = 0 # TODO: printing only 

        a_n, num_lrounds, num_grounds = self.net_optim.optimize_network_fake(tau, decs, ground=0)
        print("At round {} local rounds: {}".format(iter, num_lrounds))
        print("At round {} global rounds: {}".format(iter, num_grounds))

        max_round = int(num_grounds) - 1 
        # num_grounds = int(num_grounds)

        # FL training 
        while 1: 
            print(f"Round {iter}\n-------------------------------")             
            # TODO: view number of global rounds
            self.fed_model.train(num_epochs=int(num_lrounds), ground=iter)

            # check stop condition 
            if iter == max_round: 
                break 

            # not stop, update location for the next global round 
            print("update_location") 
            self.net_optim.update_channel_gains()
            iter += 1

            # Calculate energy consumption in the next iteration 
            obj = self.net_optim.calc_total_energy_fixedi(int(num_lrounds), 1).sum()
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