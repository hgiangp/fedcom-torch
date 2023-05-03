import numpy as np 
np.set_printoptions(precision=6, linewidth=np.inf)

from server_model import BaseFederated 
from network_optim import NetworkOptim
from system_utils import * 

seed = 1
rng = np.random.default_rng(seed=seed)

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
    
    def train_dyni(self, idx_sce): 
        t_min, decs = self.net_optim.initialize_feasible_solution() # eta = 0.317, t_min = 66.823        

        tau, t0 = 40, 0.12
        print(f"system_model train() tau = {tau}\tt0 = {t0}")

        iter = 0 # TODO: printing only 
        while 1: 
            print(f"Round {iter}\n-------------------------------") 
            if idx_sce == 4: 
                a_n, num_lrounds, num_grounds, t_total = self.net_optim.optimize_network_bs_uav(tau, ground=iter)
            if idx_sce == 2:
                a_n, num_lrounds, num_grounds, t_total = self.net_optim.optimize_network_bs(tau, ground=iter)
            
            self.fed_model.train(num_epochs=int(num_lrounds), ground=iter)

            # check stop condition
            tau = tau - t_total
            if a_n < 1 or tau < t_min: 
                break 
            # not stop, update location for the next global round 
            print("update_location") 
            self.net_optim.update_channel_gains()
            iter += 1

        print("Done!")
    
    def train_bs_fixedi(self):
        tau, t0 = 40, 0.12
        print(f"system_model train() tau = {tau}\tt0 = {t0}")

        iter = 0
        print(f"Round {iter}\n-------------------------------") 
        a_n, num_lrounds, num_grounds, t_total = self.net_optim.optimize_network_bs(tau, ground=iter)
        max_round = int(num_grounds)

        # FL training 
        for iter in range(1, max_round): 
            print(f"Round {iter}\n-------------------------------")    
            # Calculate energy consumption in the next iteration 
            t_total = self.net_optim.update_n_print(ground=iter)
            # TODO: view number of global rounds
            self.fed_model.train(num_epochs=int(num_lrounds), ground=iter)
            # not stop, update location for the next global round 
            self.net_optim.update_channel_gains()

        print("Done!")

    def train_bs_uav_fixedi(self):
        tau, t0 = 40, 0.12
        print(f"system_model train() tau = {tau}\tt0 = {t0}")
        
        # Optimize network at the first iteration 
        iter = 0 # TODO: printing only 
        print(f"Round {iter}\n-------------------------------") 
        a_n, num_lrounds, num_grounds, t_total = self.net_optim.optimize_network_bs_uav(tau, ground=iter)
        self.fed_model.train(num_epochs=int(num_lrounds), ground=iter)
        
        # iter = 1 
        max_round = int(num_grounds)
        # FL training 
        for iter in range(1, max_round): 
            print(f"Round {iter}\n-------------------------------")    
            # Calculate energy consumption in the next iteration 
            tau = tau - t_total
            a_n, num_lrounds, num_grounds, t_total = self.net_optim.optimize_network_bs_uav_fixedi(tau, iter)
            self.fed_model.train(num_epochs=int(num_lrounds), ground=iter)
            # not stop, update location for the next global round 
            self.net_optim.update_channel_gains()
        
        print("Done!")    

def test(idx_sce=4): 
    sm = SystemModel(updated_dist=2.5)
    if idx_sce == 4 or idx_sce == 2: 
        sm.train_dyni(idx_sce)
    if idx_sce == 3: 
        sm.train_bs_uav_fixedi() 
    if idx_sce == 1: 
        sm.train_bs_fixedi()
    
def main(): 
    sce_idx = read_options()['sce_idx']
    print(sce_idx)
    test(sce_idx)

if __name__=="__main__": 
    main()