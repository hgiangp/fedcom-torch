import numpy as np
import os

np.set_printoptions(precision=6, linewidth=np.inf)

from src.server_model import BaseFederated 
from src.network_optim import NetworkOptim

seed = 1
rng = np.random.default_rng(seed=seed)

class SystemModel: 
    def __init__(self, params, model, dataset, num_users=10, velocity=11, ts_duration=0.4):
        # transfer parameters to self
        for key, val in params.items(): setattr(self, key, val)
        print(f"params['model_params'] = {params['model_params']}")
        self.fed_model = BaseFederated(model, params['model_params'], dataset)
        self.net_optim = self.init_netoptim(num_users, velocity, ts_duration)
        print("SystemModel __init__!")
    
    def init_netoptim(self, num_users, velocity, ts_duration): 
        r""" Network Optimization Model"""  
        num_samples = self.fed_model.get_num_samples()
        msize = self.fed_model.get_smodel()
        data_size = np.array([msize for _ in range(num_users)])
        
        net_optim = NetworkOptim(num_users, num_samples, data_size, velocity, ts_duration, self.sce_idx)
        return net_optim

    def run(self): 
        scenario_idx = self.sce_idx
        tau = self.tau 
        if scenario_idx == 4 or scenario_idx == 2: 
            self.train_dyni(scenario_idx, tau)
        if scenario_idx == 3: 
            self.train_bs_uav_fixedi(tau) 
        if scenario_idx == 1: 
            self.train_bs_fixedi(tau)
      
    
    def train_dyni(self, idx_sce, tau): 
        t_min, decs = self.net_optim.initialize_feasible_solution() # eta = 0.317, t_min = 66.823        

        print(f"system_model train() tau = {tau}")

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
            if a_n < 1 or tau < 1: 
                break 
            # not stop, update location for the next global round 
            print("update_location") 
            self.net_optim.update_channel_gains()
            iter += 1

        print("Done!")
    
    def train_bs_fixedi(self, tau=40):
        print(f"system_model train() tau = {tau}")

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

    def train_bs_uav_fixedi(self, tau=40):
        print(f"system_model train() tau = {tau}")
        
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
            a_n, num_lrounds, num_grounds, t_total = self.net_optim.optimize_network_bs_uav_fixedi(tau, iter)
            tau = tau - t_total
            self.fed_model.train(num_epochs=int(num_lrounds), ground=iter)
            # not stop, update location for the next global round 
            self.net_optim.update_channel_gains()
        
        print("Done!")
    
    def save_model(self): 
        r""" Save the federated learning model"""
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        save_dir = os.path.join(parent_dir, 'models', self.dataset, f's{str(self.sce_idx)}')
        if not os.path.exists(save_dir): 
            os.makedirs(save_dir)
        
        self.fed_model.save_model(save_dir)
        print(f"Model saved in {save_dir}")