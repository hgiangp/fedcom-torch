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
        self.fed_model = BaseFederated(model, params, dataset)
        self.net_optim = self.init_netoptim(num_users, velocity, ts_duration, params['learning_rate'])
        print("SystemModel __init__!")
    
    def init_netoptim(self, num_users, velocity, ts_duration, delta_lr): 
        r""" Network Optimization Model"""  
        num_samples = self.fed_model.get_num_samples()
        msize = self.fed_model.get_smodel()
        data_size = np.array([msize for _ in range(num_users)])
        
        net_optim = NetworkOptim(num_users, num_samples, data_size, velocity, ts_duration, self.sce_idx, delta_lr)
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
        self.fed_model.train(num_epochs=int(num_lrounds), ground=iter)
        self.net_optim.update_channel_gains()
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
        self.net_optim.update_channel_gains()

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
    
    def test_new_design(self):
        eps_g = 1e-3 
        L_Lipschitz = 5 # Lipschitz constant of the loss function
        gamma_cv = 3 # strongly convex constant of the loss function
        xi = 1

        scenario_idx = self.sce_idx
        tau = self.tau 

        # scenario = 4 
        is_uav = True  
        is_dynamic = True 

        if scenario_idx == 1: 
            is_uav = False 
            is_dynamic = False
        if scenario_idx == 2: 
            is_uav = False 
        if scenario_idx == 3: 
            is_dynamic = False
        
        remain_eps = eps_g
        remain_tau = tau 
        ground = 0 

        while remain_eps < 1: # or remain_tau > 0 
            eta_n, t_n = self.net_optim.optimize_new_design(remain_eps, remain_tau, ground, is_uav, is_dynamic)
            eps_n = 1 - (1 - eta_n) * (gamma_cv ** 2) * xi / (2 * (L_Lipschitz ** 2))
            
            # update epsilon_0, t_max
            remain_eps = remain_eps / eps_n 
            remain_tau = remain_tau - t_n
            ground += 1 

            # update location 
            self.net_optim.update_channel_gains()
