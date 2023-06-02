import numpy as np
import os
np.set_printoptions(precision=6, linewidth=np.inf)
seed = 1
rng = np.random.default_rng(seed=seed)

from src.server_model import BaseFederated 
from src.network_optim import NetworkOptim
from src.network_params import epsilon_0, L_Lipschitz, gamma_cv, xi_factor


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
        
        net_optim = NetworkOptim(num_users, num_samples, data_size, velocity, ts_duration, delta_lr)
        return net_optim
    
    def save_model(self): 
        r""" Save the federated learning model"""
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        save_dir = os.path.join(parent_dir, 'models', self.dataset, f's{str(self.sce_idx)}')
        if not os.path.exists(save_dir): 
            os.makedirs(save_dir)
        
        self.fed_model.save_model(save_dir)
        print(f"Model saved in {save_dir}")
    
    def run(self):
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
        
        remain_eps = epsilon_0
        remain_tau = tau 
        ground = 0 
        optimize = False

        while remain_eps < 1 and remain_tau > 0: # or remain_tau > 0 
            eta_n, t_n = self.net_optim.optimize_network(remain_eps, remain_tau, ground, is_uav, is_dynamic, optimize)

            num_lrounds = self.net_optim.calc_num_lrounds(eta_n)
            self.fed_model.train(int(num_lrounds), ground)

            # calculate instataneous global accuracy 
            eps_n = 1 - (1 - eta_n) * (gamma_cv ** 2) * xi_factor / (2 * (L_Lipschitz ** 2))

            num_grounds = self.net_optim.calc_num_grounds(eta=eta_n)
            if int(num_grounds) == 1: 
                print(f"Done!")
                break
            # update epsilon_0, t_max
            remain_eps = remain_eps / eps_n 
            remain_tau = remain_tau - t_n
            ground += 1 
            print(f"ground = {ground} remain_eps = {remain_eps}\tremain_tau = {remain_tau}")
            # update location 
            self.net_optim.update_channel_gains()