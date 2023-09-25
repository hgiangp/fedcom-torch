import numpy as np
import os
np.set_printoptions(precision=6, linewidth=np.inf)
seed = 1
rng = np.random.default_rng(seed=seed)

from src.server_model import BaseFederated 
from src.network_optim import NetworkOptim
from src.network_params import epsilon_0, L_Lipschitz, gamma_cv


class SystemModel: 
    def __init__(self, params, model, dataset, num_users=10, velocity=11, ts_duration=0.4):
        # transfer parameters to self
        for key, val in params.items(): setattr(self, key, val)
        self.fed_model = BaseFederated(model, params, dataset)
        self.net_optim = self.init_netoptim(num_users, velocity, ts_duration, params)
        self.xi_factor = params['xi_factor']
        print("SystemModel __init__!")
    
    def init_netoptim(self, num_users, velocity, ts_duration, params): 
        r""" Network Optimization Model"""  
        num_samples = self.fed_model.get_num_samples()
        msize = self.fed_model.get_model_size()
        data_size = np.array([msize for _ in range(num_users)])
        
        net_optim = NetworkOptim(num_users, num_samples, data_size, params, velocity, ts_duration)
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
        meps_req =  [0.0136476 ,0.01690827,0.02031416,0.02370447,0.02700873,0.03020571,
        0.0341951 ,0.03801859,0.04169272,0.04523653,0.04866241,0.0519851 ,
        0.05521621,0.05836424,0.06143846,0.06444603,0.06739239,0.07028555,
        0.07313031,0.0759306 ,0.07869239,0.08141893,0.08473556,0.08801229,
        0.09125358,0.09446239,0.09764383,0.10080191,0.10393901,0.10705895,
        0.11016283,0.11325297,0.11633372,0.11940587,0.12247119,0.12553153,
        0.12858901,0.13105968,0.13352906,0.13599814,0.13846786,0.14093976,
        0.14341336,0.14588994,0.14837025,0.15085406,0.15334244,0.15583644,
        0.15833586,0.16084153,0.16335518,0.16587582,0.16840402,0.17094086,
        0.17348673,0.17604199,0.17860669,0.18118195,0.1837687 ,0.18636616,
        0.18897582,0.19159789,0.19423246,0.19687988,0.19954092,0.20221614,
        0.20490536,0.20761014,0.21032936,0.21306503,0.2158163 ,0.21858395,
        0.22136843,0.22417066,0.22699108,0.22982916,0.23337743,0.23695521,
        0.24056386,0.24420461,0.24787782,0.25158484,0.25532651,0.25910416,
        0.26291775,0.26676812,0.2706571 ,0.27458435,0.27855254,0.28256179,
        0.28661375,0.29070881,0.29484831,0.2990333 ,0.30326453,0.30754329,
        0.31187095,0.31624865,0.32067683,0.32515773,0.32969241,0.33428232,
        0.33892829,0.34363223,0.34839499,0.35321924,0.35810543,0.36305445,
        0.36806862,0.37315015,0.37829917,0.38351819,0.38881005,0.39417479,
        0.39961409,0.40513161,0.41072808,0.41640579,0.42216689,0.42801307,
        0.43394681,0.43997025,0.44608577,0.45229645,0.45860329,0.46501004,
        0.47151947,0.47813319,0.48616506,0.49435474,0.5027081 ,0.50984046,
        0.51850921,0.52591369,0.53344793,0.5411144 ,0.54891755,0.55686141,
        0.56495051,0.57318844,0.58157903,0.59012742,0.59883886,0.60771808,
        0.61677005,0.62599984,0.63541281,0.64501514,0.65481205,0.66481118,
        0.673006  ,0.68133736,0.68980957,0.69842833,0.70719521,0.71611483,
        0.72519059,0.73442887,0.74383234,0.75340565,0.76315409,0.7730822 ,
        0.78319616,0.79350098,0.80400125,0.81470318,0.82561354,0.83673905,
        0.84808311,0.85965497,0.87146214,0.88351006,0.89580509,0.90835978,
        0.9211808 ,0.93427374,0.94765021,0.96131784,0.97528901,0.989572  ,
        1.0041803 ,1.01912137,1.03440914,1.05005516,1.06607207,1.08247478,
        1.09927465,1.11649081,1.13413224,1.15222159,1.17077385,1.19451815,
        1.21904574,1.24439398,1.27060541,1.29772677,1.32580634,1.35489213,
        1.38504482,1.41632228,1.44878695,1.48251003,1.51756573,1.55403457,
        1.59955423,1.64737825,1.69768413,1.75068695,1.80659748,1.86565579,
        1.92815115,1.9943851 ,2.06470108,2.13949469,2.23253542]
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

        # optimze = True
        optimize = True # self.optim == 1 
        optim_power = False 
        optim_freq = False 
        if self.optim == 2:
            optimize = False
            optim_freq = True
        elif self.optim == 3: 
            optimize = False
            optim_power = True
        elif self.optim == 4:
            optimize = False 

        remain_eps = epsilon_0
        remain_tau = tau 
        ground = 0 

        test_acc_online = False
        if test_acc_online: 
            remain_eps = epsilon_0
            acccumulated_acc = 1
            loss_f = 0.1 
            loss_previous = 2.5

        test_acc_offline = True
        if test_acc_offline: 
            remain_eps = meps_req[ground]
        
        while 1: 
            if test_acc_online:             
                remain_eps /= acccumulated_acc
            
            eta_n, t_n = self.net_optim.optimize_network(remain_eps, remain_tau, ground, is_uav, is_dynamic, optimize, optim_freq, optim_power)
            
            num_lrounds = self.net_optim.calc_num_lrounds(eta_n)
            loss_n = self.fed_model.train(int(num_lrounds), ground)

            
            if test_acc_online: # online estimating accuracy
                eps_n_estimated = (loss_n - loss_f)/(loss_previous - loss_f)
                acccumulated_acc *= eps_n_estimated
                print(f"ground = {ground} acccumulated_acc = {acccumulated_acc}")
                loss_previous = loss_n 

            # theoretically calculate instataneous global accuracy  
            eps_n = 1 - (1 - eta_n) * (gamma_cv ** 2) * self.xi_factor / (2 * (L_Lipschitz ** 2))
            remain_eps = remain_eps / eps_n
            ground += 1 
            if test_acc_offline: # offline estimating accuracy
                remain_eps = meps_req[ground]

            remain_tau = remain_tau - t_n # online estimate the time 
            print(f"ground = {ground} eps_n = {eps_n}")
            print(f"ground = {ground} remain_eps = {remain_eps}\tremain_tau = {remain_tau}")
                        
            # update location 
            self.net_optim.update_channel_gains()

            # check last ground of s4 
            num_grounds = int(self.net_optim.calc_num_grounds(eta=eta_n))
            print(f"ground = {ground} num_grounds = {num_grounds}")

            if num_grounds == 1: # dynamic optimization 
                print(f"Done! int(num_grounds) == 1")
                break

            if (scenario_idx == 1 or scenario_idx == 3) and (num_grounds == ground): 
                print("(scenario_idx == 1 or scenario_idx == 3) and (num_grounds == ground)")
                break
            if (scenario_idx == 2 or scenario_idx == 4) and (remain_eps > 1 or remain_tau < 0):
                print("(scenario_idx == 2 or scenario_idx == 4) and (remain_eps > 1 or remain_tau < 0)")
                break  

    def test(self): 
        for ground in range(500): 
            self.fed_model.train(num_epochs=5, ground=ground)
    