import numpy as np
np.set_printoptions(precision=6, linewidth=np.inf)

from network_params import * 
from network_utils import * 
from location_model import LocationModel
from optimization import NewtonOptim

class NetworkOptim:
    def __init__(self, num_users, num_samples, data_size, updated_dist):
        self.num_users = num_users
        self.num_samples = num_samples # np.array (num_users, )
        self.data_size = data_size
        self.updated_dist = updated_dist
        self.loc_model = LocationModel(num_users, updated_dist)
        self.uav_gains, self.bs_gains = self.calc_channel_gains() # init channel gains
        self.an = a_0 # initialize with current round = 0 
        
        # For saving the optimal result 
        self.eta = 0.01
        self.freqs = freq_max 
        self.powers = power_max 
        self.decs = np.zeros(num_users)

    def calc_channel_gains(self): 
        xs, ys = self.loc_model.get_location()

        uav_gains = self.calc_uav_gains(xs, ys)
        bs_gains = self.calc_bs_gains(xs, ys)        
        
        return uav_gains, bs_gains

    def update_an(self, ground=0): 
        self.an =  a_0 + ground * a_alpha
        # print(f"self.an = {self.an}")

    def update_channel_gains(self): 
        r"Update uav_gains, bs_gains of users"
        self.loc_model.update_location()
        self.uav_gains, self.bs_gains = self.calc_channel_gains() # Updated channel gains 

    def calc_comp_energy(self, num_rounds, freqs): 
        r""" Calculate local computation energy
        Args: 
            num_rounds: number of local rounds (K_l): int 
            freqs: cpu frequencies: np array (N, ) 
        Return:
            ene_cp: np array (N, ) 
        """
        ene_cp = num_rounds * k_switch * C_n * self.num_samples * (freqs**2)
        return ene_cp 

    def calc_comp_time(self, num_rounds, freqs): 
        r""" Calculate local computation time
        Args: 
            num_rounds: number of local rounds (K_l): int 
            freqs: cpu frequencies: np array (N, ) 
        Return: 
            time_cp: np array (N, )
        """
        time_cp = num_rounds * C_n * self.num_samples / freqs 
        return time_cp 

    def calc_trans_time(self, decs, powers): 
        r""" Calculate communications time 
        Args:
            decs: relay-node selection decisions (decs==1: uav else bs): type=np.array, shape=(N, )
        Return:
            time_co: dtype=np.array, shape=(N, )
        """
        gains = decs * self.uav_gains + (1 - decs) * self.bs_gains # (N, )
        ti_penalty = decs * delta_t # (N, )

        time_co = (self.data_size / (bw * np.log2(1 + (powers * gains / N_0)))) + ti_penalty 
        return time_co

    def calc_trans_energy(self, decs, powers): 
        r""" Calcualte communications energy
        Args: 
            powers: transmission powers: type=np.array, shape=(N, )
            decs: relay-node selection decisions: type=np.array, shape=(N, )
        Return: 
            ene_co: dtype=np.array, shape=(N, )
        """
        ti_penalty = decs * delta_t
        coms_tis = self.calc_trans_time(decs, powers) - ti_penalty
        ene_co = powers * coms_tis 
        return ene_co

    def calc_total_energy(self, eta, freqs, decs, powers): 
        num_lrounds = v * math.log2(1 / eta)
        num_grounds = self.an / (1 - eta)

        ene_coms = self.calc_trans_energy(decs, powers)
        ene_comp = self.calc_comp_energy(num_lrounds, freqs)
        print(f"ene_coms = {ene_coms}\nene_comp = {ene_comp}")

        energy = num_grounds * (ene_coms + ene_comp)
        print(f"ene_total = {energy}")
        return energy
    
    def calc_total_energy_fixedi(self, num_lrounds, num_grounds): 
        ene_coms = self.calc_trans_energy(self.decs, self.powers)
        ene_comp = self.calc_comp_energy(num_lrounds, self.freqs)
        print(f"ene_coms = {ene_coms}\nene_comp = {ene_comp}")

        energy = num_grounds * (ene_coms + ene_comp)
        print(f"ene_total = {energy}")
        return energy

    def calc_total_time(self, eta, freqs, decs, powers): 
        num_lrounds = v * math.log2(1/eta)
        num_grounds = self.an / (1 - eta)
        
        ti_comp = self.calc_comp_time(num_lrounds, freqs)
        ti_coms = self.calc_trans_time(decs, powers)
        print(f"ti_comp = {ti_comp}\nti_coms = {ti_coms}")
        ti = num_grounds * (ti_coms + ti_comp)
        return ti 
    
    def find_bound_eta(self, freqs, decs, powers, tau):
        r""" TODO 
        Args: 
        Return: 
        """ 
        af = self.an * self.calc_comp_time(1, freqs) * v / math.log(2)
        bf = self.an * self.calc_trans_time(decs, powers)

        eta_min, eta_max = solve_bound_eta(af, bf, tau)
        return eta_min, eta_max
        
    def solve_optimal_eta(self, freqs, decs, powers): 
        r""" Find the optimal local accuracy eta by Dinkelbach method
        Args: 
        Return:
            Optimal eta 
        """

        bf = self.an * self.calc_trans_energy(decs, powers).sum()
        af = v * self.an * self.calc_comp_energy(num_rounds=1, freqs=freqs).sum() / math.log(2)

        eta = dinkelbach_method(af, bf)
        print(f"eta = {eta}")

        return eta
    
    def init_decisions(self): 
        # Solve optimal decision decs
        t_co_uav = self.calc_trans_time(1, power_max)
        t_co_bs = self.calc_trans_time(0, power_max)

        difference = t_co_uav + delta_t - t_co_bs
        idx = np.argpartition(difference, max_uav)[:max_uav]
        idx_uav = idx[np.where(difference[idx] < 0)]

        decs_opt = np.zeros(shape=num_users, dtype=int)
        decs_opt[idx_uav] = 1
        
        return decs_opt

    def initialize_feasible_solution(self): 
        r"""
        Method: bisection
        Args:
        Return: 
            eta, T_min
        """
        ### Find af, bf efficient 
        decs_opt = self.init_decisions()

        # TODO: check delta_t

        # Solve optimal eta for current resource allocation (fixed f_max, p_max, decs)
        t_cp = self.calc_comp_time(1, freq_max)
        t_co = self.calc_trans_time(decs_opt, power_max)
        af = self.an * t_cp * v / math.log(2)
        bf = self.an * t_co

        # Solve optimal eta by dinkelbach method with argument related to the total time 
        af_opt, bf_opt = af.sum(), bf.sum()
        eta_opt = dinkelbach_method(af_opt, bf_opt) 
        print(f"eta_opt = {eta_opt}")

        ### Iterative for tau 
        t_min, t_max = 10, 120 # seconds 
        acc = 1e-4
        iter = 0 

        while 1:
            tau = (t_min + t_max) / 2.0 
            eta_min, eta_max = solve_bound_eta(af, bf, tau)       
            # print(f"iter = {iter}\teta_min = {eta_min}\teta_opt = {eta_opt}\teta_max = {eta_max}") 

            # Check feasible condition for current tau  
            if eta_opt > eta_min and eta_opt < eta_max: # feasible solution  
                t_max = tau 
            else:
                t_min = tau 
            # print(f"iter = {iter}\tt_min = {t_min}\ttau = {tau}\tt_max = {t_max}")

            if (t_max - t_min)/t_max < acc: 
                break
            iter += 1
        
        print(f"initialize_feasible_solution eta = {eta_opt}, tau = {tau}")  
        return tau, decs_opt

    def solve_freqs_powers(self, eta, decs, tau): 
        r"""Solve powers, freqs for each user by newton's method"""

        gains = decs * self.uav_gains + (1 - decs) * self.bs_gains
        penalty_time = decs * delta_t # (N, )
        num_lrounds = v * math.log2(1 / eta)

        # Calculate opt coefficient for optimizer 
        opt_as = self.data_size / bw * math.log(2) # (N, )
        opt_bs = gains / N_0 # (N, )
        opt_cs = num_lrounds * C_n * self.num_samples # (N, )
        opt_tau = tau * (1-eta)/self.an - penalty_time # (N, ) penalty time for chosing uav # broadcasting   
        print(f"opt_as = {opt_as}\nopt_bs = {opt_bs}\nopt_cs = {opt_cs}\nopt_tau = {opt_tau}")

        z_min = 1/np.log(1 + power_max * opt_bs) 
        t_min = 1/freq_max

        # Normalization variables for faster convergence 
        norm_factor = 1e9

        # Initiate results 
        opt_powers = np.zeros(shape=num_users)
        opt_freqs = np.zeros(shape=num_users)

        for i in range(num_users): 
            print(f"NewtonOptim USER = {i}")
            opt = NewtonOptim(a=opt_as[i], b=opt_bs[i], c=opt_cs[i], tau=opt_tau[i], 
                    kappa=k_switch, norm_factor=norm_factor, z_min=z_min[i], t_min=t_min)
            inv_ln_power, inv_freq = opt.newton_method()

            # Update results 
            opt_freqs[i] = norm_factor * 1/inv_freq
            opt_powers[i]  = 1/opt_bs[i] * (np.exp(1/inv_ln_power) - 1)

        print(f"opt_freqs = {opt_freqs}\nopt_powers = {opt_powers}")
        return opt_freqs, opt_powers

    def optimize_network(self, tau, ground=0): 
        r""" Solve the relay-node selection and resource allocation problem
        Args: 
        Return: 
            (eta, freqs, decs, powers)
        """
        eps_eta = 1e-3 # tolerance accuracy of eta

        # Initialize a feasible solution 
        freqs = np.ones(num_users) * freq_max
        powers = np.ones(num_users) * power_max
        decs = self.init_decisions()
        eta = 0.01
        
        obj_prev = self.calc_total_energy(eta, freqs, decs, powers).sum()
        print(f"obj_prev = {obj_prev}")

        # Repeat
        iter = 0
        while 1: 
            # Tighten the bound of eta 
            eta_min, eta_max = self.find_bound_eta(freqs, decs, powers, tau)

            # stop condition for boundary eta 
            if abs(eta_max - eta_min) < eps_eta: 
                print("Done!")
                break

            # Solve eta
            eta = self.solve_optimal_eta(freqs, decs, powers) 
            
            # Check eta boundary condition 
            if eta > eta_max: 
                eta = eta_max 
            elif eta < eta_min: 
                eta = eta_min 

            # Solve powers p, freqs f and apply heursitic method for choosing decisions x 
            
            # check with all connecting to uav
            decs = np.ones(shape=num_users, dtype=int)  
            uav_freqs, uav_powers = self.solve_freqs_powers(eta, decs, tau)
            uav_ene = self.calc_total_energy(eta, uav_freqs, decs, uav_powers)
            print(f"uav_ene = {uav_ene}")

            # check with all connecting to bs
            decs = np.zeros(shape=num_users, dtype=int)
            bs_freqs, bs_powers = self.solve_freqs_powers(eta, decs, tau)
            bs_ene = self.calc_total_energy(eta, bs_freqs, decs, bs_powers)
            print(f"bs_ene = {bs_ene}")
            
            # find the better decision
            difference = uav_ene - bs_ene
            print(f"difference = {difference}")
            # https://stackoverflow.com/questions/34226400/find-the-index-of-the-k-smallest-values-of-a-numpy-array
            idx = np.argpartition(difference, max_uav)[:max_uav]
            idx_uav = idx[np.where(difference[idx] < 0)]
            
            # update decisions 
            decs = np.zeros(num_users, dtype=int)
            decs[idx_uav] = 1 # (N, )

            # update powers, freqs corresponding to the current optimal decision 
            powers = decs * uav_powers + (1 - decs) * bs_powers
            freqs = decs * uav_freqs + (1 - decs) * bs_freqs

            # Check stop condition
            obj = self.calc_total_energy(eta, freqs, decs, powers).sum()
            print(f"optimize_network iter = {iter} obj = {obj}")
            
            print(f"eta = {eta}")
            print(f"decs = {decs}")
            print(f"freqs = {freqs}")
            print(f"powers = {powers}") 

            if (abs(obj_prev - obj) < acc) or iter == iter_max: 
                print("Done!")
                break

            obj_prev = obj
            iter += 1 
        
        # Finishing 
        num_lrounds = v * math.log2(1/eta)
        num_grounds = self.an / (1 - eta)

        # update optimal result for class 
        self.eta = eta 
        self.freqs = freqs
        self.powers = powers 
        self.decs = decs
        
        # calculate time, energy consumption at the current iteration 
        t_co = self.calc_trans_time(self.decs, self.powers).sum()
        e_co = self.calc_trans_energy(self.decs, self.powers).sum()
        
        t_cp = self.calc_comp_time(num_lrounds, self.freqs).sum()   
        e_cp = self.calc_comp_energy(num_lrounds, self.freqs).sum()

        print("At round {} average t_co: {} average t_cp: {}".format(ground, t_co/num_users, t_cp/num_users))
        print("At round {} e_co: {} e_cp: {}".format(ground, e_co, e_cp))
        print("At round {} eta: {}".format(ground, eta))  
        print("At round {} a_n: {}".format(ground, self.an))

        # update a_n for calculating the next global iteration  
        self.update_an(ground=ground)
        
        return self.an, num_lrounds, num_grounds # (i, n, a_n)

        # Stop 
        # return (eta, freqs, decs, powers)

    def solve_freqs_fake(self, eta, decs, tau):
        # power fixed at power_max, decs fixed 
        # calculate coms time 
        t_co = self.calc_trans_time(decs, power_max) # (N, )

        # calculate number of local, global rounds 
        num_lrounds = v * math.log2(1/eta)
        num_grounds = self.an / (1-eta)

        # calculate computation time for one local_round 
        t_cp_1 = (tau / num_grounds - t_co) / num_lrounds # (N, )

        # Calculate optimal freqs     
        freqs = C_n * self.num_samples / t_cp_1 # (N, )
        return freqs

    def optimize_network_fake(self, tau, decs, ground=0):
        # Initialize a feasible solution 
        freqs = np.ones(num_users) * freq_max
        powers = np.ones(num_users) * power_max
        # decs = np.zeros(num_users) # decs = np.array([1, 0, 1, 1, 0, 0, 1, 1, 0, 0], dtype=int)

        # Repeat
        iter = 0
        # tau = 50

        eta = 0.01
        obj_prev = self.calc_total_energy(eta, freqs, decs, powers).sum()
        print(f"obj_prev = {obj_prev}")

        while 1: 
            # find bound eta 
            eta_min, eta_max = self.find_bound_eta(freqs, decs, powers, tau)
            print(f"eta_min = {eta_min}\teta_max = {eta_max}")
            # Solve eta
            eta = self.solve_optimal_eta(freqs, decs, powers) 
            
            if eta > eta_max: 
                eta = eta_max 
            elif eta < eta_min: 
                eta = eta_min 

            # Solve freqs f
            freqs = self.solve_freqs_fake(eta, decs, tau)

            # Check stop condition
            obj = self.calc_total_energy(eta, freqs, decs, powers).sum()
            print(f"optimize_network iter = {iter} obj = {obj}")
            print(f"eta = {eta}")
            print(f"freqs = {freqs}")

            if (abs(obj_prev - obj) < acc) or iter == iter_max: 
                print("Done!")
                break

            obj_prev = obj
            iter += 1 

        num_lrounds = v * math.log2(1/eta)
        num_grounds = self.an / (1 - eta)

        # update optimal result for class 
        self.eta = eta 
        self.freqs = freqs
        self.powers = powers 
        self.decs = decs
        
        # calculate time, energy consumption at the current iteration 
        t_co = self.calc_trans_time(self.decs, self.powers).sum()
        e_co = self.calc_trans_energy(self.decs, self.powers).sum()
        
        t_cp = self.calc_comp_time(num_lrounds, self.freqs).sum()   
        e_cp = self.calc_comp_energy(num_lrounds, self.freqs).sum()

        print("At round {} average t_co: {} average t_cp: {}".format(ground, t_co/num_users, t_cp/num_users))
        print("At round {} e_co: {} e_cp: {}".format(ground, e_co, e_cp))
        print("At round {} eta: {}".format(ground, eta))  
        print("At round {} a_n: {}".format(ground, self.an))

        # update a_n for calculating the next global iteration  
        self.update_an(ground=ground)
        
        return self.an, num_lrounds, num_grounds # (i, n, a_n)

    def calc_bs_gains(self, xs, ys): 
        r""" Calculate propagation channel gains, connect to bs 
        Args: 
            dists: distance: dtype=np.array, shape=(N, )
        Return: 
            bs_gains: dtype=np.array, shape=(N, )
        """
        # Calculate the distances to the basestation
        dists = np.sqrt(((xs - x_bs) ** 2) + ((ys - y_bs) ** 2))
        bs_gains = A_d * np.power((c / (4 * np.pi * f_c * dists)), de_r) 
        
        # LOG TRACE 
        print(f"dists_bs = {dists}")
        print(f"bs_gains = {bs_gains}")

        return bs_gains 

    def calc_uav_gains(self, xs, ys): 
        r""" Calculate propagation channel gains, connect to uav
        Args: 
            xs: x location of vehs: dtype=np.array, shape=(N, )
            ys: y location of vehs: dtype=np.array, shape=(N, )
        Return:
            uav_gains: dtype=np.array, shape=(N, )
        """
        dists = np.sqrt((xs ** 2) + (ys ** 2) + (z_uav ** 2)) # (N, )
        thetas = 180 / np.pi * np.arctan(z_uav / dists) # (N, )
        pLoSs = 1 / (1 + a_env * np.exp( -b_env * (thetas - a_env))) # (N, )
        uav_gains = ((pLoSs + alpha * (1 - pLoSs)) * g_0) / (np.power(dists, de_u)) # (N, )

        # LOG TRACE
        print(f"dists_uav = {dists}")
        print(f"uav_gains = {uav_gains}")

        return uav_gains

def test_with_location():
    num_users = 10 
    num_samples = np.array([117, 110, 165, 202, 454, 112, 213, 234, 316, 110])
    data_size = np.array([s_n for _ in range(num_users)])

    netopt = NetworkOptim(num_users, num_samples, data_size, updated_dist=500) 

    freqs = np.array([1, 0.6, 2, 0.3, 0.4, 0.5, 1.5, 1.2, 0.3, 1]) * 1e9 # max = 2GHz
    num_rounds = 30

    cp_ene = netopt.calc_comp_energy(num_rounds, freqs)
    print(f"cp_ene =", cp_ene)
    cp_time = netopt.calc_comp_time(num_rounds, freqs)
    print(f"cp_time = {cp_time}")

    decs = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
    eta = 0.15
    freqs, powers = netopt.solve_freqs_powers(eta, decs, tau=100)

def test_optimize_network(): 
    num_users = 10 
    num_samples = np.array([117, 110, 165, 202, 454, 112, 213, 234, 316, 110])
    data_size = np.array([s_n for _ in range(num_users)])

    netopt = NetworkOptim(num_users, num_samples, data_size, updated_dist=500) 

    eta, freqs, decs, powers = netopt.optimize_network()

    print(f"eta = {eta}")
    print(f"decs = {decs}")
    print(f"freqs = {freqs}")
    print(f"powers = {powers}") 
    
    ene = netopt.calc_total_energy(eta, freqs, decs, powers)
    ti = netopt.calc_total_time(eta, freqs, decs, powers)
    print(f"calc_total_energy ene = {ene}\ncalc_total_time ti = {ti}")

def test_optimize_network_fake(): 
    num_users = 10 
    num_samples = np.array([117, 110, 165, 202, 454, 112, 213, 234, 316, 110])
    data_size = np.array([s_n for _ in range(num_users)])

    netopt = NetworkOptim(num_users, num_samples, data_size, updated_dist=500) 
    num_lrounds, num_grounds = netopt.optimize_network_fake()

    print("update_location")
    netopt.update_channel_gains()
    num_lrounds, num_grounds = netopt.optimize_network_fake()

def test_feasible_solution():
    num_users = 10 
    num_samples = np.array([117, 110, 165, 202, 454, 112, 213, 234, 316, 110])
    data_size = np.array([s_n for _ in range(num_users)])

    netopt = NetworkOptim(num_users, num_samples, data_size, updated_dist=500) 
    eta, t_min = netopt.initialize_feasible_solution()

    print(f"eta = {eta}")
    print(f"t_min = {t_min}")


if __name__=='__main__': 
    # test_with_location()
    # test_optimize_network()
    # test_feasible_solution()
    test_optimize_network_fake()