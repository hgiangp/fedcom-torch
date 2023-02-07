from net_params import * 
import numpy as np 
from scipy.special import lambertw

def calc_comp_energy(num_rounds, num_samples, freqs): 
    r""" Calculate local computation energy
    Args: 
        num_rounds: number of local rounds (K_l): int 
        num_samples: number of samples: np array (N, ) N: num_users 
        freqs: cpu frequencies: np array (N, ) 
    Return:
        ene_cp: np array (N, ) 
    """
    ene_cp = num_rounds * k_switch * C_n * num_samples * (freqs**2)
    return ene_cp 

def calc_comp_time(num_rounds, num_samples, freqs): 
    r""" Calculate local computation time
    Args: 
        num_rounds: number of local rounds (K_l): int 
        num_samples: number of samples: np array (N, ) N: num_users 
        freqs: cpu frequencies: np array (N, ) 
    Return: 
        time_cp: np array (N, )
    """
    time_cp = num_rounds * C_n * num_samples / freqs 
    return time_cp 

def calc_trans_time(decs, data_size, uav_gains, bs_gains, powers): 
    r""" Calculate communications time 
    Args:
        decs: relay-node selection decisions (decs==1: uav else bs): dtype=np.array, shape=(N, )
        data_size: data transmission size: dtype=np.array, shape=(N, )
        uav_gains, bs_gains: propagation channel gains, dtype=np.array, shape=(N, ) 
    Return:
        time_co: dtype=np.array, shape=(N, )
    """
    gains = np.array([uav_gains[i] if decs[i] == 1 else bs_gains[i] for i in range(num_users)]) # (N, )
    ti_penalty = decs * delta_t # (N, )

    time_co = (data_size / (bw * np.log2(1 + (powers * gains / N_0)))) + ti_penalty 
    return time_co

def calc_trans_energy(decs, data_size, uav_gains, bs_gains, powers): 
    r""" Calcualte communications energy
    Args: 
        powers: transmission powers: dtype=np.array, shape=(N, )
        coms_tis: transmission times: dtype=np.array, shape=(N, )
    Return: 
        ene_co: dtype=np.array, shape=(N, )
    """
    coms_tis = calc_trans_time(decs, data_size, uav_gains, bs_gains, powers)
    ene_co = powers * coms_tis 
    return ene_co

def calc_bs_gains(dists): 
    r""" Calculate propagation channel gains, connect to bs 
    Args: 
        dists: distance: dtype=np.array, shape=(N, )
    Return: 
        bs_gains: dtype=np.array, shape=(N, )
    """
    # TODO: define location of bstation, calcualte distance based on xs, ys inside function
    bs_gains = A_d * np.power((c / (4 * np.pi * f_c * dists)), de_r) 
    return bs_gains 

def cals_uav_gains(xs, ys): 
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
    return uav_gains 

def find_bound_eta(decs, data_size, uav_gains, bs_gains, powers, num_samples): 
    r""" Solve Lambert W function to find the boundary of eta (eta_n_min, eta_n_max)
    Args: 
        decs: relay-node selection decisions (decs==1: uav else bs): dtype=np.array, shape=(N, )
        data_size: data transmission size: dtype=np.array, shape=(N, )
        uav_gains, bs_gains: propagation channel gains, dtype=np.array, shape=(N, )
        powers: transmission powers, dtype=np.array, shape=(N, ) 
        num_samples: number of samples, dtype=np.array, shape=(N, )
    Return: 
        eta_min, eta_max
    """
    # Calculate communcitions time
    af = a * calc_trans_time(decs, data_size, uav_gains, bs_gains, powers) # (N, )
    bf = calc_comp_time(num_rounds=1, num_samples=num_samples, freqs=freq_max) * v / math.log(2) # (N, )
    x = (af - tau)/bf - lambertw(- tau/bf * np.exp((af - tau)/bf)) # (N, )
    bound_eta = np.exp(x.real) # (N, )
    # print("bound_eta =", bound_eta)
    
    return bound_eta

def solve_optimal_eta(decs, data_size, uav_gains, bs_gains, powers, freqs, num_samples): 
    r""" Find the optimal local accuracy eta by Dinkelbach method
    Args: 
    Return:
        Optimal eta 
    """
    acc = 1e-4 
    zeta = 0.1
    eta = 1 

    bf = a * calc_trans_energy(decs, data_size, uav_gains, bs_gains, powers).sum()
    af = v * a * calc_comp_energy(num_rounds=1, num_samples=num_samples, freqs=freqs).sum() / math.log(2)
    
    h_prev = af * math.log(1/eta) + bf - zeta * (1 - eta)
    iter = 0 # TODO: remove iter writing log after test with verified params
    
    while 1:
        eta = af / zeta # calculate temporary optimal eta 
        zeta = ((af * math.log(1/eta)) + bf) / (1 - eta) # update zeta 
        h_curr = af * math.log(1/eta) + bf - zeta * (1 - eta) # check stop condition 
        print(f"iter = {iter}, h_prev = {h_prev}, h_curr = {h_curr}, h_curr / h_prev={h_curr / h_prev}")
        if (h_prev != 0) and (h_curr / h_prev < acc): 
            break
        h_prev = h_curr   
        iter += 1 # TODO: remove 
  
    return eta

def solve_optimal_freq(eta, num_samples): 
    r""" Find the optimal local cpu freq
    Args: 
        eta: optimal eta found from the previous step 
        num_samples: number of local data samples, dtype=np.array, shape=(N, ) 
    Return: 
        Optimal freqs: dtype=np.array, shape=(N, )
    """
    t_co_max = tau * (1 - eta) / a - v * math.log2(1/eta) * calc_comp_time(num_rounds=1, num_samples=num_samples, freqs=freq_max) # (N, )
    # print(f"t_co_max={t_co_max}") # TODO
    freqs = a * v * C_n * num_samples * math.log2(1/eta) / (tau * (1 - eta) - a * t_co_max) # (N, )
    # print(f"a = {a * v * C_n * num_samples * math.log2(1/eta)}") # TODO
    # print(f"b = {(tau * (1 - eta) - a * t_co_max)}") # TODO
    return freqs

def solve_optimal_power(eta, num_samples, data_size, gains, ti_penalty): 
    r""" Find the optimal transmission power (uav, or bs)
    Args: 
        data_size: (N, )
    Return: 
        power: optimal power (uav/bs), dtype=np.array, shape=(N, )
    """
    num_rounds = v * math.log2(1 / eta)
    t_co = tau * (1 - eta) / a - ti_penalty - \
        calc_comp_time(num_rounds=num_rounds, num_samples=num_samples, freqs=freq_max) # (N, )
    
    opt_powers = N_0 / gains * (2 * np.exp(data_size / (bw * t_co)) - 1) # (N, )
    return opt_powers

def calc_total_energy(eta, freqs, decs, powers, num_samples, data_size, uav_gains, bs_gains): 
    num_local_rounds = v * math.log2(eta)
    num_global_rounds = a / (1 - eta)

    energy = num_global_rounds * (calc_trans_energy(decs, data_size, uav_gains, bs_gains, powers) + \
        num_local_rounds * calc_comp_energy(num_local_rounds, num_samples, freqs))
    return energy 

def optimize_network(num_samples, data_size, uav_gains, bs_gains): 
    r""" Solve the relay-node selection and resource allocation problem
    Args: 
    Return: 

    """
    # Initialize a feasible solution 
    freqs = np.ones(num_users) * freq_max
    powers = np.ones(num_users) * power_max
    
    tco_uav = calc_trans_time(decs=1, data_size=data_size, uav_gains=uav_gains, bs_gains=bs_gains, powers=powers) # (N, )
    tco_bs = calc_trans_time(decs=0, data_size=data_size, uav_gains=uav_gains, bs_gains=bs_gains, powers=powers) # (N, )
    decs = np.array([1 if tco_uav[i] < tco_bs[i] else 0 for i in range(num_users)])

    eta = 0.0
    iter = 0

    obj_prev = calc_total_energy(eta, freqs, decs, powers, num_samples, data_size, uav_gains, bs_gains)

    # Repeat 
    while 1: 
        # Tighten the bound of eta 
        bound_eta = find_bound_eta(decs, data_size, uav_gains, bs_gains, powers, num_samples)

        # Solve eta
        eta = solve_optimal_eta(decs, data_size, uav_gains, bs_gains, powers, freqs, num_samples) 

        # Solve freqs f
        freqs = solve_optimal_freq(eta, num_samples)

        # Solve powers p and apply heursitic method for choosing decisions x 
        uav_powers = solve_optimal_power(eta, num_samples, data_size, uav_gains, delta_t)
        bs_powers = solve_optimal_power(eta, num_samples, data_size, bs_gains, 0)

        tco_uav = calc_trans_time(decs=1, data_size=data_size, uav_gains=uav_gains, bs_gains=bs_gains, powers=uav_powers)
        tco_bs = calc_trans_time(decs=0, data_size=data_size, uav_gains=uav_gains, bs_gains=bs_gains, powers=bs_powers)

        decs = np.array([1 if tco_uav[i] < tco_bs[i] else 0 for i in range(num_users)], dtype=int)
        powers = decs * uav_powers + (1 - decs) * bs_powers

        # Check stop condition
        obj = calc_total_energy(eta, freqs, decs, powers, num_samples, data_size, uav_gains, bs_gains)
        print(f"iter = {iter} obj = {obj}")

        if (iter == iter_max) or (obj_prev - obj < acc): 
            print("Done!")
            break
        obj_prev = obj
        iter += 1 

    # Stop 
    return (eta, freqs, decs, powers)

def test(): 
    # num_rounds = 2
    num_samples = np.random.randint(low=1, high=3, size=(num_users, ))
    # freqs = np.random.randint(low=4, high=8, size=(num_users, ))
    # ene_cp = calc_comp_energy(num_rounds, num_samples, freqs)
    # print(f"ene_cp = {ene_cp}\ndtype={type(ene_cp)}\nsize={ene_cp.shape}") 
    decs = np.random.randint(low=0, high=2, size=num_users)
    uav_gains = np.random.randint(low=1, high=5, size=num_users)
    bs_gains = np.random.randint(low=1, high=4, size=num_users)
    data_size = np.random.randint(low=0, high=4, size=num_users)
    powers = np.random.randint(low=2, high=4, size=num_users)
    freqs = np.random.randint(low=2, high=4, size=num_users)

    # x = find_bound_eta(decs, data_size, uav_gains, bs_gains, powers, num_samples)
    # print(f"x = {x}")
    # solve_optimal_eta(decs, data_size, uav_gains, bs_gains, powers, freqs, num_samples)
    eta = 0.9
    # opt_freqs = solve_optimal_freq(eta, num_samples)
    # print(f"freqs = {opt_freqs}")

if __name__=='__main__': 
    test()