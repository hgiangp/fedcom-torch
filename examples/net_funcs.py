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

def calc_trans_energy(powers, coms_tis): 
    r""" Calcualte communications energy
    Args: 
        powers: transmission powers: dtype=np.array, shape=(N, )
        coms_tis: transmission times: dtype=np.array, shape=(N, )
    Return: 
        ene_co: dtype=np.array, shape=(N, )
    """
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

def solve_optimal_eta(): 
    r""" Find the optimal local accuracy eta by Dinkelbach method
    Args: 
    Return:
        Optimal eta 
    """
    # Initialize 
    iter, acc = 0, 1e-4 
    zeta = 0.1
    ti_co = 
    
    # Repeat


    pass 

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

    x = find_bound_eta(decs, data_size, uav_gains, bs_gains, powers, num_samples)
    print(f"x = {x}")

if __name__=='__main__': 
    test()