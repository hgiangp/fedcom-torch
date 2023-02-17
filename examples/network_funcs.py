from network_params import * 
import numpy as np 
from scipy.special import lambertw

from location_model import init_location

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
    ti_penalty = decs * delta_t
    coms_tis = calc_trans_time(decs, data_size, uav_gains, bs_gains, powers) - ti_penalty
    ene_co = powers * coms_tis 
    return ene_co

def calc_bs_gains(xs, ys): 
    r""" Calculate propagation channel gains, connect to bs 
    Args: 
        dists: distance: dtype=np.array, shape=(N, )
    Return: 
        bs_gains: dtype=np.array, shape=(N, )
    """
    # Calculate the distances to the basestation
    dists = np.sqrt(((xs - x_bs) ** 2) + ((ys - y_bs) ** 2))
    print(f"dists_bs = {dists}")
    bs_gains = A_d * np.power((c / (4 * np.pi * f_c * dists)), de_r) 
    return bs_gains 

def calc_uav_gains(xs, ys): 
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
    print(f"dists_uav =", dists)
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
    # print(f"af = {af}")
    # print(f"bf = {bf}")
    # print(f"a = {a}, v = {v}")
    x = (af - tau)/bf - lambertw(- tau/bf * np.exp((af - tau)/bf)) # (N, )
    # print(f"x = {x}")
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
    eta = 1 

    bf = a * calc_trans_energy(decs, data_size, uav_gains, bs_gains, powers).sum()
    af = v * a * calc_comp_energy(num_rounds=1, num_samples=num_samples, freqs=freqs).sum() / math.log(2)

    h_prev = 0
    zeta = af * 2 
    while 1:
        eta = af / zeta # calculate temporary optimal eta
        print(f"eta = {eta}")
        print(f"af = {af}\tbf={bf}\tzeta = {zeta}")
        h_curr = af * math.log(1/eta) + bf - zeta * (1 - eta) # check stop condition
        if (h_prev != 0) and (abs(h_curr - h_prev) < acc): 
            break
        h_prev = h_curr   
        
        zeta = ((af * math.log(1/eta)) + bf) / (1 - eta) # update zeta 
  
    return eta

def solve_powers_freqs(eta, num_samples, data_size, gains, ti_penalty): 
    opt_t_co = (data_size / bw) / (1 + (lambertw(1 / (2 * np.exp(1))).real)) + ti_penalty # optimal communnication time # (N, )
    print(f"opt_t_co = {opt_t_co}")
    opt_powers = N_0 / gains * (2 * np.exp((data_size / bw) / (opt_t_co - ti_penalty)) - 1) # optimal power transmission, (N, )
    
    num_local_rounds = v * np.log2(1/eta)
    max_t_cp = (tau * (1 - eta)/a - opt_t_co) / num_local_rounds
    print(f"max_t_cp = {max_t_cp}")
    opt_freqs = C_n * num_samples / max_t_cp 

    return opt_powers, opt_freqs

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
        (eta, freqs, decs, powers)
    """
    # Initialize a feasible solution 
    freqs = np.ones(num_users) * freq_max
    powers = np.ones(num_users) * power_max
    decs = np.random.randint(low=0, high=2, size=num_users) # random initial decisions
    eta = 0.01
    obj_prev = calc_total_energy(eta, freqs, decs, powers, num_samples, data_size, uav_gains, bs_gains).sum()
    print(f"obj_prev = {obj_prev}")
    # Repeat
    iter = 0 
    while 1: 
        # Tighten the bound of eta 
        # bound_eta = find_bound_eta(decs, data_size, uav_gains, bs_gains, powers, num_samples) # TODO 

        # Solve eta
        eta = solve_optimal_eta(decs, data_size, uav_gains, bs_gains, powers, freqs, num_samples) 

        # Solve powers p, freqs f and apply heursitic method for choosing decisions x 
        uav_powers, uav_freqs = solve_powers_freqs(eta, num_samples, data_size, gains=uav_gains, ti_penalty=delta_t)
        bs_powers, bs_freqs = solve_powers_freqs(eta, num_samples, data_size, gains=bs_gains, ti_penalty=0)

        decs = np.ones(shape=num_users, dtype=int) # all UAV 
        uav_ene = calc_total_energy(eta, uav_freqs, decs, uav_powers, num_samples, data_size, uav_gains, bs_gains)
        decs = np.zeros(shape=num_users, dtype=int) # all BS
        bs_ene = calc_total_energy(eta, bs_freqs, decs, bs_powers, num_samples, data_size, uav_gains, bs_gains)

        difference = uav_ene - bs_ene
        idx = np.argpartition(difference, max_uav)[:max_uav] # https://stackoverflow.com/questions/34226400/find-the-index-of-the-k-smallest-values-of-a-numpy-array
        decs[idx] = 1

        powers = decs * uav_powers + (1 - decs) * bs_powers
        freqs = decs * uav_freqs + (1 - decs) * bs_freqs 

        # Check stop condition
        obj = calc_total_energy(eta, freqs, decs, powers, num_samples, data_size, uav_gains, bs_gains).sum()
        print(f"iter = {iter} obj = {obj}")

        if (abs(obj_prev - obj) < acc) or iter == iter_max: 
            print("Done!")
            break

        obj_prev = obj
        iter += 1 

    # Stop 
    return (eta, freqs, decs, powers)

def test_with_location():
    xs, ys, _ =  init_location()
    print("xs =", xs)
    print("ys =", ys)
    uav_gains = calc_uav_gains(xs, ys) 
    bs_gains = calc_bs_gains(xs, ys)
    print(f"uav_gains = {uav_gains}")
    print(f"bs_gains = {bs_gains}")

    num_samples = np.array([117, 110, 165, 202, 454, 112, 213, 234, 316, 110])
    freqs = np.array([1, 0.6, 2, 0.3, 0.4, 0.5, 1.5, 1.2, 0.3, 1]) * 1e9 # max = 2GHz
    num_rounds = 30
    cp_ene = calc_comp_energy(num_rounds=num_rounds, num_samples=num_samples, freqs=freqs)
    print(f"cp_ene =", cp_ene)
    cp_time = calc_comp_time(num_rounds=num_rounds, num_samples=num_samples, freqs=freqs)
    print(f"cp_time = {cp_time}")

    decs = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
    data_size = np.array([s_n for _ in range(num_users)])
    powers = np.array([0.1, 0.06, 0.1, 0.05, 0.07, 0.07, 0.1, 0.04, 0.04, 0.05])

    co_time = calc_trans_time(decs=decs, data_size=data_size, uav_gains=uav_gains, bs_gains=bs_gains, powers=powers)
    print(f"decs = {decs}")
    print(f"co_time = {co_time}")

    co_ene = calc_trans_energy(decs=decs, data_size=data_size, uav_gains=uav_gains, bs_gains=bs_gains, powers=powers)
    print(f"co_ene = {co_ene}")

    bound_eta = find_bound_eta(decs=decs, data_size=data_size, uav_gains=uav_gains, bs_gains=bs_gains, powers=powers, num_samples=num_samples)
    print(f"bound_eta = {bound_eta}")

    eta = solve_optimal_eta(decs=decs, data_size=data_size, uav_gains=uav_gains, bs_gains=bs_gains, powers=powers, freqs=freqs, num_samples=num_samples)
    print(f"eta = {eta}")

    opt_powers_uav, opt_freqs_uav = solve_powers_freqs(eta, num_samples, data_size, uav_gains, delta_t)
    print(f"opt_powers_uav = {opt_powers_uav}")
    print(f"opt_freqs_uav = {opt_freqs_uav}")

    opt_powers_bs, opt_freqs_bs = solve_powers_freqs(eta, num_samples, data_size, bs_gains, 0)
    print(f"opt_powers_bs = {opt_powers_bs}")
    print(f"opt_freqs_bs = {opt_freqs_bs}")

def test_optimize_network(): 
    xs, ys, _ =  init_location()
    print("xs =", xs)
    print("ys =", ys)
    uav_gains = calc_uav_gains(xs, ys) 
    bs_gains = calc_bs_gains(xs, ys)
    print(f"uav_gains = {uav_gains}")
    print(f"bs_gains = {bs_gains}")

    num_samples = np.array([117, 110, 165, 202, 454, 112, 213, 234, 316, 110])
    freqs = np.array([1, 0.6, 2, 0.3, 0.4, 0.5, 1.5, 1.2, 0.3, 1]) * 1e9 # max = 2GHz

    decs = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
    data_size = np.array([s_n for _ in range(num_users)])
    powers = np.array([0.1, 0.06, 0.1, 0.05, 0.07, 0.07, 0.1, 0.04, 0.04, 0.05])
    eta, freqs, decs, powers = optimize_network(num_samples, data_size, uav_gains, bs_gains)

    print(f"eta = {eta}")
    print(f"decs = {decs}")
    print(f"freqs = {freqs}")
    print(f"decs = {decs}")
    print(f"powers = {powers}") 

if __name__=='__main__': 
    # test_with_location()
    test_optimize_network()