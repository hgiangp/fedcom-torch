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
    gains = decs * uav_gains + (1 - decs) * bs_gains # (N, )
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

def find_bound_eta(decs, data_size, uav_gains, bs_gains, powers, num_samples, tau): 
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
    # eta_min, eta_max = 0.0, 1.0
    
    # Find eta_min 
    # f = f_max, t_co min at p = powers 
    af = a * calc_trans_time(decs, data_size, uav_gains, bs_gains, powers) # (N, )
    bf = a * calc_comp_time(num_rounds=1, num_samples=num_samples, freqs=freq_max) * v / math.log(2) # (N, )
    # print(f"af = {af}")
    # print(f"bf = {bf}")
    # print(f"a = {a}, v = {v}, tau = {tau}")
    x = (af - tau)/bf - lambertw(- tau/bf * np.exp((af - tau)/bf)) # (N, )
    # print(f"x = {x}")
    lower_bound_eta = np.exp(x.real) # (N, )
    # print("lower_bound_eta =", lower_bound_eta)
    eta_min = np.amax(lower_bound_eta)

    # Find eta_max
    # f = 0 t_co max = tau / num_global_rounds  
    upper_bound_eta = 1 - af/tau
    # print(f"upper_bound_eta = {upper_bound_eta}")
    eta_max = np.amin(upper_bound_eta) 
    print(f"eta_min = {eta_min}\teta_max = {eta_max}")
    return eta_min, eta_max

def solve_optimal_eta(decs, data_size, uav_gains, bs_gains, powers, freqs, num_samples): 
    r""" Find the optimal local accuracy eta by Dinkelbach method
    Args: 
    Return:
        Optimal eta 
    """
    acc = 1e-4

    bf = a * calc_trans_energy(decs, data_size, uav_gains, bs_gains, powers).sum()
    af = v * a * calc_comp_energy(num_rounds=1, num_samples=num_samples, freqs=freqs).sum() / math.log(2)
    
    eta = 1.0
    zeta = af * 2 
    h_prev = af * math.log(1/eta) + bf - zeta * (1 - eta)

    while 1:
        eta = af / zeta # calculate temporary optimal eta
        print(f"af = {af}\tbf = {bf}\tzeta = {zeta}\teta = {eta}")
        h_curr = af * math.log(1/eta) + bf - zeta * (1 - eta) # check stop condition
        if abs(h_curr - h_prev) < acc: 
            break
        h_prev = h_curr   
        
        zeta = ((af * math.log(1/eta)) + bf) / (1 - eta) # update zeta
    
    print(f"eta = {eta}")
    return eta

def initialize_feasible_solution(data_size, uav_gains, bs_gains, num_samples): 
    r"""
    Method: bisection
    Args:
    Return: 
        eta, T_min
    """
    powers = np.ones(shape=num_users) * power_max # (N, )
    
    decs = np.ones(shape=num_users, dtype=int)
    tco_uav = calc_trans_time(decs, data_size, uav_gains, bs_gains, powers)
    decs = np.zeros(shape=num_users, dtype=int)
    tco_bs = calc_trans_time(decs, data_size, uav_gains, bs_gains, powers)
    
    difference = tco_uav - tco_bs 
    idx = np.argpartition(difference, max_uav)[:max_uav]
    idx_uav = idx[np.where(difference[idx] < 0)]
    decs[idx_uav] = 1 # (N, )

    freqs = np.ones(shape=num_users) * freq_max # (N, )

    t_min, t_max = 60, 200 # seconds 
    tau = t_min 
    acc = 1e-4

    iter = 0 
    eta = solve_optimal_eta(decs, data_size, uav_gains, bs_gains, powers, freqs, num_samples)
    while 1:
        tau = (t_min + t_max) / 2.0 
        eta_min, eta_max = find_bound_eta(decs, data_size, uav_gains, bs_gains, powers, num_samples, tau)
        
        # Check feasible condition for current tau  
        if eta > eta_min and eta < eta_max: # feasible solution  
            t_max = tau 
        else: 
            t_min = tau        

        print(f"iter = {iter}\ttau = {tau}\tt_max = {t_max}\tt_min = {t_min}") 
        if (t_max - t_min)/t_max < acc: 
            break
        iter += 1
    
    print(f"eta = {eta}, tau = {tau}")
    
    return eta, tau 

def solve_freqs_powers(eta, num_samples, decs, data_size, uav_gains, bs_gains, powers, tau): 
    
    # Update optimal freqs 
    t_co = calc_trans_time(decs, data_size, uav_gains, bs_gains, powers)    
    num_local_rounds = v * math.log2(1 / eta)
    
    t_cp = (tau * (1 - eta)/a - t_co) / num_local_rounds
    opt_freqs = C_n * num_samples / t_cp

    # Update optimal powers
    x_convex = lambertw(-1/(2 * math.exp(1))).real + 1 

    t_trans = t_co - decs * delta_t
    x_opt = np.maximum(data_size/bw/t_trans, x_convex)
    print(f"decs = {decs}\tx_convex = {x_convex}\nx_opt = {x_opt}")
    
    gains = decs * uav_gains + (1 - decs) * bs_gains # (N, )
    opt_powers = N_0/gains * (2 * np.exp(x_opt) - 1)

    print(f"opt_freqs = {opt_freqs}\nopt_powers = {opt_powers}")
    return opt_freqs, opt_powers

def calc_total_energy(eta, freqs, decs, powers, num_samples, data_size, uav_gains, bs_gains): 
    num_local_rounds = v * math.log2(1 / eta)
    num_global_rounds = a / (1 - eta)
    print(f"eta = {eta}\tnum_local_rounds = {num_local_rounds}\tnum_global_rounds = {num_global_rounds}")

    ene_coms = calc_trans_energy(decs, data_size, uav_gains, bs_gains, powers)
    ene_comp = calc_comp_energy(num_local_rounds, num_samples, freqs)
    print(f"ene_coms = {ene_coms}\nene_comp = {ene_comp}")

    energy = num_global_rounds * (ene_coms + ene_comp)
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
    eta, t_min = initialize_feasible_solution(data_size, uav_gains, bs_gains, num_samples) # eta = 0.317, t_min = 66.823
    tau = 80 # > t_min (= t_min + const) e.g t_min + t_min/10 TODO 

    while 1: 
        # Tighten the bound of eta 
        eta_min, eta_max = find_bound_eta(decs, data_size, uav_gains, bs_gains, powers, num_samples, tau) # (eta_min, eta_max) # TODO: tau

        # Solve eta
        eta = solve_optimal_eta(decs, data_size, uav_gains, bs_gains, powers, freqs, num_samples) 

        # Solve powers p, freqs f and apply heursitic method for choosing decisions x 
        uav_freqs, uav_powers = solve_freqs_powers(eta, num_samples, 1, data_size, uav_gains, bs_gains, powers, tau)
        bs_freqs, bs_powers = solve_freqs_powers(eta, num_samples, 0, data_size, uav_gains, bs_gains, powers, tau)
        
        uav_ene = calc_total_energy(eta, uav_freqs, 1, uav_powers, num_samples, data_size, uav_gains, bs_gains)
        print(f"uav_ene = {uav_ene}")
        bs_ene = calc_total_energy(eta, bs_freqs, 0, bs_powers, num_samples, data_size, uav_gains, bs_gains)
        print(f"bs_ene = {bs_ene}")
        difference = uav_ene - bs_ene
        
        print(f"difference = {difference}")
        idx = np.argpartition(difference, max_uav)[:max_uav] # https://stackoverflow.com/questions/34226400/find-the-index-of-the-k-smallest-values-of-a-numpy-array
        idx_uav = idx[np.where(difference[idx] < 0)]
        
        decs = np.zeros(num_users, dtype=int)
        decs[idx_uav] = 1 # (N, )

        powers = decs * uav_powers + (1 - decs) * bs_powers
        freqs = decs * uav_freqs + (1 - decs) * bs_freqs

        # LOG TRACE
        num_local_rounds = v * math.log2(1 / eta)
        t_cp = calc_comp_time(num_local_rounds, num_samples, freqs)
        t_co = calc_trans_time(decs, data_size, uav_gains, bs_gains, powers)
        print(f"t_coms = {t_co}\nt_comp = {t_cp}")

        # Check stop condition
        obj = calc_total_energy(eta, freqs, decs, powers, num_samples, data_size, uav_gains, bs_gains).sum()
        print(f"iter = {iter} obj = {obj}")
        print(f"eta = {eta}")
        print(f"decs = {decs}")
        print(f"freqs = {freqs}")
        print(f"powers = {powers}") 

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
    powers = np.array([0.08, 0.1, 0.08, 0.09, 0.1, 0.07, 0.1, 0.09, 0.04, 0.08])

    co_time = calc_trans_time(decs=decs, data_size=data_size, uav_gains=uav_gains, bs_gains=bs_gains, powers=powers)
    print(f"decs = {decs}")
    print(f"co_time = {co_time}")

    co_ene = calc_trans_energy(decs=decs, data_size=data_size, uav_gains=uav_gains, bs_gains=bs_gains, powers=powers)
    print(f"co_ene = {co_ene}")

    # bound_eta = find_bound_eta(decs=decs, data_size=data_size, uav_gains=uav_gains, bs_gains=bs_gains, powers=powers, num_samples=num_samples)
    # print(f"bound_eta = {bound_eta}")

    eta = solve_optimal_eta(decs=decs, data_size=data_size, uav_gains=uav_gains, bs_gains=bs_gains, powers=powers, freqs=freqs, num_samples=num_samples)
    print(f"eta = {eta}")

    freqs, powers = solve_freqs_powers(eta, num_samples, decs, data_size, uav_gains, bs_gains, powers, tau=90)
    print(f"freqs = {freqs}")
    print(f"powers = {powers}")

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
    data_size = np.array([s_n for _ in range(num_users)])
    powers = np.array([0.1, 0.06, 0.1, 0.05, 0.07, 0.07, 0.1, 0.04, 0.04, 0.05])
    eta, freqs, decs, powers = optimize_network(num_samples, data_size, uav_gains, bs_gains)

    print(f"eta = {eta}")
    print(f"decs = {decs}")
    print(f"freqs = {freqs}")
    print(f"powers = {powers}") 

def test_feasible_solution():
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

    data_size = np.array([s_n for _ in range(num_users)])
    
    eta, t_min = initialize_feasible_solution(data_size, uav_gains, bs_gains, num_samples)

    print(f"eta = {eta}")
    print(f"t_min = {t_min}")


if __name__=='__main__': 
    # test_with_location()
    test_optimize_network()
    # test_feasible_solution()