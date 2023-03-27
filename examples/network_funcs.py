from network_params import * 
import numpy as np 
from scipy.special import lambertw

from location_model import init_location
from location_model import update_location 
from optimization import NewtonOptim

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
    # print(f"calc_comp_energy num_rounds = {num_rounds}\tnum_samples = {num_samples}\tfreqs = {freqs}")
    # print(f"calc_comp_energy ene_cp = {ene_cp}")
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

def solve_initial_eta(af, bf, tau): 
    r""" TODO
    Args:
        af: 
        bf: 
        tau: 
    Return:
        Optimal eta 
    """
    print(f"solve_initial_eta tau = {tau}")

    tmp = - tau/af * np.exp((bf - tau)/af)
    w0 = lambertw(tmp, k=0) # (N, ) # lambert at 0 branch (-> eta min) complex number, imaginary part = 0 
    w_1 = lambertw(tmp, k=-1) # (N, ) # lambert at -1 branch (-> eta max) complex number, imaginary part = 0 
    # print(f"w0 = {w0}\nw_1 = {w_1}")

    xs_min = (bf - tau)/af - w0.real # (N, ) # all negative
    xs_max = (bf - tau)/af - w_1.real # (N, ) # all negative 
    # print(f"xs_min = {xs_min}\nxs_max = {xs_max}")

    x_min = np.max(xs_min) # 1 
    x_max = np.min(xs_max) # 1 
    # print(f"x_min = {x_min}\tx_max = {x_max}")

    eta_min = np.exp(x_min)
    eta_max = np.exp(x_max)
    print(f"eta_min = {eta_min}\neta_max = {eta_max}")

    return eta_min, eta_max

def find_bound_eta(num_samples, data_size, uav_gains, bs_gains, decs, freqs, powers, tau):
    r""" TODO 
    Args: 
        af: ln(eta) coefficient, computation related (time)
        bf: free coefficient, communication related 
    """ 
    af = a * calc_comp_time(1, num_samples, freqs) * v / math.log(2)
    bf = a * calc_trans_time(decs, data_size, uav_gains, bs_gains, powers)

    eta_min, eta_max = solve_initial_eta(af, bf, tau)
    return eta_min, eta_max

def dinkelbach_method(af, bf): 
    r""" TODO 
    Args: 
        af: ln(eta) coefficient, computation related (time, or energy)
        bf: free coefficient, communication related 
    """

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
    
def solve_optimal_eta(decs, data_size, uav_gains, bs_gains, powers, freqs, num_samples): 
    r""" Find the optimal local accuracy eta by Dinkelbach method
    Args: 
    Return:
        Optimal eta 
    """

    bf = a * calc_trans_energy(decs, data_size, uav_gains, bs_gains, powers).sum()
    af = v * a * calc_comp_energy(num_rounds=1, num_samples=num_samples, freqs=freqs).sum() / math.log(2)

    eta = dinkelbach_method(af, bf)
    print(f"eta = {eta}")

    ## LOGTRACE
    ene_opt = calc_total_energy(eta, freqs, decs, powers, num_samples, data_size, uav_gains, bs_gains)
    print(f"ene_opt = {ene_opt}")
    ## LOGTRACE 
    return eta

def initialize_feasible_solution(data_size, uav_gains, bs_gains, num_samples): 
    r"""
    Method: bisection
    Args:
    Return: 
        eta, T_min
    """
    ### Find af, bf efficient 
    # Solve optimal decision decs
    t_co_uav = calc_trans_time(1, data_size, uav_gains, bs_gains, power_max)
    t_co_bs = calc_trans_time(0, data_size, uav_gains, bs_gains, power_max)

    difference = t_co_uav + delta_t - t_co_bs
    idx = np.argpartition(difference, max_uav)[:max_uav]
    idx_uav = idx[np.where(difference[idx] < 0)]

    decs_opt = np.zeros(shape=num_users, dtype=int)
    decs_opt[idx_uav] = 1

    # TODO: check delta_t 
    # TRACELOG
    print(f"t_co_uav = {t_co_uav}\nt_co_bs = {t_co_bs}\ndifference = {difference}\ndecs_opt = {decs_opt}")
    # TRACELOG 

    # Solve optimal eta for current resource allocation (fixed f_max, p_max, decs)
    t_cp = calc_comp_time(1, num_samples, freq_max)
    t_co = calc_trans_time(decs_opt, data_size, uav_gains, bs_gains, power_max)

    af = a * t_cp * v / math.log(2)
    bf = a * t_co

    # Solve optimal eta by dinkelbach method with argument related to the total time 
    af_opt, bf_opt = af.sum(), bf.sum()
    eta_opt = dinkelbach_method(af_opt, bf_opt) 
    print(f"eta_opt = {eta_opt}")

    ### Iterative for tau 
    t_min, t_max = 40, 120 # seconds 
    acc = 1e-4
    iter = 0 

    while 1:
        tau = (t_min + t_max) / 2.0 
        eta_min, eta_max = solve_initial_eta(af, bf, tau)       
        print(f"iter = {iter}\teta_min = {eta_min}\teta_opt = {eta_opt}\teta_max = {eta_max}") 

        # Check feasible condition for current tau  
        if eta_opt > eta_min and eta_opt < eta_max: # feasible solution  
            t_max = tau 
        else:
            t_min = tau 
        print(f"iter = {iter}\tt_min = {t_min}\ttau = {tau}\tt_max = {t_max}")

        if (t_max - t_min)/t_max < acc: 
            break
        iter += 1
    
    print(f"eta = {eta_opt}, tau = {tau}")
    
    # LOGTRACE
    t_total = calc_total_time(eta_opt, freq_max, decs_opt, power_max, num_samples, data_size, uav_gains, bs_gains)
    print(f"t_total = {t_total}")
    ## LOGTRACE
    return eta_opt, tau 

def solve_freqs_powers(eta, num_samples, decs, data_size, uav_gains, bs_gains, tau): 
    r"""Solve powers, freqs for each users by newton's method"""

    gains = decs * uav_gains + (1 - decs) * bs_gains
    penalty_time = decs * delta_t # (N, )
    num_local_rounds = v * math.log2(1 / eta)

    # Calculate opt coefficient for optimizer 
    opt_as = data_size / bw * math.log(2) # (N, )
    opt_bs = gains / N_0 # (N, )
    opt_cs = num_local_rounds * C_n * num_samples # (N, )
    opt_tau = tau * (1-eta)/a - penalty_time # (N, ) penalty time for chosing uav # broadcasting   
    print(f"opt_as = {opt_as}\nopt_bs = {opt_bs}\nopt_cs = {opt_cs}\nopt_tau = {opt_tau}")

    # Normalization variables for faster convergence 
    norm_factor = 1e9

    # Initiate results 
    opt_powers = np.zeros(shape=num_users)
    opt_freqs = np.zeros(shape=num_users)

    for i in range(num_users): 
        print(f"NewtonOptim USER = {i}")
        opt = NewtonOptim(a=opt_as[i], b=opt_bs[i], c=opt_cs[i], tau=opt_tau[i], kappa=k_switch, norm_factor=norm_factor)
        inv_ln_power, inv_freq = opt.newton_method()

        # Update results 
        opt_freqs[i] = norm_factor * 1/inv_freq
        opt_powers[i]  = 1/opt_bs[i] * np.exp(1/inv_ln_power)

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

def calc_total_time(eta, freqs, decs, powers, num_samples, data_size, uav_gains, bs_gains): 
    num_local_rounds = v * math.log2(1/eta)
    num_global_rounds = a / (1 - eta)
    
    ti_comp = calc_comp_time(num_local_rounds, num_samples, freqs)
    ti_coms = calc_trans_time(decs, data_size, uav_gains, bs_gains, powers)
    print(f"ti_comp = {ti_comp}\nti_coms = {ti_coms}")
    ti = num_global_rounds * (ti_coms + ti_comp)
    return ti 

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
    tau = int(1.3 * t_min) # > t_min (= t_min + const) e.g t_min + t_min/10 TODO 
    print(f"optimize_network_tau = {tau}")

    while 1: 
        # Tighten the bound of eta 
        eta_min, eta_max = find_bound_eta(num_samples, data_size, uav_gains, bs_gains, decs, freqs, powers, tau)

        # Solve eta
        eta = solve_optimal_eta(decs, data_size, uav_gains, bs_gains, powers, freqs, num_samples) 

        # Solve powers p, freqs f and apply heursitic method for choosing decisions x 
        decs = np.ones(shape=num_users, dtype=int) # check with all connecting to uav 
        uav_freqs, uav_powers = solve_freqs_powers(eta, num_samples, decs, data_size, uav_gains, bs_gains, tau)
        uav_ene = calc_total_energy(eta, uav_freqs, decs, uav_powers, num_samples, data_size, uav_gains, bs_gains)
        print(f"uav_ene = {uav_ene}")

        decs = np.zeros(shape=num_users, dtype=int) # check with all connecting to bs 
        bs_freqs, bs_powers = solve_freqs_powers(eta, num_samples, decs, data_size, uav_gains, bs_gains, tau)
        bs_ene = calc_total_energy(eta, bs_freqs, decs, bs_powers, num_samples, data_size, uav_gains, bs_gains)
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
        t_total = calc_total_time(eta, freqs, decs, powers, num_samples, data_size, uav_gains, bs_gains)
        print(f"t_total = {t_total}")

        # Check stop condition
        obj = calc_total_energy(eta, freqs, decs, powers, num_samples, data_size, uav_gains, bs_gains).sum()
        print(f"optimize_network_iter = {iter} obj = {obj}")
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

def solve_freqs_fake(eta, num_samples, decs, data_size, uav_gains, bs_gains, tau):
    # power fixed at power_max, decs fixed 
    # calculate coms time 
    t_co = calc_trans_time(decs, data_size, uav_gains, bs_gains, power_max) # (N, )

    # calculate number of local, global rounds 
    num_local_rounds = v * math.log2(1/eta)
    num_global_rounds = a / (1-eta)

    # calculate computation time for one local_round 
    t_cp_1 = (tau / num_global_rounds - t_co) / num_local_rounds # (N, )

    # Calculate optimal freqs     
    freqs = C_n * num_samples / t_cp_1 # (N, )
    return freqs

def optimize_network_fake(num_samples, data_size, uav_gains, bs_gains):
    print(f"optimize_network_fake uav_gains = {uav_gains}\nbs_gains = {bs_gains}")
    # Initialize a feasible solution 
    freqs = np.ones(num_users) * freq_max
    powers = np.ones(num_users) * power_max
    decs = np.array([1, 0, 1, 1, 0, 0, 1, 1, 0, 0], dtype=int)

    eta = 0.01
    obj_prev = calc_total_energy(eta, freqs, decs, powers, num_samples, data_size, uav_gains, bs_gains).sum()
    print(f"obj_prev = {obj_prev}")

    # Repeat
    iter = 0
    eta, t_min = initialize_feasible_solution(data_size, uav_gains, bs_gains, num_samples) # eta = 0.317, t_min = 66.823
    tau = int(1.3 * t_min) # > t_min (= t_min + const) e.g t_min + t_min/10 TODO 
    print(f"optimize_network_fake eta = {eta}\ttau = {tau}\tt_min = {t_min}")
    # tau = 50

    while 1: 
        # find bound eta 
        eta_min, eta_max = find_bound_eta(num_samples, data_size, uav_gains, bs_gains, decs, freqs, powers, tau)
        print(f"eta_min = {eta_min}\teta_max = {eta_max}")
        # Solve eta
        eta = solve_optimal_eta(decs, data_size, uav_gains, bs_gains, powers, freqs, num_samples) 

        # Solve freqs f
        freqs = solve_freqs_fake(eta, num_samples, decs, data_size, uav_gains, bs_gains, tau)

        # LOG TRACE
        t_total = calc_total_time(eta, freqs, decs, powers, num_samples, data_size, uav_gains, bs_gains)
        print(f"t_total = {t_total}")

        # Check stop condition
        obj = calc_total_energy(eta, freqs, decs, powers, num_samples, data_size, uav_gains, bs_gains).sum()
        print(f"optimize_network_iter = {iter} obj = {obj}")
        print(f"eta = {eta}")
        print(f"freqs = {freqs}")

        if (abs(obj_prev - obj) < acc) or iter == iter_max: 
            print("Done!")
            break

        obj_prev = obj
        iter += 1 
    
    num_local_rounds = v * math.log2(1/eta)
    num_global_rounds = a / (1 - eta)
    print(f"optimize_network_fake num_local_rounds = {num_local_rounds}\tnum_global_rounds = {num_global_rounds}")
    return num_local_rounds, num_global_rounds

def test_with_location():
    xs, ys, _ =  init_location()
    print("xs =", xs)
    print("ys =", ys)
    uav_gains = calc_uav_gains(xs, ys) 
    bs_gains = calc_bs_gains(xs, ys)
    print(f"uav_gains = {uav_gains}")
    print(f"bs_gains = {bs_gains}")

    bs_gains_n0 = bs_gains / N_0
    uav_gains_n0 = uav_gains / N_0 
    print(f"bs_gains_n0 = {bs_gains_n0}\nuav_gains_n0 = {uav_gains_n0}")

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

    # co_time = calc_trans_time(decs=decs, data_size=data_size, uav_gains=uav_gains, bs_gains=bs_gains, powers=powers)
    # print(f"decs = {decs}")
    # print(f"co_time = {co_time}")

    # co_ene = calc_trans_energy(decs=decs, data_size=data_size, uav_gains=uav_gains, bs_gains=bs_gains, powers=powers)
    # print(f"co_ene = {co_ene}")

    eta = 0.15
    freqs, powers = solve_freqs_powers(eta, num_samples, decs, data_size, uav_gains, bs_gains, tau=100)

def test_optimize_network(): 
    xs, ys, _ =  init_location()
    print("xs =", xs)
    print("ys =", ys)
    uav_gains = calc_uav_gains(xs, ys) 
    bs_gains = calc_bs_gains(xs, ys)
    print(f"uav_gains = {uav_gains}")
    print(f"bs_gains = {bs_gains}")

    num_samples = np.array([117, 110, 165, 202, 454, 112, 213, 234, 316, 110])
    data_size = np.array([s_n for _ in range(num_users)])
    print(f"data_size = {data_size}")
    eta, freqs, decs, powers = optimize_network(num_samples, data_size, uav_gains, bs_gains)

    print(f"eta = {eta}")
    print(f"decs = {decs}")
    print(f"freqs = {freqs}")
    print(f"powers = {powers}") 
    
    ene = calc_total_energy(eta, freqs, decs, powers, num_samples, data_size, uav_gains, bs_gains)
    ti = calc_total_time(eta, freqs, decs, powers, num_samples, data_size, uav_gains, bs_gains)
    print(f"calc_total_energy ene = {ene}\ncalc_total_time ti = {ti}")

def test_optimize_network_fake(): 
    xs, ys, dirs =  init_location()
    print("xs =", xs)
    print("ys =", ys)
    uav_gains = calc_uav_gains(xs, ys) 
    bs_gains = calc_bs_gains(xs, ys)
    print(f"uav_gains = {uav_gains}")
    print(f"bs_gains = {bs_gains}")

    num_samples = np.array([117, 110, 165, 202, 454, 112, 213, 234, 316, 110])
    data_size = np.array([s_n for _ in range(num_users)])
    print(f"data_size = {data_size}")
    num_local_rounds, num_global_rounds = optimize_network_fake(num_samples, data_size, uav_gains, bs_gains)

    xs_new, ys_new, dirs_new = update_location(xs, ys, dirs)
    print("xs_new =", xs_new)
    print("ys_new =", ys_new)
    print("update_location")
    uav_gains = calc_uav_gains(xs_new, ys_new) 
    bs_gains = calc_bs_gains(xs_new, ys_new)
    print(f"uav_gains = {uav_gains}")
    print(f"bs_gains = {bs_gains}")
    num_local_rounds, num_global_rounds = optimize_network_fake(num_samples, data_size, uav_gains, bs_gains)

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
    # test_optimize_network()
    # test_feasible_solution()
    test_optimize_network_fake()