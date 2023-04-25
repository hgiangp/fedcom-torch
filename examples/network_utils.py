import numpy as np 
import math 
from scipy.special import lambertw 

def solve_bound_eta(af, bf, tau): 
        r""" TODO
        Args:
            af: 
            bf: 
            tau: 
        Return:
            Optimal eta 
        """
        print(f"solve_bound_eta tau = {tau}\naf = {af}\nbf = {bf}")

        tmp = - tau/af * np.exp((bf - tau)/af)
        print(f"tmp = {tmp}")
        w0 = lambertw(tmp, k=0) # (N, ) # lambert at 0 branch (-> eta min) complex number, imaginary part = 0 
        w_1 = lambertw(tmp, k=-1) # (N, ) # lambert at -1 branch (-> eta max) complex number, imaginary part = 0 
        print(f"w0 = {w0}\nw_1 = {w_1}")

        xs_min = (bf - tau)/af - w0.real # (N, ) # all negative
        xs_max = (bf - tau)/af - w_1.real # (N, ) # all negative 
        print(f"xs_min = {xs_min}\nxs_max = {xs_max}")

        x_min = np.max(xs_min) # 1 
        x_max = np.min(xs_max) # 1 
        print(f"solve_bound_eta x_min = {x_min}\tx_max = {x_max}")

        eta_min = np.exp(x_min)
        eta_max = np.exp(x_max)
        print(f"solve_bound_eta eta_min = {eta_min}\teta_max = {eta_max}")

        return eta_min, eta_max

def dinkelbach_method(af, bf): 
    r""" TODO 
    Args: 
        af: ln(eta) coefficient, computation related (time, or energy)
        bf: free coefficient, communication related 
    """
    acc = 1e-4 # absolute tolerant 

    eta = 0.01
    zeta = af * 1.01
    h_prev = af * math.log(1/eta) + bf - zeta * (1 - eta)

    while 1:
        eta = af / zeta # calculate temporary optimal eta # too large eta make the cost trap         
        print(f"af = {af}\tbf = {bf}\tzeta = {zeta}\teta = {eta}")
        h_curr = af * math.log(1/eta) + bf - zeta * (1 - eta) # check stop condition
        if abs(h_curr - h_prev) < acc: 
            break
        h_prev = h_curr   
        
        zeta = ((af * math.log(1/eta)) + bf) / (1 - eta) # update zeta
    
    # print(f"eta = {eta}")
    return eta
