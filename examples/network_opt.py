from scipy.special import lambertw
import numpy as np 

def find_bound_eta(): 
    r""" Solve Lambert W function to find the boundary of eta (eta_n_min, eta_n_max)
    Args: 
    Return: 
        eta_min, eta_max
    """
    z = np.array([1, 3])
    w = lambertw(z=z)
    print("w =", w)
    # print(w * np.exp(w))
    print("type(w[0]) = ", type(w[0]))
    print("real w =", w.real)

if __name__=="__main__": 
    find_bound_eta()