import numpy as np 
np.set_printoptions(precision=6, linewidth=np.inf)

from scipy.optimize import Bounds 
from scipy.optimize import LinearConstraint
from scipy.optimize import minimize 

class NewtonOptim:
    def __init__(self, a=1, b=1, c=2, tau=1, kappa=1, norm_factor=1, z_min=1, t_min=1):
        self.a, self.b = a, b
        self.c = c / norm_factor
        self.kappa = kappa * (norm_factor**3)
        self.tau = tau 
        self.A = np.array([self.a, self.c]) #(2, 1)
        self.z_min = z_min # - z + z_min <= 0 
        self.t_min = t_min * norm_factor # - t + t_min <= 0 

        # for testing system capacity 
        taumin = a * z_min + c * t_min
        print(f"a = {self.a}\tb = {self.b}\tc = {self.c}\tkappa = {self.kappa}")
        print(f"tau = {self.tau}\ttaumin = {taumin}\tz_min = {self.z_min}\tt_min = {self.t_min}")
         
    def objective(self, x): 
        z, t = x[0], x[1]
        f0 = self.a/self.b * z * (np.exp(1/z) - 1) + self.kappa * self.c / (t**2) # (1)
        # print(f"objective x = {x}\tf0 = {f0}")
        return f0 
    
    def grad_obj(self, x): 
        z, t = x[0], x[1]
        grad = np.array([self.a/self.b * ((1 - 1/z) * np.exp(1/z) - 1), -2 * self.kappa * self.c / (t**3)])
        # grad = grad[:, np.newaxis] # (2, 1)
        # print(f"grad_obj x = {x}\tgrad = {grad}")
        return grad 

    def hessian(self, x): 
        z, t = x[0], x[1]
        hess = np.array([[self.a/self.b * np.exp(1/z) / (z**3), 0],
                        [0, 6 * self.kappa * self.c / (t**4)]]) # (2, 2)
        # print(f"hessian = {hess}")
        return hess 

    def newton_method(self):
        r"""Primal-dual interior-point method"""
        bounds = Bounds([self.z_min, self.t_min], [np.inf, np.inf])
        linear_constraint = LinearConstraint([self.a, self.c], self.tau, self.tau)

        x0 = np.array([self.z_min, self.t_min]) + np.random.rand(2)
        res = minimize(self.objective, x0, method='trust-constr', jac=self.grad_obj, hess=self.hessian,
               constraints=linear_constraint, bounds=bounds)
        x = res.x 
        return x[0], x[1]

def test(a, b, c, kappa, tau, norm_factor, z_min, t_min):
    ## Newton optimization 
    opt = NewtonOptim(a, b, c, tau, kappa, norm_factor, z_min, t_min)
    inv_ln_power, inv_freq = opt.newton_method()

    ## Results 
    print(f"inv_power = {inv_ln_power}\tinv_freq = {inv_freq}")
    x_opt = np.array([inv_ln_power, inv_freq])
    print(f"x_opt = {x_opt} obj = {opt.objective(x_opt)}")

    # Original problem solutions 
    power = 1/b * np.exp(1/inv_ln_power)
    freq = norm_factor * 1/inv_freq
    print("power = {:3f}\tfreq = {:.3e}".format(power, freq))
    return 

if __name__=='__main__':
    ## Normalized parameters 

    # opt_as = self.data_size / bw * math.log(2) # (N, )
    # opt_bs = gains / N_0 # (N, )
    # opt_cs = num_lrounds * C_n * self.num_samples # (N, )
    # opt_tau = tau * (1-eta)/self.an - penalty_time # (N, )  

    print("Normalized parameters") 
    norm_factor = 1e9 

    # a, b, c = 0.0348, 120, 1*1e8 
    # a = 0.03482371435133165
    # b = 16.717536192138855
    # c = 0.03362095304577377 * norm_factor
    # power = 0.066290        freq = 1.279e+08 obj = 0.002248790683097171

    a = 0.03482371435133165	
    b = 9863.430506596913	
    c = 0.05116275624731778 * norm_factor
    tau = 0.8098728122557564
    # taumin = 0.030631940230047387	z_min = 0.14503226322827228	t_min = 0.5
    # iter = 14 converged! (z, t) = (-0.6456984132273457, 16.26883480107134)
    kappa = 1e-28 

    import math 
    f_max = 2*1e9 
    z_min = 1/math.log(1 + 0.1 * b) 
    t_min = 1/f_max
    

    test(a, b, c, kappa, tau, norm_factor, z_min, t_min)

    ## Non-normalized parameters
    # print("Non-normalized parameters") 
    # norm_factor = 1 
    # test(a, b, c, kappa, tau, norm_factor)