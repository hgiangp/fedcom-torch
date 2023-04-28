import numpy as np 
np.set_printoptions(precision=6, linewidth=np.inf)

class NewtonOptim:
    def __init__(self, a=1, b=1, c=2, tau=1, kappa=1, norm_factor=1, z_min=1, t_min=1):
        self.a, self.b = a, b
        self.c = c / norm_factor
        self.kappa = kappa * (norm_factor**3)
        self.tau = tau 
        self.A = np.array([self.a, self.c]) #(2, 1)
        self.z_min = z_min # - z + z_min <= 0 
        self.t_min = t_min * norm_factor # - t + t_min <= 0 
        print(f"a = {self.a}\tb = {self.b}\tc = {self.c}\tkappa = {self.kappa}\ttau = {self.tau}\tself.t_min = {self.t_min}\tself.z_min={self.z_min}")
         
    def objective(self, x): 
        z, t = x[0], x[1]
        f0 = self.a/self.b * z * (np.exp(1/z) - 1) + self.kappa * self.c / (t**2) # (1)
        # print(f"objective x = {x}\tf0 = {f0}")
        return f0 
    
    def grad_obj(self, x): 
        z, t = x[0], x[1]
        grad = np.array([self.a/self.b * ((1 - 1/z) * np.exp(1/z) - 1), -2 * self.kappa * self.c / (t**3)])
        grad = grad[:, np.newaxis] # (2, 1)
        # print(f"grad_obj x = {x}\tgrad = {grad}")
        return grad 

    def eq_const(self, x): 
        hi = np.matmul(self.A, x) - self.tau # (1)
        # print(f"eq_const = {hi}")
        return hi
    
    def ineq_const(self, x): 
        r"""Return inequality constraints (2, 1)"""
        z, t = x[0], x[1]
        fi = np.array([-z + self.z_min, -t + self.t_min])
        fi = fi[:, np.newaxis] # (2, 1)
        # print(f"ineq_const = {fi}")
        return fi 
    
    def grad_ineq_const(self, x): 
        dfi = np.array([[-1, 0], [0, -1]]) # (2, 2)
        # print(f"grad_ineq_const = {dfi}")
        return dfi

    def hessian(self, x): 
        z, t = x[0], x[1]
        hess = np.array([[self.a/self.b * np.exp(1/z) / (z**3), 0],
                        [0, 6 * self.kappa * self.c / (t**4)]]) # (2, 2)
        # print(f"hessian = {hess}")
        return hess 

    def hessian_dual(self, x, lambd, v): 
        r"""x (2, ), lambda (2, ), v (1) """
        AT = self.A[:, np.newaxis]
        hess_dual = np.concatenate((self.hessian(x), self.grad_ineq_const(x), AT), axis=1) # (2, 5)
        hess_cent = np.concatenate((-np.matmul(np.diag(lambd), self.grad_ineq_const(x)), 
                    -np.diag(self.ineq_const(x).reshape(-1)), np.array([[0], [0]])), axis=1) # (2, 5)
        hess_pri = np.concatenate((self.A, np.array([0, 0]), np.array([0])), axis=0)[np.newaxis, :] # (1, 5)

        hess = np.concatenate((hess_dual, hess_cent, hess_pri), axis=0) # (5, 5)
        print(f"hessian_dual = \n{hess}")
        return hess
    
    def r_dual_func(self, x, lambd, v): 
        grad_obj = self.grad_obj(x)
        grad_ineq = np.matmul(self.grad_ineq_const(x), lambd.T)[:, np.newaxis]
        grad_eq = (self.A.T * v)[:, np.newaxis] # (2, 1)
        
        r_dual = grad_obj + grad_ineq + grad_eq 
        # print(f"r_dual = {r_dual}")
        return r_dual

    def residual_t(self, x, lambd, v, t_dual): 
        r"""Return optimality dual matrix r = (r_dual, r_cent, r_pri)
        Args: 
            x (2, 1), lambd: (2, 1), v: 1 
        """
        r_dual = self.r_dual_func(x, lambd, v) # (2, 1)
        r_cent = - np.matmul(np.diag(lambd), self.ineq_const(x)) - (1/t_dual) # broadcasting (2, 1)
        r_pri = np.array([self.eq_const(x)])[:, np.newaxis] # (1, )
        r = np.concatenate((r_dual, r_cent, r_pri), axis=0) # (5, 1)
        print(f"residual_t = \n{r}")
        return r 

    def backtracking_line_search(self, x, lambd, v, dir_x, dir_lambd, dir_v, t_dual, alpha=0.01, beta=0.8):
        # init the largest positive step length, not exceeding 1, give lambd = lambd + s * dir_lambd >=0 
        # s_max = sup{s \in [0, 1]| lambd + s * dir_lambd >= 0}
        #       = min{1, min{-lambd_i/dir_lambdi_i} | dir_lambdi_i < 0}
        div = -lambd/dir_lambd
        step_size = min(1, min(div[div > 0], default=1))

        print(f"backtracking_line_search x = {x}\tdir_x = {dir_x}")
        # current residual norm
        r_dual = np.linalg.norm(self.residual_t(x, lambd, v, t_dual)) # 2-norm default 
        while 1: 
            # new variable with updated direction 
            x_new = x + step_size * dir_x
            lambd_new = lambd + step_size * dir_lambd
            v_new = v + step_size * dir_v

            # new residual norm value  
            r_dual_new = np.linalg.norm(self.residual_t(x_new, lambd_new, v_new, t_dual)) # 2-norm default 
            print(f"r_dual_new = {r_dual_new} r_dual = {r_dual}")
            if r_dual_new <= (1 - alpha * step_size) * r_dual:
                print("backtracking_line_search terminate!")
                break 
            step_size = step_size * beta 
            
        print(f"backtracking_line_search step_size = {step_size}")
        return step_size

    def newton_method(self):
        r"""Primal-dual interior-point method"""
        
        max_iter = 100

        # Initialization
        m = 2 # number of inequality constraints 
        mu = 20
        eps_feas = 1e-5
        eps = 1e-5

        x = np.array([1, 1])
        lambd = np.array([1, 1])
        v = 1         

        dual_gap = - np.matmul(self.ineq_const(x).T, lambd)[-1] # (1)        
        for iter in range(max_iter): 
            # 1. Determize t
            t = mu * m / dual_gap

            # 2. compute primal newton step dir_x_nt, dual newton step dir_v_nt 
            inv_hess_x = np.linalg.inv(self.hessian_dual(x, lambd, v)) # (5, 5)
            dir_xlv = - np.matmul(inv_hess_x, self.residual_t(x, lambd, v, t)) 
            dir_x, dir_lambd, dir_v = dir_xlv[:2].reshape(-1), dir_xlv[2:4].reshape(-1), dir_xlv[-1]

            # 3. Line search and update 
            step_size = self.backtracking_line_search(x, lambd, v, dir_x, dir_lambd, dir_v, t)

            # update primal, dual variable 
            x = x + step_size * dir_x
            lambd = lambd + step_size * dir_lambd
            v = v + step_size * dir_v 
            print(f"iter = {iter} x = {x} dir_x = {dir_x}")
            print(f"iter = {iter} lambd = {lambd} dir_lambd = {dir_lambd}")
            print(f"iter = {iter} v = {v} dir_v = {dir_v}")

            # check stopping condition
            norm_rpri = np.linalg.norm(self.eq_const(x))
            norm_rdual = np.linalg.norm(self.r_dual_func(x, lambd, v))
            dual_gap = - np.matmul(self.ineq_const(x).T, lambd)[-1]

            print(f"norm_rpri = {norm_rpri}\tnorm_residual = {norm_rdual}\tdual_gap = {dual_gap}")

            if norm_rpri <= eps_feas and norm_rdual <= eps_feas and dual_gap <= eps: 
                print(f"iter = {iter} converged! ({x[0]}, {x[1]})")
                break 
        
        return x[0], x[1]

def test(a, b, c, kappa, tau, norm_factor, z_min, t_min):
    ## Newton optimization 
    opt = NewtonOptim(a, b, c, tau, kappa, norm_factor, z_min, t_min)
    inv_ln_power, inv_freq = opt.newton_method()

    ## Results 
    print(f"inv_power = {inv_ln_power}\tinv_freq = {inv_freq}")
    x_opt = np.array([inv_ln_power, inv_freq])
    print(f"x_opt = {x_opt} obj = {opt.objective(x_opt)} Ax - b = {opt.eq_const(x_opt)}")
    tmp = [0.1, 0.1] # broadcasting 
    print(f"x_opt+tmp = {x_opt+tmp} obj = {opt.objective(x_opt+tmp)} Ax - b = {opt.eq_const(x_opt+tmp)}")
    print(f"x_opt-tmp = {x_opt-tmp} obj = {opt.objective(x_opt-tmp)} Ax - b = {opt.eq_const(x_opt-tmp)}")

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

    a, b, c = 0.0348, 120, 1*1e8 # power = 0.005364        freq = 1.619e+08
    kappa = 1e-28 

    import math 
    f_max = 2*1e9 
    z_min = 1/math.log(1 + 0.1 * b) 
    t_min = 1/f_max
    
    tmin = a / z_min + c / f_max # 0.13926023763966147 
    # tau = 0.14 #              power = 0.077522 freq = 8.039e+08 obj = 0.0075417986204784045
    # tau = 0.13926023763966147 power = 0.078307 freq = 8.082e+08 obj = 0.007619271633460264
    # tau = 0.15 #              power = 0.068399 freq = 7.492e+08 obj = 0.006606562686656652
    # tau = 0.2  #              power = 0.042411 freq = 5.599e+08 obj = 0.0038633707860031784
    tau = 0.14
    print("Normalized parameters") 
    norm_factor = 1e9 
    test(a, b, c, kappa, tau, norm_factor, z_min, t_min)

    ## Non-normalized parameters
    # print("Non-normalized parameters") 
    # norm_factor = 1 
    # test(a, b, c, kappa, tau, norm_factor)
