import numpy as np 
class NewtonOptim(object):
    def __init__(self, a=1, b=1, c=2, tau=1, kappa=1):
        self.a, self.b, self.c = a, b, c
        self.kappa = kappa 
        self.tau = tau 
        self.A = np.array([a, c]) 
         
    def objective(self, x): 
        z, t = x[0], x[1]
        f0 = self.a/self.b * z * (np.exp(1/z) - 1) + self.kappa * self.c / (t**2)
        print(f"objective x = {x}\tf0 = {f0}")

        return f0 

    def eq_constraints(self, x): 
        fi = np.dot(self.A, x) - self.tau # broadcasting 
        print(f"eq_constraints = {fi}")

        return fi 

    def hessian_dual(self, x): 
        z, t = x[0], x[1]
        hess = np.array([[self.a/self.b * np.exp(1/z) / (z**3), 0, self.a],
                        [0, 6 * self.kappa * self.c / (t**4), self.c], 
                        [self.a, self.c, 0]])

        print(f"hessian_dual x = {x}\nhess = {hess}")
        return hess 

    def hessian(self, x): 
        z, t = x[0], x[1]
        hess = np.array([[self.a/self.b * np.exp(1/z) / (z**3), 0],
                        [0, 6 * self.kappa * self.c / (t**4)]])
        
        print(f"hessian x = {x}\nhess = {hess}")
        return hess 

    def gradient(self, x): 
        z, t = x[0], x[1]
        grad = np.array([self.a/self.b * ((1 - 1/z) * np.exp(1/z) - 1), -2 * self.kappa * self.c / (t**3)])

        # print(f"gradient x = {x}\tgrad = {grad}")
        return grad 

    def dual_optimality(self, x, v): 
        r"""Return optimality dual matrix r = (r_dual, r_pri)"""
        
        r_dual = self.gradient(x).T + self.A.T * v # (N, )
        r_pri = np.array([self.eq_constraints(x)]) # (1, )
        r = np.concatenate((r_dual, r_pri), axis=0)

        print(f"r = {r}")
        return r 

    def backtracking_line_search(self, x, v, dir_x, dir_v, alpha=0.01, beta=0.8):
        step_size = 1

        print(f"backtracking_line_search x = {x}\tdir_x = {dir_x}")
        r_dual = np.linalg.norm(self.dual_optimality(x, v)) # 2-norm default 
        while 1: 
            r_dual_new = np.linalg.norm(self.dual_optimality(x + step_size * dir_x, v + step_size * dir_v)) # 2-norm default 
            print(f"r_dual_new = {r_dual_new} r_dual = {r_dual}")
            if r_dual_new <= (1 - alpha * step_size) * r_dual:
                print("backtracking_line_search terminate!")
                break 
            step_size = step_size * beta 
            
        print(f"backtracking_line_search step_size = {step_size}")
        return step_size

    def newton_method(self):
        r"""Infeasible starting point newton method"""
        
        max_iter = 50 
        dim = 2

        # initiate dual starting point
        x, v = np.array([1, 1]), 1 
        acc = 1e-5

        for iter in range(max_iter): 
            # compute primal newton step dir_x_nt, dual newton step dir_v_nt 
            inv_hess_x = np.linalg.inv(self.hessian_dual(x)) # (N+1, N+1)
            dir_xv = - np.dot(inv_hess_x, self.dual_optimality(x, v)) 
            dir_x, dir_v = dir_xv[:dim], dir_xv[dim:]

            # backtracking line search on r = (r_dual, r_pri)
            step_size = self.backtracking_line_search(x, v, dir_x, dir_v)

            # update primal, dual variable 
            x = x + step_size * dir_x
            v = v + step_size * dir_v 
            print(f"iter = {iter} x = {x} dir_x = {dir_x}")
            print(f"iter = {iter} v = {v} dir_v = {dir_v}")

            # check stopping condition
            equality_satisfied = np.allclose(self.eq_constraints(x), 0, atol=1e-5) 
            norm_residual = np.linalg.norm(self.dual_optimality(x, v))

            print(f"equality_satisfied = {equality_satisfied}\tnorm_residual = {norm_residual}")

            if equality_satisfied and norm_residual <= acc: 
                print(f"iter = {iter} converged!")
                break 
        
        return x

def test(): 
    opt = NewtonOptim()
    x_opt = opt.newton_method()
    print(f"x_opt = {x_opt} obj = {opt.objective(x_opt)} Ax - b = {opt.eq_constraints(x_opt)}")
    tmp = [0.1, 0.1] # broadcasting 
    print(f"x_opt+tmp = {x_opt+tmp} obj = {opt.objective(x_opt+tmp)} Ax - b = {opt.eq_constraints(x_opt+tmp)}")
    print(f"x_opt-tmp = {x_opt-tmp} obj = {opt.objective(x_opt-tmp)} Ax - b = {opt.eq_constraints(x_opt-tmp)}")
    return 

if __name__=='__main__': 
    test()