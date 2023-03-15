import numpy as np 


a, b, c = 1, 1, 1
kappa = 1

def objective(x): 
    z, t = x[0], x[1]
    f0 = a/b * z * (np.exp(1/z) - 1) + kappa * c / (t**2)
    print(f"objective = {f0}")

    return f0 

def hessian_dual(x): 
    z, t = x[0], x[1]
    hess = np.array([[a/b * np.exp(1/z) / (z**3), 0, a],
                    [0, 6 * kappa * c / (t**4), c], 
                    [a, c, 0]])

    print(f"hessian_dual = {hess}")
    return hess 

def hessian(x): 
    z, t = x[0], x[1]
    hess = np.array([[a/b * np.exp(1/z) / (z**3), 0],
                    [0, 6 * kappa * c / (t**4)]])
    
    print(f"hessian = {hess}")
    return hess 

def gradient(x): 
    z, t = x[0], x[1]
    grad = np.array([a/b * ((1 - 1/z) * np.exp(1/z) - 1), -2 * kappa * c / (t**3)])

    print(f"gradient = {grad}")
    return grad 

def backtracking_line_search(x, dir_x, alpha=0.01, beta=0.8, step_size=1): 
    print(f"backtracking_line_search dir_x = {dir_x}")
    while (objective(x + step_size * dir_x) > (objective(x) + alpha * step_size * np.dot(gradient(x), dir_x))): 
        step_size = beta * step_size
    
    print(f"backtracking_line_search step_size = {step_size}")
    return step_size

class NewtonMethod(object): 
    def __init__(self, max_iter=100):
        self.max_iter = max_iter 

    def optimize(self): 
        # initiate feasible starting point 
        x_max = [1, 1] 
        x = x_max
        acc = 1e-5

        dim = 2

        # repeat 
        for iter in range(self.max_iter): 
            # compute newton step dir_x_nt, newton decrement lambda_x 
            grad_x = gradient(x) # (N, )
            grad_x_nt = np.append(grad_x, 0) # (N+1, )
            inv_hess_x = np.linalg.inv(hessian_dual(x)) # (N+1, N+1)

            dir_x = - np.dot(inv_hess_x, grad_x_nt)[:dim] # (N, )
            decrement_x_squared = - np.dot(np.transpose(grad_x), dir_x)  # (1)

            # stopping criterion: quit if lambda^2 \leq epsilon 
            if decrement_x_squared / 2 < acc: 
                print(f"iter = {iter} converged\tdecrement_x_squared = {decrement_x_squared}")
                break 
            
            # line search: choose stepsize t by backtracking line search 
            step_size = backtracking_line_search(x, dir_x)

            # update: x = x + t * dir_x_nt 
            x = x + step_size * dir_x
            print(f"iter = {iter} step_size = {step_size} x = {x}")
        
        return x

def test(): 
    opt = NewtonMethod()

    x_opt = opt.optimize()
    print(f"x_opt = {x_opt} obj = {objective(x_opt)}")
    tmp = [0.1, 0.1] # broadcasting 
    print(f"x_opt+tmp = {x_opt+tmp} obj = {objective(x_opt+tmp)}")
    print(f"x_opt-tmp = {x_opt-tmp} obj = {objective(x_opt-tmp)}")
    return 

if __name__=='__main__': 
    test()