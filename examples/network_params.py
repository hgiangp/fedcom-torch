import math 

num_users = 10 # number of participating vehicles 
tau = 1 # total time threshold  

# Local computation params 
L_Lipschitz = 1.0 # L-Lipschitz of the loss function
delta_lr = 0.1 # Learning rate of local surrogate function 
gamma_cv = 0.1 # gamma-strongly convex of the loss function

# k_switch = 1e-26 # switch capacity, depends on chip's architecture 
k_switch = 2 # TODO: test back switch capacity, depends on chip's architecture
C_n = 3 #10e7 TODO: testback # number of cpu cycles per sample 
# D_n = 100 # data size, number of samples, varies  
freq_max = 3 # TODO: testback, maximum cpu computation frequency 

# Offloading, global aggregation params 
xi_factor = 1.0 # global gradient factor 
epsilon_0 = 1e-4 # global accuracy 
s_n = 1e3 # data transmission size 
bw = 1e6 # bandwidth 
delta_t = 2 # uav's time penalty 
N_0 = 2 # -174, TODO # dB, noise density, should be multiplied with bw
power_max = 3 # TODO 

# base station propagation channel params 
A_d = 3 # attenna gain 
f_c = 915 # MHz, carrier frequency
c = 3 * 1e8 # speed of light 
de_r = 3 # pathloss exponent, connect to bs 
# dn_r = 1.0 # distance between veh and bs, varies 

# UAV propagation channel params 
# Lu = (0, 0, z) # location of uav 
z_uav = 100 # flighting height of uav 
# Ln = (x_n, y_n, 0) # location of veh n at itertion t 
g_0 = 1 # reference channel gain 
alpha = 0.6 # < 1, attenutation effect of NLoS channel 
de_u = 3 # pathloss exponent, connect to uav 
a_env, b_env = 0.1, 0.2 # evironment constants

# Calculable params 
v = 2 / ((2 - L_Lipschitz * delta_lr) * gamma_cv * delta_lr)
a = 2 * (L_Lipschitz**2) * math.log(1/epsilon_0) / ((gamma_cv**2) * xi_factor)

# Optimization params 
acc = 1e-4 # TODO
iter_max = 10 # TODO