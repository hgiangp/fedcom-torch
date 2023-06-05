num_users = 10 # number of participating vehicles 
max_uav = 5 # maximum number of vehicles connecting to uav 

# Local computation params 
L_Lipschitz = 5 # Lipschitz constant of the loss function
# delta_lr = 5*1e-3 # TODO Learning rate of local surrogate function 
gamma_cv = 3 # strongly convex constant of the loss function

k_switch = 1e-28 # switch capacity, depends on chip's architecture 
C_n = 0.05*1e4 # TODO: testback # number of cpu cycles per sample 
# D_n = 100 # data size, number of samples, varies  
freq_max = 2 * 1e9 # 2GHz -> Hz, maximum cpu computation frequency 

# Offloading, global aggregation params 
xi_factor = 1 # global gradient factor 
epsilon_0 = 1e-3 # global accuracy 

compression_ratio = 0.1 # 1-0.95
s_n = 300240 # 502400*compression_ratio=25120 bits # data transmission size TODO (784 * 10 + 10) * 2 * 32 = 502400 bits 
bw = 1e6 # 1MHz bandwidth 
delta_t = 0.0 # TODO Based on current results uav's time penalty 
N_0 = 4*1e-15 # -174 dBm?/Hz, noise density, should be multiplied with bw: 1e(-17.4)*1e6 = 4*1e(-12) (4*1e-15?)
power_max = 0.1 # W TODO  

# base station propagation channel params 
x_bs, y_bs = 0, -500 # m (250/sqrt(2), cover radius = 500m, at the cell edge)
A_d = 3 # attenna gain 
f_c = 915 * 1e6 # MHz -> Hz, carrier frequency
c = 3 * 1e8 # speed of light 
de_r = 2.65 # pathloss exponent, connect to bs 
# dn_r = 1.0 # distance between veh and bs, varies 

# UAV propagation channel params 

# Params from: Dynamic Offloading and Trajectory Control for UAV-Enabled 
# Mobile Edge Computing System With Energy Harvesting Devices
x_uav, y_uav = 200, 220
z_uav = 100 # flighting height of uav 
g_0 = 1e-5 # -50 dB reference channel gain (10**(x/10))
alpha = 0.2 # < 1, attenuation effect of NLoS channel 
de_u = 2.3 # pathloss exponent, connect to uav 
a_env, b_env = 15, 0.5 # evironment constants

# Optimization params 
acc = 1e-4 # TODO
iter_max = 50 # TODO