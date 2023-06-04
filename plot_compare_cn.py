import numpy as np 
import matplotlib.pyplot as plt
from parse_log import * 
plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams.update({'font.size': 10})

def get_data(prefix_log='./logs/mnist/s4/', log_file='system_model.log'):
    t_co, t_cp, t, e_co, e_cp, e = parse_net_tien(prefix_log + log_file)
    t_co = t_co.sum()
    t_cp = t_cp.sum()
    e_co = e_co.sum()
    e_cp = e_cp.sum()
    t = t.sum()
    e = e.sum()
    return t_co, t_cp, t, e_co, e_cp, e

def plot_tien_bar(prefix_log='./logs/mnist/', prefix_fig='./figures/mnist/comparison/'):
    tau = 20.0 
    gamma = 2.0 
    c_ns = [0.1, 0.25, 0.4, 0.5, 0.6, 0.8, 0.9, 0.95, 1.0]
    log_files = [f'system_model_tau{tau}_gamma{gamma}_cn{cn}.log' for cn in c_ns] 
    sce_idxes = [1, 2, 3, 4]
    prefixes = [prefix_log + f's{sce_idx}/' for sce_idx in sce_idxes]

    t_co_s, t_cp_s, t_s, e_co_s, e_cp_s, e_s = [], [], [], [], [], []

    for prefix in prefixes: 
        t_co_tmp, t_cp_tmp, t_tmp, e_co_tmp, e_cp_tmp, e_tmp = [], [], [], [], [], []
        for log_file in log_files: 
            t_co, t_cp, t, e_co, e_cp, e = get_data(prefix_log=prefix, log_file=log_file)
            t_co_tmp.append(t_co)
            t_cp_tmp.append(t_cp)
            t_tmp.append(t)
            e_co_tmp.append(e_co)
            e_cp_tmp.append(e_cp)
            e_tmp.append(e)
        t_co_s.append(t_co_tmp)
        t_cp_s.append(t_cp_tmp)
        t_s.append(t_tmp)
        e_co_s.append(e_co_tmp)
        e_cp_s.append(e_cp_tmp)
        e_s.append(e_tmp)

    # plt.figure(1)
    fig, ax = plt.subplots()
    ax.grid(True, axis='both', color = '0.6', linestyle = '-')

    ylabel=['Communication time (s)', 'Computation time (s)', 'Communication energy (J)', 'Computation energy (J)']
    ys = [t_co_s, t_cp_s, e_co_s,  e_cp_s]
    linestyles = ["solid", "dotted", "dashed", "dashdot"]
    labels = ['bs-fixedi', 'bs-dyni', 'bs-uav-fixedi', 'bs-uav-dyni']
    markers = ['s', 'o', '^', '*']
    fig_names = ['cn_time_co.png', 'cn_time_cp.png', 'cn_energy_co.png', 'cn_energy_cp.png']

    for t in range(len(ys)):
        fig, ax = plt.subplots()
        ax.grid(True, axis='both', color = '0.6', linestyle = '-')
        for i in range(len(ys[t])): 
            ax.plot(c_ns, ys[t][i], label=labels[i], marker=markers[i], fillstyle='none')
        ax.set_ylabel(ylabel[t])
        ax.set_xlabel(r'$C_n$')
        # xsticks = [9.0, 10.0, 12.0, 15.0, 18.0, 20.0, 22.0, 25.0]
        # ax.set_xticks(xsticks)
        # ax.set_xlim(xsticks[0], xsticks[-1])
        ax.legend()
        plt.savefig(prefix_fig + fig_names[t], bbox_inches='tight')
        plt.close()
        
if __name__=='__main__': 
    plot_tien_bar()