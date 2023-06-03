import numpy as np 
import matplotlib.pyplot as plt
from parse_log import * 

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
    taus = [9.0, 10.0, 12.0, 15.0, 18.0, 20.0, 22.0, 25.0]
    log_files = [f'system_model_tau{tau}.log' for tau in taus] 
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

    linestyles = ["solid", "dotted", "dashed", "dashdot"]
    labels = ['bs-fixedi', 'bs-dyni', 'bs-uav-fixedi', 'bs-uav-dyni']
    markers = ['s', 'o', '^', '*']
    for i in range(len(sce_idxes)): 
        ax.plot(taus, t_s[i], label=labels[i], marker=markers[i])
    ax.legend()
    ax.set_ylabel('Completion time (s)')
    ax.set_xlabel(r'Required latency $\tau_g$ (s)')
    fig_file_time = 'tau_time.png'
    plt.savefig(prefix_fig + fig_file_time, bbox_inches='tight')
    plt.close()

    # plt.figure(2)
    fig, ax = plt.subplots()
    ax.grid(True, axis='y', color = '0.6', linestyle = '-')

    for i in range(len(sce_idxes)): 
        ax.plot(taus, e_s[i], label=labels[i], marker=markers[i])
    # ax.set_ylim(0, 2.2)
    ax.set_ylabel('Energy Consumption (J)')
    ax.set_xlabel(r'Required Latency $\tau_g$ (s)')
    xsticks = [9.0, 10.0, 12.0, 15.0, 18.0, 20.0, 22.0, 25.0]
    ax.set_xticks(xsticks)
    ax.set_xlim(xsticks[0], xsticks[-1])
    ax.legend()
    fig_file_energy = 'tau_energy.png'
    plt.savefig(prefix_fig + fig_file_energy, bbox_inches='tight')
    plt.close()

if __name__=='__main__': 
    plot_tien_bar()