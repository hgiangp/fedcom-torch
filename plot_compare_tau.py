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
    gamma=2.0
    C_n=0.2
    taus = [8.0, 9.0, 10.0, 12.0, 15.0, 18.0, 20.0]
    log_files = [f'system_model_tau{tau}_gamma{gamma}_cn{C_n}.log' for tau in taus] 
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

    linestyles = ["solid", "dotted", "dashed", "dashdot"]
    labels = ['BSTA', 'BDYN', 'UBSTA', 'UBDYN']
    markers = ['s', 'o', '^', '*']
    ylabels = ['Completion time (s)', 'Energy Consumption (J)']
    fignames = ['tau_time', 'tau_energy']
    
    ys = [t_s, e_s]
    for t in range(len(ys)):
        vals = ys[t]

        fig, ax = plt.subplots()
        ax.grid(True, axis='both', color = '0.6', linestyle = '-')
        for i in range(len(sce_idxes)): 
            ax.plot(taus, vals[i], label=labels[i], marker=markers[i])
        ax.legend()
        xsticks = [8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
        ax.set_xticks(xsticks)
        ax.set_xlim(xsticks[0], xsticks[-1])
        ax.set_ylabel(ylabels[t])
        ax.set_xlabel(r'Required latency $\tau_g$ (s)')
        plt.savefig(f'{prefix_fig}{fignames[t]}.eps', bbox_inches='tight')
        plt.savefig(f'{prefix_fig}{fignames[t]}.png', bbox_inches='tight')
        plt.close()

if __name__=='__main__': 
    plot_tien_bar()