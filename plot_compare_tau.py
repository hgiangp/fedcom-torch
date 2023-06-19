import numpy as np 
import matplotlib.pyplot as plt
from parse_log import * 
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 22})

def get_data(prefix_log='./logs/mnist/s4/', log_file='system_model.log'):
    t_co, t_cp, t, e_co, e_cp, e = parse_net_tien(prefix_log + log_file)
    t_co = t_co.mean()
    t_cp = t_cp.mean()
    e_co = e_co.mean()*1000
    e_cp = e_cp.mean()*1000
    t = t.mean()
    e = e.mean()*1000
    return t_co, t_cp, t, e_co, e_cp, e

def plot_tien_bar(prefix_log='./logs/mnist/', prefix_fig='./figures/mnist/comparison/'):
    gamma=3
    C_n=1.2
    optim=1
    vec=40
    taus = [10, 15, 20, 30]
    log_files = [f'tau{tau}_gamma{gamma}_cn{C_n}_vec{vec}_optim{optim}.log' for tau in taus] 
    sce_idxes = [4]
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
    ylabels = ['Completion Time (s)', 'Consumption Energy (mJ)']
    fignames = ['tau_time', 'tau_energy']
    
    ys = [t_s, e_s]
    for t in range(len(ys)):
        vals = ys[t]

        fig, ax = plt.subplots()
        ax.grid(True, axis='both', color = '0.6', linestyle = '-')
        for i in range(len(sce_idxes)): 
            ax.plot(taus, vals[i], label=labels[i], marker=markers[i])
        ax.legend()
        xsticks = taus
        ax.set_xticks(xsticks)
        ax.set_xlim(xsticks[0], xsticks[-1])
        ax.set_ylabel(ylabels[t])
        ax.set_xlabel(r'Required Latency $\tau_g$ (s)')
        plt.savefig(f'{prefix_fig}{fignames[t]}.eps', bbox_inches='tight')
        plt.savefig(f'{prefix_fig}{fignames[t]}.png', bbox_inches='tight')
        plt.close()
    
def plot_tau_join(prefix_log='./logs/mnist/', prefix_fig='./figures/mnist/comparison/'):
    gamma=3
    C_n=1.2
    optim=1
    vec=40
    taus = [10, 15, 20, 30]
    log_files = [f'tau{tau}_gamma{gamma}_cn{C_n}_vec{vec}_optim{optim}.log' for tau in taus] 
    sce_idxes = [4]
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

    ylabel=['Completion Time (s)', 'Consumption Energy (mJ)']
    ys = [[t_co_s, t_cp_s], [e_co_s,  e_cp_s]]
    linestyles = ["solid", "dotted", "dashed", "dashdot"]
    labels = [['UBSTA, Communication', 'UBDYN, Communication'], ['UBSTA, Computation', 'UBDYN, Computation']]
    markers = ['*', 's', 'o', '^']
    fig_names = ['tau_time_join', 'tau_energy_join']
    colors = ['r', 'b']

    for t in range(len(ys)):
        fig, ax = plt.subplots()
        ax.grid(True, axis='both', color = '0.6', linestyle = '-')
        vals = ys[t]
        for k in range(len(vals)): 
            for i in range(len(sce_idxes)):
                ax.plot(taus, vals[k][i], label=labels[k][i], linestyle=linestyles[k], fillstyle='none', linewidth=2, markersize=9, color=colors[i], marker=markers[k])
        ax.set_ylabel(ylabel[t])
        ax.set_xlabel(r'Required Latency $\tau_g$ (s)')
        xsticks = taus
        ax.set_xticks(xsticks)
        ax.set_xlim(xsticks[0], xsticks[-1])
        ax.legend(fontsize=12)
        plt.savefig(f'{prefix_fig}{fig_names[t]}.eps', bbox_inches='tight')
        plt.savefig(f'{prefix_fig}{fig_names[t]}.png', bbox_inches='tight')
        plt.close()
if __name__=='__main__': 
    # plot_tien_bar()
    plot_tau_join()