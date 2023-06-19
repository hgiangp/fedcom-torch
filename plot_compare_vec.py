import numpy as np 
import matplotlib.pyplot as plt
import itertools
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
    C_n=1
    optim=1
    vecs=[40, 70, 90, 100]
    tau=15
    log_files = [f'tau{tau}_gamma{gamma}_cn{C_n}_vec{vec}_optim{optim}.log' for vec in vecs] 
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
    
    ys = [t_s, e_s]
    N = 3
    ind = np.arange(N) 
    width = 0.18
    space = 0.05
    ti_s = np.array([[t_cp_s[i], t_co_s[i], t_s[i]] for i in range(len(sce_idxes))])
    en_s = np.array([[e_cp_s[i], e_co_s[i], e_s[i]] for i in range(len(sce_idxes))])

    edgecolors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    hatches = ['xx', '\\\\', '//', 'xx']
    legends = ('BSTA', 'BDYN', 'UBSTA', 'UBDYN')
    xsticks = ['Computation', 'Communication', 'Total']
    ylabels = ['Time (s)', 'Energy Comsumption (mJ)']
    fignames = ['vec_time_bar', 'vec_energy_bar']
    delta_ys = [0.25, 0.001]
    ys = [ti_s, en_s]


    for t in range(len(ys)): 
        fig, ax = plt.subplots()
        ax.grid(True, axis = 'y', color = '0.6', linestyle = '-')
        bars = []
        for i in range(len(sce_idxes)): 
            bars.append(ax.bar(space*i+ind+width*i, ys[t][i], width, color='none', hatch=hatches[i], edgecolor=edgecolors[i], linewidth=1))
        
        ax.set_xticks(ind+1.5*width+1.5*space, xsticks, fontsize=16)
        ax.legend((bars[i] for i in range(len(bars))), legends, handlelength = 2, handleheight = 2, fontsize=16)
        ax.set_ylabel(ylabels[t])
        # ax.set_ylim(ylims[t])
        
        # Make some labels
        rects = ax.patches
        ti_int_s = np.around(ys[t], decimals=1)
        # print(ti_int_s)

        labels = [x for x in itertools.chain(ti_int_s[0].tolist(), ti_int_s[1].tolist(), ti_int_s[2].tolist(),  ti_int_s[3].tolist())]
        # print(labels)
        for rect, label in zip(rects, labels):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, height + delta_ys[t], label, ha="center", va="bottom", fontsize=16)

        plt.savefig(f'{prefix_fig}{fignames[t]}.png', bbox_inches='tight')
        plt.savefig(f'{prefix_fig}{fignames[t]}.eps', bbox_inches='tight')
        plt.close()
    
def plot_tau_join(prefix_log='./logs/mnist/', prefix_fig='./figures/mnist/comparison/'):
    gamma=3
    C_n=1
    optim=1
    vecs=[40, 70, 90, 100]
    tau=15
    log_files = [f'tau{tau}_gamma{gamma}_cn{C_n}_vec{vec}_optim{optim}.log' for vec in vecs] 
    sce_idxes = [3, 4]
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
    fig_names = ['vec_time', 'vec_energy']
    colors = ['r', 'b']

    for t in range(len(ys)):
        fig, ax = plt.subplots()
        ax.grid(True, axis='both', color = '0.6', linestyle = '-')
        vals = ys[t]
        for k in range(len(vals)): 
            for i in range(len(sce_idxes)):
                ax.plot(vecs, vals[k][i], label=labels[k][i], linestyle=linestyles[k], fillstyle='none', linewidth=2, markersize=9, color=colors[i], marker=markers[k])
        ax.set_ylabel(ylabel[t])
        ax.set_xlabel(r'Velocity (km/h)')
        xsticks = vecs
        ax.set_xticks(xsticks)
        ax.set_xlim(xsticks[0], xsticks[-1])
        ax.legend(fontsize=12)
        plt.savefig(f'{prefix_fig}{fig_names[t]}.eps', bbox_inches='tight')
        plt.savefig(f'{prefix_fig}{fig_names[t]}.png', bbox_inches='tight')
        plt.close()
if __name__=='__main__': 
    plot_tien_bar()
    plot_tau_join()