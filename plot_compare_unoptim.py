import numpy as np 
import matplotlib.pyplot as plt
import itertools
from parse_log import *
plt.rcParams["font.family"] = "Times New Roman"

def get_data(prefix_log='./logs/mnist/s4/', log_file='system_model.log'):
    t_co, t_cp, t, e_co, e_cp, e = parse_net_tien(prefix_log + log_file)
    t_co = t_co.sum()
    t_cp = t_cp.sum()
    e_co = e_co.sum()
    e_cp = e_cp.sum()
    t = t.sum()
    e = e.sum()
    return t_co, t_cp, t, e_co, e_cp, e

def plot_tien_bar(prefix_log='./logs/mnist/s4/', prefix_fig='./figures/mnist/comparison/'):
    gamma=2.0
    C_n=0.01
    tau=2.53
    optims = [1, 2, 3, 4]
    lognames = [f'system_model_tau{tau}_gamma{gamma}_cn{C_n}_optim{optim}.log' for optim in optims]
    # postfix = f'tau{tau}_gamma{gamma}_cn{C_n}'
    t_co_s, t_cp_s, t_s, e_co_s, e_cp_s, e_s = [], [], [], [], [], []
    for logname in lognames: 
        t_co, t_cp, t, e_co, e_cp, e = get_data(prefix_log, logname)
        t_co_s.append(t_co)
        t_cp_s.append(t_cp)
        t_s.append(t)
        e_co_s.append(e_co)
        e_cp_s.append(e_cp)
        e_s.append(e)

    N = 3
    ind = np.arange(N) 
    width = 0.18
    space = 0.03

    # plt.figure(1)
    
    ti_s = np.array([[t_co_s[i], t_cp_s[i], t_s[i]] for i in range(len(optims))])
    en_s = np.array([[e_co_s[i], e_cp_s[i], e_s[i]] for i in range(len(optims))])

    edgecolors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    hatches = ['xx', '\\\\', '//', 'xx']
    legends = ('UBDYN', 'OPTFR', 'OPTPO', 'UNOPT')
    xsticks = ['Communication', 'Computation', 'Total']
    ylabels = ['Time (s)', 'Energy Comsumption (J)']
    fignames = ['unoptim_time', 'unoptim_ene']
    delta_ys = [0.25, 0.05]

    ys = [ti_s, en_s]

    for t in range(len(ys)): 
        fig, ax = plt.subplots()
        ax.grid(True, axis = 'y', color = '0.6', linestyle = '-')
        bars = []
        for i in range(len(optims)): 
            bars.append(ax.bar(space*i+ind+width*i, ys[t][i], width, color='none', hatch=hatches[i], edgecolor=edgecolors[i], linewidth=1))
        
        ax.set_xticks(ind+1.5*width+1.5*space, xsticks)
        ax.legend((bars[i] for i in range(len(bars))), legends, handlelength = 2, handleheight = 2)
        ax.set_ylabel(ylabels[t])
        
        # Make some labels
        rects = ax.patches
        ti_int_s = np.around(ys[t], decimals=2)

        labels = [x for x in itertools.chain(ti_int_s[0].tolist(), ti_int_s[1].tolist(), ti_int_s[2].tolist(),  ti_int_s[3].tolist())]
        # labels = [x for x in itertools.chain(ti_int_s[0].tolist())]
        # print(labels)
        for rect, label in zip(rects, labels):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, height + delta_ys[t], label, ha="center", va="bottom")

        plt.savefig(f'{prefix_fig}{fignames[t]}.png', bbox_inches='tight')
        plt.savefig(f'{prefix_fig}{fignames[t]}.eps', bbox_inches='tight')
        plt.close()
    

if __name__=='__main__': 
    plot_tien_bar()