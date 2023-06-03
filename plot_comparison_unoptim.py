import numpy as np 
import matplotlib.pyplot as plt
from parse_log import * 

def plot_tien_bar(prefix_log='./logs/mnist/', prefix_fig='./figures/mnist/comparison/'):
    import itertools
    log_file_1 = 'system_model.log'
    log_file_2 = 'system_model_optim_freq.log'
    log_file_3 = 'system_model_optim_power.log'
    log_file_4 = 'system_model_unoptim.log'

    t_co_1, t_cp_1, t_1, e_co_1, e_cp_1, e_1 = parse_net_tien(prefix_log + 's4/' + log_file_1)
    t_co_2, t_cp_2, t_2, e_co_2, e_cp_2, e_2 = parse_net_tien(prefix_log + 's4/' + log_file_2)
    t_co_3, t_cp_3, t_3, e_co_3, e_cp_3, e_3 = parse_net_tien(prefix_log + 's4/' + log_file_3)
    t_co_4, t_cp_4, t_4, e_co_4, e_cp_4, e_4 = parse_net_tien(prefix_log + 's4/' + log_file_4)
    
    max_round = min(len(t_co_1), len(t_co_2))
    max_round_fixedi = len(t_co_1)

    t_co_1 = t_co_1.sum()
    t_cp_1 = t_cp_1.sum()
    e_co_1 = e_co_1.sum()
    e_cp_1 = e_cp_1.sum()

    t_co_2 = t_co_2.sum()
    t_cp_2 = t_cp_2.sum()
    e_co_2 = e_co_2.sum()
    e_cp_2 = e_cp_2.sum()

    t_co_3 = t_co_3.sum()
    t_cp_3 = t_cp_3.sum()
    e_co_3 = e_co_3.sum()
    e_cp_3 = e_cp_3.sum()

    t_co_4 = t_co_4.sum()
    t_cp_4 = t_cp_4.sum()
    e_co_4 = e_co_4.sum()
    e_cp_4 = e_cp_4.sum()


    t_1 = t_1.sum()
    t_2 = t_2.sum()
    t_3 = t_3.sum()
    t_4 = t_4.sum()
    
    e_1 = e_1.sum()
    e_2 = e_2.sum()
    e_3 = e_3.sum()
    e_4 = e_4.sum()
    
    N = 3
    ind = np.arange(N) 
    width = 0.18
    space = 0.03

    # plt.figure(1)
    fig, ax = plt.subplots()
    ax.grid(True, axis = 'y', color = '0.6', linestyle = '-')

    ti_1 = np.array([t_co_1, t_cp_1, t_1])
    ti_2 = np.array([t_co_2, t_cp_2, t_2])
    ti_3 = np.array([t_co_3, t_cp_3, t_3])
    ti_4 = np.array([t_co_4, t_cp_4, t_4])

    # greedy_vals = [720, 771, 812, 866]
    bar1 = ax.bar(ind, ti_1, width, color = 'none', hatch= 'xx', edgecolor = 'tab:blue', linewidth = 1 )
    
    bar2 = ax.bar(space + ind+width, ti_2, width, color = 'none', hatch= '\\\\', edgecolor='tab:orange', linewidth = 1)
    
    bar3 = ax.bar(space*2 + ind+width*2, ti_3, width,  color = 'none', hatch = '//', edgecolor = 'tab:green', linewidth = 1)

    bar4 = ax.bar(space*3 + ind+width*3, ti_4, width,  color = 'none', hatch = 'xx', edgecolor = 'tab:red', linewidth = 1)

    # ax.set_xlabel("Number of IDs", fontsize = 12)
    ax.set_ylabel('Time (s) ', fontsize = 12)
    # ax.set_ylim(0, 50)
    # plt.grid(True, axis = 'y', color = '0.6', linestyle = '-')
    
    ax.set_xticks(ind+width+space,['t_co', 't_cp', 'total'])
    ax.legend( (bar1, bar2, bar3, bar4), ('optim', 'opt_freq', 'opt_power', 'unoptim'), handlelength = 2, handleheight = 2, fontsize = 12)
    rects = ax.patches

    # Make some labels
    ti1_int = np.around(ti_1, decimals=2)
    ti2_int = np.around(ti_2, decimals=2)
    ti3_int = np.around(ti_3, decimals=2)
    ti4_int = np.around(ti_4, decimals=2)

    labels = [x for x in itertools.chain(ti1_int, ti2_int, ti3_int, ti4_int)]
    # labels = [x for x in itertools.chain(ti_1, ti_2, ti_3, ti_4)]

    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2, height + 1, label, ha="center", va="bottom"
        )

    fig_file_time = 'unoptim_time.png'
    plt.savefig(prefix_fig + fig_file_time, bbox_inches='tight')
    # plt.show()
    plt.close()

    # plt.figure(2)
    fig, ax = plt.subplots()
    ax.grid(True, axis = 'y', color = '0.6', linestyle = '-')

    en_1 = np.array([e_co_1, e_cp_1, e_1])
    en_2 = np.array([e_co_2, e_cp_2, e_2])
    en_3 = np.array([e_co_3, e_cp_3, e_3])
    en_4 = np.array([e_co_4, e_cp_4, e_4])

    bar1 = ax.bar(ind, en_1, width, color = 'none', hatch= 'xx', edgecolor = 'tab:blue', linewidth = 1 )
    bar2 = ax.bar(space + ind+width, en_2, width, color = 'none', hatch= '\\\\', edgecolor='tab:orange', linewidth = 1) 
    bar3 = ax.bar(space*2 + ind+width*2, en_3, width,  color = 'none', hatch = '//', edgecolor = 'tab:green', linewidth = 1)
    bar4 = ax.bar(space*3 + ind+width*3, en_4, width,  color = 'none', hatch = 'xx', edgecolor = 'tab:red', linewidth = 1)

    ax.set_ylabel('Energy (J) ', fontsize = 12)
    # ax.set_ylim(0, 0.2)
    # plt.grid(True, axis = 'y', color = '0.6', linestyle = '-')
    
    ax.set_xticks(ind+width+space,['e_co', 'e_cp', 'total'])
    ax.legend( (bar1, bar2, bar3, bar4), ('optim', 'opt_freq', 'opt_power', 'unoptim'), handlelength = 2, handleheight = 2, fontsize = 12)
    rects = ax.patches

    # Make some labels
    en1_int = np.around(en_1, decimals=2)
    en2_int = np.around(en_2, decimals=2)
    en3_int = np.around(en_3, decimals=2)
    en4_int = np.around(en_4, decimals=2)

    labels = [x for x in itertools.chain(en1_int, en2_int, en3_int, en4_int)]

    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2, height + 0.008, label, ha="center", va="bottom"
        )
    fig_file_ene = 'unoptim_ene.png'
    plt.savefig(prefix_fig + fig_file_ene, bbox_inches='tight')
    # plt.show()
    plt.close()

if __name__=='__main__': 
    plot_tien_bar()