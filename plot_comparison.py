import numpy as np 
import matplotlib.pyplot as plt
from parse_log import * 
from main import read_options

def plot_feld_performance(prefix_log='./logs/', prefix_fig='./figures/comparison/'):
    log_file = 'system_model.log'

    rounds_1, acc_1, loss_1, sim_1, tloss_1 = parse_fedl(prefix_log + 's1/' + log_file)
    rounds_2, acc_2, loss_2, sim_2, tloss_2 = parse_fedl(prefix_log + 's2/' + log_file)
    rounds_3, acc_3, loss_3, sim_3, tloss_3 = parse_fedl(prefix_log + 's3/' + log_file)
    rounds_4, acc_4, loss_4, sim_4, tloss_4 = parse_fedl(prefix_log + 's4/' + log_file)

    max_round = min(len(rounds_1), len(rounds_2), len(rounds_3), len(rounds_4))

    plt.figure(1)
    plt.subplot(311)
    plt.plot(rounds_1[:max_round], acc_1[:max_round], label='bs-fixedi')
    plt.plot(rounds_2[:max_round], acc_2[:max_round], label='bs-dyni')
    plt.plot(rounds_3[:max_round], acc_3[:max_round], label='bs-uav-fixedi')
    plt.plot(rounds_4[:max_round], acc_4[:max_round], label='bs-uav-dyni')
    plt.ylabel("Train Accuracy")
    plt.grid(which='both')
    plt.legend()

    plt.subplot(312)
    plt.plot(rounds_1[:max_round], loss_1[:max_round], label='bs-fixedi')
    plt.plot(rounds_2[:max_round], loss_2[:max_round], label='bs-dyni')
    plt.plot(rounds_3[:max_round], loss_3[:max_round], label='bs-uav-fixedi')
    plt.plot(rounds_4[:max_round], loss_4[:max_round], label='bs-uav-dyni')
    plt.ylabel("Train Loss")
    plt.grid(which='both')
    # plt.legend()

    plt.subplot(313)
    plt.plot(rounds_1[:max_round], tloss_1[:max_round], label='bs-fixedi')
    plt.plot(rounds_2[:max_round], tloss_2[:max_round], label='bs-dyni')
    plt.plot(rounds_3[:max_round], tloss_3[:max_round], label='bs-uav-fixedi')
    plt.plot(rounds_4[:max_round], tloss_4[:max_round], label='bs-uav-dyni')
    plt.ylabel("Test Loss")
    plt.grid(which='both')
    
    plt.savefig(prefix_fig + 'plot_synthetic_fedl.png') 
    # plt.show()
    plt.close()

def plot_tien_performance(prefix_log='./logs/', prefix_fig='./figures/comparison/'): 
    log_file = 'system_model.log'
    fig_file_time = 'synthetic_time.png'
    fig_file_ene = 'synthetic_ene.png'

    t_co_1, t_cp_1, e_co_1, e_cp_1 = parse_net_tien(prefix_log + 's1/' + log_file)
    t_co_2, t_cp_2, e_co_2, e_cp_2 = parse_net_tien(prefix_log + 's2/' + log_file)
    t_co_3, t_cp_3, e_co_3, e_cp_3 = parse_net_tien(prefix_log + 's3/' + log_file)
    t_co_4, t_cp_4, e_co_4, e_cp_4 = parse_net_tien(prefix_log + 's4/' + log_file)

    max_round = min(len(t_co_1), len(t_co_2), len(t_co_3), len(t_co_4))

    rounds = np.arange(0, max_round, 1) 
    
    # time, energy in each grounds  
    plt.figure(1, figsize=(10, 5))
    plt.subplot(121)
    plt.plot(rounds, t_co_1[:max_round], label='bs-fixedi')
    plt.plot(rounds, t_co_2[:max_round], label='bs-dyni')
    plt.plot(rounds, t_co_3[:max_round], label='bs-uav-fixedi')
    plt.plot(rounds, t_co_4[:max_round], label='bs-uav-dyni')
    plt.grid(visible=True, which='both')
    plt.ylabel('Communication time (s)')
    plt.legend()

    plt.subplot(122)
    plt.plot(rounds, t_cp_1[:max_round], label='bs-fixedi')
    plt.plot(rounds, t_cp_2[:max_round], label='bs-dyni')
    plt.plot(rounds, t_cp_3[:max_round], label='bs-uav-fixedi')
    plt.plot(rounds, t_cp_4[:max_round], label='bs-uav-dyni')
    plt.grid(visible=True, which='both')
    plt.ylabel('Computation time (s)')
    plt.legend()
    plt.savefig(prefix_fig + fig_file_time)
    # plt.show()
    plt.close()

    plt.figure(2, figsize=(10, 5))
    plt.subplot(121)
    plt.plot(rounds, e_co_1[:max_round], label='bs-fixedi')
    plt.plot(rounds, e_co_2[:max_round], label='bs-dyni')
    plt.plot(rounds, e_co_3[:max_round], label='bs-uav-fixedi')
    plt.plot(rounds, e_co_4[:max_round], label='bs-uav-dyni')
    plt.grid(visible=True, which='both')
    plt.ylabel('Communication energy (J)')
    plt.legend()

    plt.subplot(122)
    plt.plot(rounds, e_cp_1[:max_round], label='bs-fixedi')
    plt.plot(rounds, e_cp_2[:max_round], label='bs-dyni')
    plt.plot(rounds, e_cp_3[:max_round], label='bs-uav-fixedi')
    plt.plot(rounds, e_cp_4[:max_round], label='bs-uav-dyni')
    plt.grid(visible=True, which='both')
    plt.ylabel('Computation energy (J)')
    plt.legend()
    plt.savefig(prefix_fig + fig_file_ene)
    plt.close()
    # plt.show()

def plot_tien_bar(prefix_log='./logs/', prefix_fig='./figures/comparison/'):
    import itertools
    # plt.rcParams.update({'font.family':'Helvetica'})

    log_file = 'system_model.log'
    fig_file_time = 'synthetic_time_bar.png'
    fig_file_ene = 'synthetic_ene_bar.png'

    t_co_1, t_cp_1, e_co_1, e_cp_1 = parse_net_tien(prefix_log + 's1/' + log_file)
    t_co_2, t_cp_2, e_co_2, e_cp_2 = parse_net_tien(prefix_log + 's2/' + log_file)
    t_co_3, t_cp_3, e_co_3, e_cp_3 = parse_net_tien(prefix_log + 's3/' + log_file)
    t_co_4, t_cp_4, e_co_4, e_cp_4 = parse_net_tien(prefix_log + 's4/' + log_file)

    max_round = min(len(t_co_1), len(t_co_2), len(t_co_3), len(t_co_4))
    max_round_fixedi = min(len(t_co_1), len(t_co_3))
    
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
    
    N = 3
    ind = np.arange(N) 
    width = 0.18
    space = 0.03

    # plt.figure(1)
    fig, ax = plt.subplots()
    ax.grid(True, axis = 'y', color = '0.6', linestyle = '-')

    ti_1 = np.array([t_co_1, t_cp_1, t_co_1 + t_cp_1])
    ti_2 = np.array([t_co_2, t_cp_2, t_co_2 + t_cp_2])
    ti_3 = np.array([t_co_3, t_cp_3, t_co_3 + t_cp_3])
    ti_4 = np.array([t_co_4, t_cp_4, t_co_4 + t_cp_4])

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
    ax.legend( (bar1, bar2, bar3, bar4), ('bs-fixedi', 'bs-dyni', 'bs-uav-fixedi', 'bs-uav-dyni'), handlelength = 2, handleheight = 2, fontsize = 12)
    rects = ax.patches

    # Make some labels
    ti1_int = np.array([int(ti1) for ti1 in ti_1])
    ti2_int = np.array([int(ti2) for ti2 in ti_2])
    ti3_int = np.array([int(ti3) for ti3 in ti_3])
    ti4_int = np.array([int(ti4) for ti4 in ti_4])

    labels = [x for x in itertools.chain(ti1_int, ti2_int, ti3_int, ti4_int)]
    # labels = [x for x in itertools.chain(ti_1, ti_2, ti_3, ti_4)]

    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2, height + 1, label, ha="center", va="bottom"
        )

    plt.savefig(prefix_fig + fig_file_time, bbox_inches='tight')
    # plt.show()
    plt.close()

    # plt.figure(2)
    fig, ax = plt.subplots()
    ax.grid(True, axis = 'y', color = '0.6', linestyle = '-')

    en_1 = np.array([e_co_1, e_cp_1, e_co_1 + e_cp_1])
    en_2 = np.array([e_co_2, e_cp_2, e_co_2 + e_cp_2])
    en_3 = np.array([e_co_3, e_cp_3, e_co_3 + e_cp_3])
    en_4 = np.array([e_co_4, e_cp_4, e_co_4 + e_cp_4])

    bar1 = ax.bar(ind, en_1, width, color = 'none', hatch= 'xx', edgecolor = 'tab:blue', linewidth = 1 )
    bar2 = ax.bar(space + ind+width, en_2, width, color = 'none', hatch= '\\\\', edgecolor='tab:orange', linewidth = 1) 
    bar3 = ax.bar(space*2 + ind+width*2, en_3, width,  color = 'none', hatch = '//', edgecolor = 'tab:green', linewidth = 1)
    bar4 = ax.bar(space*3 + ind+width*3, en_4, width,  color = 'none', hatch = 'xx', edgecolor = 'tab:red', linewidth = 1)

    ax.set_ylabel('Energy (J) ', fontsize = 12)
    # ax.set_ylim(0, 0.2)
    # plt.grid(True, axis = 'y', color = '0.6', linestyle = '-')
    
    ax.set_xticks(ind+width+space,['e_co', 'e_cp', 'total'])
    ax.legend( (bar1, bar2, bar3, bar4), ('bs-fixedi', 'bs-dyni', 'bs-uav-fixedi', 'bs-uav-dyni'), handlelength = 2, handleheight = 2, fontsize = 12)
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

    plt.savefig(prefix_fig + fig_file_ene, bbox_inches='tight')
    # plt.show()
    plt.close()

def plot_lround(prefix_log='./logs/', prefix_fig='./figures/comparison/'):
    log_file = 'system_model.log'

    lrounds_1, _, _, _ = parse_netopt(prefix_log + 's1/' + log_file)
    lrounds_2, _, _, _ = parse_netopt(prefix_log + 's2/' + log_file)
    lrounds_3, _, _, _ = parse_netopt(prefix_log + 's3/' + log_file)
    lrounds_4, _, _, _ = parse_netopt(prefix_log + 's4/' + log_file)

    max_round = min(len(lrounds_1), len(lrounds_2), len(lrounds_3), len(lrounds_4))
    rounds = np.arange(max_round)
    
    plt.figure(1)
    plt.plot(rounds, lrounds_1[:max_round], label='bs-fixedi')
    plt.plot(rounds, lrounds_2[:max_round], label='bs-dyni')
    plt.plot(rounds, lrounds_3[:max_round], label='bs-uav-fixedi')
    plt.plot(rounds, lrounds_4[:max_round], label='bs-uav-dyni')
    plt.ylabel("# Local Rounds")
    plt.grid(which='both')
    plt.legend()

    plt.savefig(prefix_fig + 'plot_synthetic_lround.png') 
    # plt.show()
    plt.close()


if __name__=='__main__':
    options, _ = read_options()
    dataset=options['dataset']
    
    prefix_log = f'./logs/{dataset}/'
    prefix_fig = f'./figures/{dataset}/comparison/'

    plot_feld_performance(prefix_log, prefix_fig)
    plot_tien_performance(prefix_log, prefix_fig)
    plot_tien_bar(prefix_log, prefix_fig)
    plot_lround(prefix_log, prefix_fig)