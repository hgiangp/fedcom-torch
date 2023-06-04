import numpy as np 
import matplotlib.pyplot as plt
from parse_log import * 
from main import read_options
plt.rcParams["font.family"] = "Times New Roman"

def plot_feld(prefix_log='./logs/', prefix_fig='./figures/comparison/', log_file = 'system_model.log'):
    rounds, acc, loss, sim, tloss = [], [], [], [], []
    sce_idxes = [1, 2, 3, 4]
    prefixes = [prefix_log + f's{sce_idx}/' for sce_idx in sce_idxes]

    for prefix in prefixes: 
        rounds_tmp, acc_tmp, loss_tmp, sim_tmp, tloss_tmp = parse_fedl(prefix + log_file)
        rounds.append(rounds_tmp)
        acc.append(acc_tmp)
        loss.append(loss_tmp)
        sim.append(sim_tmp)
        tloss.append(tloss_tmp)

    max_round = min(len(rounds[i]) for i in range(len(sce_idxes)))

    ylabels = ["Train Accuracy", "Train Loss", "Test Loss"]
    legends = ['bs-fixedi', 'bs-dyni', 'bs-uav-fixedi', 'bs-uav-dyni']
    fignames = ['train_acc.png', 'train_loss.png', 'test_loss.png']
    ys = [acc, loss, tloss]

    ystick1 = [20, 40, 60, 80, 90]
    ystick2 = [0.1, 0.5, 1, 1.5, 2.0, 2.5]
    ystick3 = [0.35, 0.5, 1, 1.5, 2.0, 2.5]
    ysticks = [ystick1, ystick2, ystick3]
    
    for t in range(len(ys)): 
        fig, ax = plt.subplots()
        ax.grid(which='both')
        for i in range(len(sce_idxes)): 
            ax.plot(rounds[i][:max_round], acc[i][:max_round], label=legends[i])
        ax.legend()
        ax.set_yticks(ysticks[t])
        ax.set_ylabel(ylabels[t])
        plt.savefig(prefix_fig + fignames[t], bbox_inches='tight')
        plt.close()

def plot_tien_performance(prefix_log='./logs/', prefix_fig='./figures/comparison/', log_file = 'system_model.log'): 
    t_co, t_cp, t, e_co, e_cp, e = [], [], [], [], [], []
    sce_idxes = [1, 2, 3, 4]
    prefixes = [prefix_log + f's{sce_idx}/' for sce_idx in sce_idxes]

    for prefix in prefixes: 
        t_co_tmp, t_cp_tmp, t_tmp, e_co_tmp, e_cp_tmp, e_tmp = parse_net_tien(prefix + log_file)
        t_co.append(t_co_tmp)
        t_cp.append(t_cp_tmp)
        t.append(t_tmp)
        e_co.append(e_co_tmp)
        e_cp.append(e_cp_tmp)
        e.append(e_tmp)
    num_sces = len(sce_idxes)
    max_round = min(len(t_co[i]) for i in range(num_sces))
    rounds = np.arange(0, max_round, 1) 
    ylabels = [["Communication time (s)", "Computation time (s)"], ["Communication energy (J)", "Computation energy (J)"]]
    legends = ['bs-fixedi', 'bs-dyni', 'bs-uav-fixedi', 'bs-uav-dyni']
    fignames = ['synthetic_time.png', 'synthetic_ene.png']

    ys = [[t_co, t_cp], [e_co, e_cp]]

    for t in range(len(fignames)): 
        plt.figure(t, figsize=(10, 5))
        plt.subplot(121)
        for i in range(len(sce_idxes)): 
            plt.plot(rounds, ys[t][0][i][:max_round], label=legends[i])
        plt.grid(visible=True, which='both')
        plt.ylabel(ylabels[t][0])
        plt.legend()

        plt.subplot(122)
        for i in range(len(sce_idxes)): 
            plt.plot(rounds, ys[t][1][i][:max_round], label=legends[i])
        plt.grid(visible=True, which='both')
        plt.ylabel(ylabels[t][1])
        plt.legend()

        plt.savefig(prefix_fig + fignames[t], bbox_inches='tight')
        plt.close()

def get_data(prefix_log='./logs/mnist/s4/', log_file='system_model.log'):
    t_co, t_cp, t, e_co, e_cp, e = parse_net_tien(prefix_log + log_file)
    t_co = t_co.sum()
    t_cp = t_cp.sum()
    e_co = e_co.sum()
    e_cp = e_cp.sum()
    t = t.sum()
    e = e.sum()
    return t_co, t_cp, t, e_co, e_cp, e

def plot_tien_bar(prefix_log='./logs/', prefix_fig='./figures/comparison/', log_file = 'system_model.log'):
    import itertools
    # plt.rcParams.update({'font.family':'Helvetica'})

    sce_idxes = [1, 2, 3, 4]
    prefixes = [prefix_log + f's{sce_idx}/' for sce_idx in sce_idxes]
    
    t_co_s, t_cp_s, t_s, e_co_s, e_cp_s, e_s = [], [], [], [], [], []
    for prefix in prefixes: 
        t_co, t_cp, t, e_co, e_cp, e = get_data(prefix_log=prefix, log_file=log_file)
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
    
    ti_s = np.array([[t_co_s[i], t_cp_s[i], t_s[i]] for i in range(len(sce_idxes))])
    en_s = np.array([[e_co_s[i], e_cp_s[i], e_s[i]] for i in range(len(sce_idxes))])

    edgecolors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    hatches = ['xx', '\\\\', '//', 'xx']
    legends = ('BSTA', 'BDYN', 'UBSTA', 'UBDYN')
    xsticks = ['Communication', 'Computation', 'Total']
    ylabels = ['Time (s)', 'Energy (J)']
    fignames = ['synthetic_time_bar', 'synthetic_ene_bar']
    delta_ys = [0.25, 0.05]

    ys = [ti_s, en_s]

    for t in range(len(ys)): 
        fig, ax = plt.subplots()
        ax.grid(True, axis = 'y', color = '0.6', linestyle = '-')
        bars = []
        for i in range(len(sce_idxes)): 
            bars.append(ax.bar(space*i+ind+width*i, ys[t][i], width, color='none', hatch=hatches[i], edgecolor=edgecolors[i], linewidth=1))
        
        ax.set_xticks(ind+1.5*width+1.5*space, xsticks)
        # ax.legend((bars[i] for i in range(len(bars))), legends, handlelength = 2, handleheight = 2, fontsize = 12)
        # ax.set_ylabel(ylabels[t], fontsize = 12)
        ax.legend((bars[i] for i in range(len(bars))), legends, handlelength = 2, handleheight = 2)
        ax.set_ylabel(ylabels[t])
        
        # Make some labels
        rects = ax.patches
        ti_int_s = np.around(ys[t], decimals=2)
        # print(ti_int_s)

        labels = [x for x in itertools.chain(ti_int_s[0].tolist(), ti_int_s[1].tolist(), ti_int_s[2].tolist(),  ti_int_s[3].tolist())]
        # print(labels)
        for rect, label in zip(rects, labels):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, height + delta_ys[t], label, ha="center", va="bottom")

        plt.savefig(f'{prefix_fig}{fignames[t]}.png', bbox_inches='tight')
        plt.savefig(f'{prefix_fig}{fignames[t]}.eps', bbox_inches='tight')
        plt.close()
    
def plot_lround(prefix_log='./logs/', prefix_fig='./figures/comparison/',  log_file = 'system_model.log'):
    lrounds = []
    sce_idxes = [1, 2, 3, 4]
    prefixes = [prefix_log + f's{sce_idx}/' for sce_idx in sce_idxes]
    for prefix in prefixes: 
        lrounds_tmp, _, _, _ = parse_netopt(prefix + log_file)
        lrounds.append(lrounds_tmp)

    max_round = min(len(lrounds[i]) for i in range(len(sce_idxes)))
    rounds = np.arange(max_round)
    legends = ('bs-fixedi', 'bs-dyni', 'bs-uav-fixedi', 'bs-uav-dyni')

    
    plt.figure(1)
    for i in range(len(sce_idxes)):     
        plt.plot(rounds, lrounds[i][:max_round], label=legends[i])
    plt.ylabel("# Local Rounds")
    plt.grid(which='both')
    plt.legend()
    plt.savefig(prefix_fig + 'plot_synthetic_lround.png')
    plt.close()


if __name__=='__main__':
    options, _ = read_options()
    dataset=options['dataset']

    prefix_log = f'./logs/{dataset}/'
    prefix_fig = f'./figures/{dataset}/comparison/'
    log_file = 'system_model_tau20.0_gamma2.0_cn0.5.log'

    plot_tien_performance(prefix_log, prefix_fig, log_file)
    plot_tien_bar(prefix_log, prefix_fig, log_file)
    plot_lround(prefix_log, prefix_fig, log_file)
    plot_feld(prefix_log, prefix_fig, log_file)