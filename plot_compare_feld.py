import numpy as np 
import matplotlib.pyplot as plt
from parse_log import * 
from main import read_options

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 22})

def plot_feld(prefix_log='./logs/', prefix_fig='./figures/comparison/', log_file = 'system_model.log'):
    # log_file = 'system_model_tau20.0_gamma2.0_cn0.5.log' # xi = 1
    files = [log_file]

    rounds, acc, loss, sim, tloss = [], [], [], [], []
    sce_idxes = [1, 2, 3, 4]
    prefixes = [prefix_log + f's{sce_idx}/' for sce_idx in sce_idxes]

    for file in files: 
        roundsf, accf, lossf, simf, tlossf = [], [], [], [], []
        for prefix in prefixes: 
            rounds_tmp, acc_tmp, loss_tmp, sim_tmp, tloss_tmp = parse_fedl(prefix + file)
            roundsf.append(rounds_tmp)
            accf.append(acc_tmp)
            lossf.append(loss_tmp)
            simf.append(sim_tmp)
            tlossf.append(tloss_tmp)
        rounds.append(roundsf)
        acc.append(accf)
        loss.append(lossf)
        sim.append(simf)
        tloss.append(tlossf)
    
    num_sces = len(sce_idxes)
    num_logs = len(files)

    max_round = min(len(rounds[i][j]) for i in range(num_logs) for j in range(num_sces))
    ylabels = ["Testing Accuracy", "Training Loss", "Testing Loss"]
    legends = ['BSTA', 'BDYN', 'UBSTA', 'UBDYN']
    fignames = ['test_acc', 'train_loss', 'test_loss']
    ys = [acc, loss, tloss]

    ystick1 = [20, 40, 60, 80, 90]
    ystick2 = [0.1, 0.5, 1, 1.5, 2.0, 2.5]
    ystick3 = [0.35, 0.5, 1, 1.5, 2.0, 2.5]
    ysticks = [ystick1, ystick2, ystick3]
    
    for t in range(len(ys)): 
        fig, ax = plt.subplots()
        ax.grid(which='both')
        vals = ys[t]
        for k in range(num_logs): 
            for i in range(num_sces): 
                ax.plot(rounds[k][i][:max_round], vals[k][i][:max_round], label=legends[i])
            
        ax.legend()
        # ax.set_yticks(ysticks[t])
        ax.set_ylabel(ylabels[t])
        # ax.set_xticks([10, 20, 30, 40, 50, 60, 70, 80])
        ax.set_xlim(0, 35)
        # ax.set_xlim(0, 20)
        ax.set_xlabel('Global Rounds')
        plt.savefig(f'{prefix_fig}{fignames[t]}.eps', bbox_inches='tight')
        plt.savefig(f'{prefix_fig}{fignames[t]}.png', bbox_inches='tight')
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
    fignames = ['time_plot.png', 'energy_plot.png']

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
        e_co_s.append(e_co*1000)
        e_cp_s.append(e_cp*1000)
        e_s.append(e*1000)

    N = 3
    ind = np.arange(N) 
    width = 0.18
    space = 0.05

    # plt.figure(1)
    
    ti_s = np.array([[t_cp_s[i], t_co_s[i], t_s[i]] for i in range(len(sce_idxes))])
    en_s = np.array([[e_cp_s[i], e_co_s[i], e_s[i]] for i in range(len(sce_idxes))])

    edgecolors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    hatches = ['xx', '\\\\', '//', 'xx']
    legends = ('BSTA', 'BDYN', 'UBSTA', 'UBDYN')
    xsticks = ['Computation', 'Communication', 'Total']
    ylabels = ['Time (s)', 'Energy Comsumption (mJ)']
    fignames = ['bar_time', 'bar_ene']
    delta_ys = [0.25, 0.001]
    ylims = [[12, 30], [0.015, 0.15]]

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
    plt.savefig(prefix_fig + 'local_round.png')
    plt.close()


if __name__=='__main__':
    dataset='mnist'
    prefix_log = f'./logs/{dataset}/'
    prefix_fig = f'./figures/{dataset}/comparison/'
    # log_file = 'system_model_tau20.0_gamma2.0_cn0.5.log'
    log_file = 'system_model_tau30_gamma10_cn1.2_optim1.log'

    plot_tien_performance(prefix_log, prefix_fig, log_file)
    plot_tien_bar(prefix_log, prefix_fig, log_file)
    plot_lround(prefix_log, prefix_fig, log_file)
    plot_feld(prefix_log, prefix_fig, log_file)