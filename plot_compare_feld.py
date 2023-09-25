import numpy as np 
import matplotlib.pyplot as plt
from parse_log import * 

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 18})

def decay_plot(prefix_log='./logs/', prefix_fig='./figures/comparison/', log_file = 'system_model.log'): 
    sce_idxes = [3, 4]
    prefixes = [prefix_log + f's{sce_idx}/' for sce_idx in sce_idxes]
    maccs, eps_n_s  = [], []
    loss_f = 0.1
    eps_req_s = []

    for prefix in prefixes: 
        rounds, acc, loss, sim, tloss = parse_fedl(prefix + log_file)
        eps_n, eps_req = parse_epsilon(prefix + log_file)    
        macc = (loss[1:] - loss_f)/(loss[:-1] - loss_f)
        # macc[-1] = macc[-2]
        
        maccs.append(macc)
        eps_n_s.append(eps_n)
        eps_req_s.append(eps_req)
    
    plt.figure(1)
    for t in range(len(sce_idxes)): 
        plt.plot(maccs[t], label=f' meps idx = {sce_idxes[t]}')
        plt.plot(eps_n_s[t], label=f'eps_n idx = {sce_idxes[t]}', marker='o',  markevery=10, markerfacecolor="None")
    plt.legend()
    plt.grid()
    plt.xlabel('Global Rounds')
    plt.ylabel('Measured Accuracy')
    plt.savefig(f'{prefix_fig}measured_acc.png', bbox_inches='tight')
    plt.close()

    # Calculate measured remaining accuracy
    eps_f = 1e-2
    meps_req_s = []
    for t in range(len(sce_idxes)): 
        meps_req = []
        acccumulated_acc = 1
        eps_req = eps_req_s[t]
        macc = maccs[t]

        for i in range(len(eps_req)-1):
            acccumulated_acc *= macc[i]
            meps_req.append(eps_f/acccumulated_acc)
        
        meps_req_s.append(meps_req)

    print("meps_req = ", np.array2string(np.asarray(meps_req_s[1]), separator=","))
    print("eps_req = ", np.array2string(np.asarray(eps_req_s[1]), separator=","))

    labels = ['static', 'dynamic']
    plt.figure(2)
    for t in range(len(sce_idxes)): 
        plt.plot(meps_req_s[t], label=f'meps_req-{labels[t]}')
        plt.plot(eps_req_s[t], label=f'eps_req-{labels[t]}', marker='o',  markevery=10, markerfacecolor="None")
    plt.legend()
    plt.grid()
    plt.xticks([0, 180, 216, 272])
    plt.xlabel('Global Rounds')
    plt.ylabel('Required Accuracy')
    plt.savefig(f'{prefix_fig}required_acc.png', bbox_inches='tight')
    plt.close()

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

    ystick1 = [0, 20, 40, 60, 80, 90, 100]
    ystick2 = [0, 0.11, 0.5, 1, 1.5, 2.0, 2.5]
    ystick3 = [0.35, 0.5, 1, 1.5, 2.0, 2.5]
    ysticks = [ystick1, ystick2, ystick3]
    xstick = np.arange(0, 36, 5)
    for t in range(len(ys)): 
        fig, ax = plt.subplots()
        ax.grid(which='both')
        vals = ys[t]
        for k in range(num_logs): 
            for i in range(num_sces): 
                # ax.plot(rounds[k][i][:max_round], vals[k][i][:max_round], label=legends[i])
                ax.plot(vals[k][i], label=legends[i])

            
        ax.legend()
        ax.set_ylim(ysticks[t][0], ysticks[t][-1])
        ax.set_xticks([0, 166, 196, 272, 215])
        ax.set_yticks(ysticks[t])
        ax.set_ylabel(ylabels[t])
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
    ylabels = [["Communication time (ms)", "Computation time (ms)"], ["Communication energy (mJ)", "Computation energy (mJ)"], ["Total time (ms)", "Total energy (mJ)"]]
    legends = ['BSTA', 'BDYN', 'BUSTA', 'BUDYN']

    fignames = [['time_comm.png', 'time_comp.png'], ['energy_comm.png', 'energy_comp.png'], ['total_time.png', 'total_energy.png']]

    ys = [[t_co, t_cp], [e_co, e_cp], [t, e]]

    for t in range(len(fignames)): 
        # plt.figure(t, figsize=(10, 5))
        plt.figure(1)
        # plt.subplot(121)
        for i in range(len(sce_idxes)): 
            # plt.plot(rounds, ys[t][0][i][:max_round], label=legends[i])
            plt.plot(ys[t][0][i], label=legends[i])
        plt.grid(visible=True, which='both')
        plt.ylabel(ylabels[t][0])
        plt.legend()
        plt.savefig(prefix_fig + fignames[t][0], bbox_inches='tight')
        plt.close()

        # plt.subplot(122)
        plt.figure(2)
        for i in range(len(sce_idxes)): 
            # plt.plot(rounds, ys[t][1][i][:max_round], label=legends[i])
            plt.plot(ys[t][1][i], label=legends[i])
        plt.grid(visible=True, which='both')
        plt.ylabel(ylabels[t][1])
        plt.legend()
        plt.savefig(prefix_fig + fignames[t][1], bbox_inches='tight')
        plt.close()

def get_data(prefix_log='./logs/mnist/s4/', log_file='system_model.log'):
    t_co, t_cp, t, e_co, e_cp, e = parse_net_tien(prefix_log + log_file)
    t_co = t_co.sum()/1000 # s
    t_cp = t_cp.sum()/1000 # s
    e_co = e_co.sum()/1000 # J
    e_cp = e_cp.sum()/1000 # J
    t = t.sum()/1000 # s
    e = e.sum()/1000 # J
    return t_co, t_cp, t, e_co, e_cp, e

def plot_tien_bar(prefix_log='./logs/', prefix_fig='./figures/comparison/', log_file = 'system_model.log'):
    import itertools
    # plt.rcParams.update({'font.family':'Helvetica'})

    sce_idxes = [1, 2, 3, 4]
    prefixes = [prefix_log + f's{sce_idx}/' for sce_idx in sce_idxes]
    max_rounds = [144, 190, 271, 216]
    t_co_s, t_cp_s, t_s, e_co_s, e_cp_s, e_s = [], [], [], [], [], []
    for i in range(len(sce_idxes)):
        max_round = max_rounds[i] 
        prefix = prefixes[i]
        t_co, t_cp, t, e_co, e_cp, e = parse_net_tien(prefix + log_file)
        t_co = t_co[:max_round].sum()/1000 # s 
        t_cp = t_cp[:max_round].sum()/1000 # s [:max_round]
        e_co = e_co[:max_round].sum()/1000 # J
        e_cp = e_cp[:max_round].sum()/1000 # J
        t = t.sum()/1000 # s
        e = e.sum()/1000 # J

        # t_co, t_cp, t, e_co, e_cp, e = get_data(prefix_log=prefix, log_file=log_file)
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
    # ylims = [[12, 30], [0.015, 0.15]]

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
            ax.text(rect.get_x() + rect.get_width() / 2, height + delta_ys[t], label, ha="center", va="bottom", fontsize=15)

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
        # plt.plot(rounds, lrounds[i][:max_round], label=legends[i])
        plt.plot(lrounds[i], label=legends[i])
    plt.ylabel("# Local Rounds")
    plt.grid(which='both')
    plt.legend()
    plt.savefig(prefix_fig + 'local_round.png')
    plt.close()

def test_decay(): 
    sce_idx=4
    tau=40
    dataset='mnist' 
    model='mclr' 
    learning_rate=0.0017
    optim=1
    gamma=100
    C_n=0.7 # 0.2 
    xi_factor=1
    velocity=40
    
    options = {
        'sce_idx': sce_idx, 
        'dataset': dataset,
        'tau': tau
    }
    log_file=f'tau{tau}_gamma{gamma}_cn{C_n}_vec{velocity}_optim{optim}_check.log'
    prefix_log = f'./logs/{dataset}/'
    prefix_fig = f'./figures/{dataset}/comparison/'
    decay_plot(prefix_log, prefix_fig, log_file)

def test(): 
    sce_idx = 4
    tau = 40
    dataset='mnist' 
    model='mclr' 
    learning_rate=0.0017
    optim=1
    gamma=100
    C_n=0.7 # 0.2 
    xi_factor=1
    velocity=40
    
    options = {
        'sce_idx': sce_idx, 
        'dataset': dataset,
        'tau': tau
    }
    prefix_log = f'./logs/{dataset}/s4/'
    prefix_fig = f'./figures/{dataset}/comparison/'
    lrounds = [1, 10, 20]

    files=[f'tau{tau}_gamma{gamma}_cn{C_n}_vec{velocity}_optim{optim}_lround{lround}.log' for lround in lrounds]
    rounds, acc, loss, sim, tloss = [], [], [], [], []
    for file in files: 
        rounds_tmp, acc_tmp, loss_tmp, sim_tmp, tloss_tmp = parse_fedl(prefix_log + file)
        rounds.append(rounds_tmp)
        acc.append(acc_tmp)
        loss.append(loss_tmp)
        sim.append(sim_tmp)
        tloss.append(tloss_tmp)
    
    num_logs = len(files)

    ylabels = ["Testing Accuracy", "Training Loss", "Testing Loss"]
    legends = [f'lround {t}' for t in lrounds]
    fignames = ['test_acc', 'train_loss', 'test_loss']
    ys = [acc, loss, tloss]

    ystick1 = [0, 20, 40, 60, 80, 90, 100]
    ystick2 = [0, 0.1, 0.5, 1, 1.5, 2.0, 2.5]
    ystick3 = [0.35, 0.5, 1, 1.5, 2.0, 2.5]
    ysticks = [ystick1, ystick2, ystick3]
    xstick = np.arange(0, 36, 5)
    for t in range(len(ys)): 
        fig, ax = plt.subplots()
        ax.grid(which='both')
        vals = ys[t]
        for i in range(num_logs): 
            ax.plot(vals[i], label=legends[i])
            
        ax.legend()
        # ax.set_xlim(0, 500)
        # ax.set_ylim(ysticks[t][0], ysticks[t][-1])
        ax.set_yticks(ysticks[t])
        ax.set_ylabel(ylabels[t])
        ax.set_xlabel('Global Rounds')
        plt.savefig(f'{prefix_fig}{fignames[t]}.eps', bbox_inches='tight')
        plt.savefig(f'{prefix_fig}{fignames[t]}.png', bbox_inches='tight')
        plt.close()

def main(): 
    sce_idx=4
    tau=40
    dataset='mnist' 
    model='mclr' 
    learning_rate=0.0017
    optim=1
    gamma=100
    C_n=0.7 # 0.2 
    xi_factor=1
    velocity=40
    
    options = {
        'sce_idx': sce_idx, 
        'dataset': dataset,
        'tau': tau
    }
    log_file=f'tau{tau}_gamma{gamma}_cn{C_n}_vec{velocity}_optim{optim}_check.log'
    prefix_log = f'./logs/{dataset}/'
    prefix_fig = f'./figures/{dataset}/comparison/'
    plot_tien_performance(prefix_log, prefix_fig, log_file)
    plot_tien_bar(prefix_log, prefix_fig, log_file)
    plot_lround(prefix_log, prefix_fig, log_file)
    plot_feld(prefix_log, prefix_fig, log_file)

if __name__ == '__main__': 
    # test()
    main()
    # test_decay()