import re 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider 
import numpy as np 

def parse_fedl(file_name): 
    rounds, acc, loss, sim = [], [], [], []
    test_loss = []
    
    for line in open(file_name, 'r'):
        search_train_acc = re.search(r'At round (.*) training accuracy: (.*)', line, re.M|re.I)
        if search_train_acc: 
            rounds.append(int(search_train_acc.group(1)))
        else: 
            search_test_acc = re.search(r'At round (.*) accuracy: (.*)', line, re.M|re.I)
            if search_test_acc: 
                acc.append(float(search_test_acc.group(2)))
        
        search_loss = re.search(r'At round (.*) training loss: (.*)', line, re.M|re.I)
        if search_loss: 
            loss.append(float(search_loss.group(2)))

        search_test_loss = re.search(r'At round (.*) test loss: (.*)', line, re.M|re.I)
        if search_test_loss: 
            test_loss.append(float(search_test_loss.group(2)))
        
        search_grad = re.search(r'gradient difference: (.*)', line, re.M|re.I)
        if search_grad: 
            sim.append(float(search_grad.group(1)))
    
    rounds = np.asarray(rounds)
    acc = np.asarray(acc) * 100
    loss = np.asarray(loss)
    sim = np.asarray(sim)
    test_loss = np.asarray(test_loss)
        
    return rounds, acc, loss, sim, test_loss

def parse_netopt(file_name):
    lrounds, grounds, ans, etas = [], [], [], []

    for line in open(file_name, 'r'):
        search_lround = re.search(r'At round (.*) local rounds: (.*)', line, re.M|re.I)
        if search_lround: 
            lrounds.append(float(search_lround.group(2)))
        
        search_ground = re.search(r'At round (.*) global rounds: (.*)', line, re.M|re.I)
        if search_ground: 
            grounds.append(float(search_ground.group(2)))    

        search_an = re.search(r'At round (.*) a_n: (.*)', line, re.M|re.I)
        if search_an: 
            ans.append(float(search_an.group(2)))

        search_eta = re.search(r'At round (.*) eta: (.*)', line, re.M|re.I)
        if search_eta: 
            etas.append(float(search_eta.group(2)))
    
    return lrounds, grounds, ans, etas

def parse_net_tien(file_name): 
    t_co, t_cp, e_co, e_cp = [], [], [], []

    for line in open(file_name, 'r'):
        search_time = re.search(r'At round (.*) average t_co: (.*) average t_cp: (.*)', line, re.M|re.I)
        if search_time: 
            t_co.append(float(search_time.group(2)))
            t_cp.append(float(search_time.group(3)))  

        search_ene = re.search(r'At round (.*) average e_co: (.*) average e_cp: (.*)', line, re.M|re.I)
        if search_ene: 
            e_co.append(float(search_ene.group(2)))
            e_cp.append(float(search_ene.group(3)))

    t_co = np.asarray(t_co)
    t_cp = np.asarray(t_cp)
    e_co = np.asarray(e_co)
    e_cp = np.asarray(e_cp) 
    return t_co, t_cp, e_co, e_cp

def parse_gains(file_name): 
    uav_gains, bs_gains = [], []
    for line in open(file_name, 'r'): 
        search_uav = re.search(r'uav_gains = \[(.*)\]$', line, re.M|re.I)
        if search_uav:
            gain = np.fromstring(search_uav.group(1), sep=' ')
            gain_db_mean = 10 * np.log10(gain).mean()
            uav_gains.append(gain_db_mean)
        
        search_bs = re.search(r'bs_gains = \[(.*)\]$', line, re.M|re.I)
        if search_bs: 
            gain = np.fromstring(search_bs.group(1), sep=' ')
            gain_db_mean = 10 * np.log10(gain).mean()
            bs_gains.append(gain_db_mean)  
    
    return uav_gains, bs_gains

def parse_location(file_name): 
    xs, ys = [], []

    for line in open(file_name, 'r'): 
        search_xs = re.search(r'xs = \[(.*)\]', line, re.M|re.I)
        if search_xs: 
            x = np.fromstring(search_xs.group(1), sep=' ')
            xs.append(x)
        
        search_ys = re.search(r'ys = \[(.*)\]', line, re.M|re.I)
        if search_ys: 
            y = np.fromstring(search_ys.group(1), sep=' ')
            ys.append(y)
    

    # convert to numpy array 
    xs_np = np.array(xs) # (num_grounds, num_users)
    ys_np = np.array(ys) # (num_grounds, num_users)
    return xs_np, ys_np 

def plot_fedl(log_file, fig_file): 
    rounds, acc, loss, sim, test_loss = parse_fedl(log_file)

    rounds = np.asarray(rounds)
    acc = np.asarray(acc) * 100
    loss = np.asarray(loss)
    sim = np.asarray(sim)
    test_loss = np.asarray(test_loss)

    plt.figure(1)
    plt.subplot(411)
    plt.plot(rounds, acc)
    plt.ylabel("Train Accuracy")
    plt.grid(which='both')

    plt.subplot(412)
    plt.plot(rounds, loss)
    plt.ylabel("Train Loss")
    plt.grid(which='both')

    plt.subplot(413)
    plt.plot(rounds, test_loss)
    plt.ylabel("Test Loss")
    plt.grid(which='both')
        
    plt.subplot(414)
    plt.plot(rounds, sim)
    plt.ylabel("Dissimilarity")

    plt.grid(which='both')
    plt.savefig(fig_file) # plt.savefig('plot_mnist.png')
    plt.show()

def plot_netopt(log_file, fig_file): 
    lrounds, grounds, ans, etas = parse_netopt(log_file)
    
    lrounds = np.asarray(lrounds)
    grounds = np.asarray(grounds)
    ans = np.asarray(ans)
    etas = np.asarray(etas)

    rounds = np.arange(0, len(lrounds), dtype=int)

    plt.figure(1)
    plt.subplot(411)
    plt.plot(rounds, lrounds)
    plt.grid(which='both')
    plt.ylabel("LRounds")

    plt.subplot(412)
    plt.plot(rounds, grounds)
    plt.grid(which='both')
    plt.ylabel("GRounds")
    
    plt.subplot(413)
    plt.plot(rounds, ans)
    plt.grid(which='both')
    plt.ylabel("a_n")

    plt.subplot(414)
    plt.plot(rounds, etas)
    plt.grid(which='both')
    plt.ylabel("eta")

    plt.savefig(fig_file)
    plt.show()

def plot_gains(log_file, fig_file): 
    uav_gains, bs_gains = parse_gains(file_name=log_file)
    rounds = np.arange(0, len(uav_gains))
    
    plt.figure(1)
    plt.plot(rounds, uav_gains, label='UAV Gains')
    plt.plot(rounds, bs_gains, label='BS Gains')
    plt.grid(visible=True, which='both')
    plt.legend()
    plt.xlabel('Global Rounds')
    plt.ylabel('Channel Gains (dB)')
    plt.savefig(fig_file)
    plt.show()

def plot_location_act(): 
    r'''Create interactive plot'''
    xs, ys = parse_location('./logs/system_model.log') # (num_grounds, num_users)
    num_grounds, num_users = xs.shape

    fig = plt.figure(figsize=(6, 4))

    ax = fig.add_subplot(111) 
    fig.subplots_adjust(top=0.85) # 0.25 top to place sliders 

    # Create axes for sliders 
    ax_rounds = fig.add_axes([0.3, 0.92, 0.4, 0.05])
    ax_rounds.spines['top'].set_visible(True)
    ax_rounds.spines['right'].set_visible(True)

    # Create sliders 
    s_rounds = Slider(ax=ax_rounds, label='GRound', valmin=0, valmax=num_grounds-1, valfmt='%0.0f', facecolor='#cc7000')

    colors = plt.get_cmap('viridis', num_users)(np.linspace(0.2, 0.7, num_users))
    # labels = [f'user {i}' for i in range(num_users)]
    
    # Plot default data
    sc = ax.scatter(xs[0], ys[0], c=colors, alpha=0.5)
    ax.set_xlim(-1200, 1200)
    ax.set_ylim(-2000, 1200)
    ax.grid(True, 'both')

    # Update values 
    def update(round):
        sc.set_offsets(np.c_[xs[int(round)], ys[int(round)]])
        fig.canvas.draw_idle()

    s_rounds.on_changed(update)
    # plt.savefig('./figures/locations.png')
    plt.show()

def plot_location_ani(log_file, fig_file): 
    xs, ys = parse_location(log_file) # (num_grounds, num_users)
    num_grounds, num_users = xs.shape

    colors = plt.get_cmap('viridis', num_users)(np.linspace(0.2, 0.7, num_users))
    # labels = [f'user {i}' for i in range(num_users)]

    fig, ax = plt.subplots()

    uav_x, uav_y = 0, 0
    bs_x, bs_y = -400, 400
    
    ax.scatter([uav_x], [uav_y], marker="*", s=100, alpha=0.7, c='red')
    ax.annotate('UAV', (uav_x+10, uav_y+20))
    ax.scatter([bs_x], [bs_y], marker="p", s=70, alpha=0.7, c='green')
    ax.annotate('BS', (bs_x-15, bs_y+30))

    plt.xlim(-700,700)
    plt.ylim(-1000,500)
    plt.grid(True, 'both')
    
    sc = ax.scatter(xs[0], ys[0], c=colors, alpha=0.5)

    def animate(i):
        sc.set_offsets(np.c_[xs[i], ys[i]])
    ani = FuncAnimation(fig, animate, frames=num_grounds, interval=50, repeat=False) 
    
    # Ensure the entire plot is visible 
    fig.tight_layout()

    # Save and show animation
    ani.save(fig_file, writer='imagemagick', fps=24)

def plot_tien(log_file, fig_file_time, fig_file_ene): 
    t_co, t_cp, e_co, e_cp = parse_net_tien(log_file)

    round_max = 185
    t_co = np.asarray(t_co)[:round_max]
    t_cp = np.asarray(t_cp)[:round_max]
    e_co = np.asarray(e_co)[:round_max]
    e_cp = np.asarray(e_cp)[:round_max]
    rounds = np.arange(0, len(t_co))[:round_max]
    
    # time, energy in each grounds  
    plt.figure(1)
    plt.plot(rounds, t_co, label='temp coms')
    plt.plot(rounds, t_cp, label='temp comp')
    # plt.plot(rounds, (t_co + t_cp).cumsum(), label='accumulated')
    # plt.yscale('log')
    plt.grid(visible=True, which='both')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.savefig(fig_file_time)
    plt.show()

    plt.figure(2)
    plt.plot(rounds, e_co, label='temp coms')
    plt.plot(rounds, e_cp, label='temp comp')
    # plt.plot(rounds, (e_co + e_cp).cumsum(), label='accumulated')
    # plt.yscale('log')
    plt.grid(visible=True, which='both')
    plt.ylabel('Energy (J)')
    plt.legend()
    plt.savefig(fig_file_ene)
    plt.show()

def test_fixedi(): 
    log_file = './logs/system_model_fixedi.log'
    fig_file = './figures/plot_synthetic_fixedi.png'
    plot_fedl(log_file, fig_file)

def test_system_model(index=4):
    log_file = f'./logs/s{index}/system_model.log'
    prefix_figure = f'./figures/s{index}/' 
    fig_file_fedl = prefix_figure + 'plot_synthetic_dy1.png'
    fig_file_netopt = prefix_figure + 'plot_synthetic_dy2.png'
    fig_file_gain = prefix_figure + 'channel_gains.png'
    fig_file_time = prefix_figure + 'plot_synthetic_time.png'
    fig_file_ene = prefix_figure + 'plot_synthetic_ene.png'
    fig_file_ani = prefix_figure + 'location_ani.gif'
    plot_fedl(log_file, fig_file_fedl)
    plot_netopt(log_file, fig_file_netopt)
    plot_gains(log_file, fig_file_gain)
    plot_tien(log_file, fig_file_time, fig_file_ene)
    plot_location_ani(log_file, fig_file_ani)

def test_server_model(): 
    log_file, fig_file = './logs/server_model.log', './figures/plot_synthetic.png'
    plot_fedl(log_file, fig_file)

def test_combine(): 
    rounds_sys, acc_sys, loss_sys, sim_sys, tloss_sys = parse_fedl('./logs/system_model.log')
    rounds_ser, acc_ser, loss_ser, sim_ser, tloss_ser = parse_fedl('./logs/server_model.log')

    rounds_sys = np.asarray(rounds_sys)
    acc_sys = np.asarray(acc_sys) * 100
    loss_sys = np.asarray(loss_sys)
    sim_sys = np.asarray(sim_sys)

    rounds_ser = np.asarray(rounds_ser)
    acc_ser = np.asarray(acc_ser) * 100
    loss_ser = np.asarray(loss_ser)
    sim_ser = np.asarray(sim_ser)

    max_round = min(len(rounds_sys), len(rounds_ser))

    plt.figure(1)
    plt.subplot(411)
    plt.plot(rounds_sys[:max_round], acc_sys[:max_round], label='system')
    plt.plot(rounds_ser[:max_round], acc_ser[:max_round], label='server')
    plt.ylabel("Train Accuracy")
    plt.grid(which='both')
    plt.legend()

    plt.subplot(412)
    plt.plot(rounds_sys[:max_round], loss_sys[:max_round], label='system')
    plt.plot(rounds_ser[:max_round], loss_ser[:max_round], label='server')
    plt.ylabel("Train Loss")
    plt.grid(which='both')
    # plt.legend()

    plt.subplot(413)
    plt.plot(rounds_sys[:max_round], tloss_sys[:max_round], label='system')
    plt.plot(rounds_ser[:max_round], tloss_ser[:max_round], label='server')
    plt.ylabel("Test Loss")
    plt.grid(which='both')
        
    plt.subplot(414)
    plt.plot(rounds_sys[:max_round], sim_sys[:max_round], label='system')
    plt.plot(rounds_ser[:max_round], sim_ser[:max_round], label='server')
    plt.ylabel("Dissimilarity")
    plt.grid(which='both')
    # plt.legend()

    plt.savefig('./figures/plot_synthetic_fedl.png') 
    plt.show()

def plot_feld_performance():
    # idxes = np.arange(1, 5, 1)
    prefix_log = './logs/'
    log_file = 'system_model.log'
    prefix_fig = './figures/comparison/'

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
    plt.show()

def plot_tien_performance(): 
    prefix_log = './logs/'
    log_file = 'system_model.log'
    prefix_fig = './figures/comparison/'
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
    plt.show()

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
    plt.show()

def plot_tien_bar():
    import itertools
    plt.rcParams.update({'font.family':'Helvetica'})

    prefix_log = './logs/'
    log_file = 'system_model.log'
    prefix_fig = './figures/comparison/'
    fig_file_time = 'synthetic_time.png'
    fig_file_ene = 'synthetic_ene.png'

    t_co_1, t_cp_1, e_co_1, e_cp_1 = parse_net_tien(prefix_log + 's1/' + log_file)
    t_co_2, t_cp_2, e_co_2, e_cp_2 = parse_net_tien(prefix_log + 's2/' + log_file)
    t_co_3, t_cp_3, e_co_3, e_cp_3 = parse_net_tien(prefix_log + 's3/' + log_file)
    t_co_4, t_cp_4, e_co_4, e_cp_4 = parse_net_tien(prefix_log + 's4/' + log_file)

    max_round = min(len(t_co_1), len(t_co_2), len(t_co_3), len(t_co_4))
    
    t_co_1 = t_co_1[:max_round].sum()
    t_cp_1 = t_cp_1[:max_round].sum()
    e_co_1 = e_co_1[:max_round].sum()
    e_cp_1 = e_cp_1[:max_round].sum()

    t_co_2 = t_co_2[:max_round].sum()
    t_cp_2 = t_cp_2[:max_round].sum()
    e_co_2 = e_co_2[:max_round].sum()
    e_cp_2 = e_cp_2[:max_round].sum()

    t_co_3 = t_co_3[:max_round].sum()
    t_cp_3 = t_cp_3[:max_round].sum()
    e_co_3 = e_co_3[:max_round].sum()
    e_cp_3 = e_cp_3[:max_round].sum()

    t_co_4 = t_co_4[:max_round].sum()
    t_cp_4 = t_cp_4[:max_round].sum()
    e_co_4 = e_co_4[:max_round].sum()
    e_cp_4 = e_cp_4[:max_round].sum()
    
    N = 3
    ind = np.arange(N) 
    width = 0.18
    space = 0.03

    plt.figure(1)
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
    ax.set_ylim(0, 55)
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

    plt.savefig(prefix_fig + 'synthetic_time_bar.png', bbox_inches='tight')
    plt.show()

    plt.figure(2)
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
    ax.set_ylim(0, 1.4)
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
            rect.get_x() + rect.get_width() / 2, height + 0.03, label, ha="center", va="bottom"
        )

    plt.savefig(prefix_fig + 'synthetic_ene_bar.png', bbox_inches='tight')
    plt.show()

if __name__=='__main__':
    # idx_sce = 3 
    # test_system_model(index=idx_sce)
    # plot_location_act()
    # test_fixedi()
    # test_server_model()
    # test_combine()
    # plot_location_ani()
    # plot_feld_performance()
    plot_tien_performance()
    # plot_tien_bar()
