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

        search_ene = re.search(r'At round (.*) e_co: (.*) e_cp: (.*)', line, re.M|re.I)
        if search_ene: 
            e_co.append(float(search_ene.group(2)))
            e_cp.append(float(search_ene.group(3)))    
    
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

def plot_gains(): 
    uav_gains, bs_gains = parse_gains('./logs/system_model.log')
    rounds = np.arange(0, len(uav_gains))
    
    plt.figure(1)
    plt.plot(rounds, uav_gains, label='UAV Gains')
    plt.plot(rounds, bs_gains, label='BS Gains')
    plt.grid(visible=True, which='both')
    plt.legend()
    plt.xlabel('Global Rounds')
    plt.ylabel('Channel Gains (dB)')
    plt.savefig('./figures/channel_gains.png')
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

def plot_location_ani(): 
    xs, ys = parse_location('./logs/system_model.log') # (num_grounds, num_users)
    num_grounds, num_users = xs.shape

    colors = plt.get_cmap('viridis', num_users)(np.linspace(0.2, 0.7, num_users))
    # labels = [f'user {i}' for i in range(num_users)]

    fig, ax = plt.subplots()
    sc = ax.scatter(xs[0], ys[0], c=colors, alpha=0.5)
    plt.xlim(-1200,1200)
    plt.ylim(-2000,1200)
    plt.grid(True, 'both')

    def animate(i):
        sc.set_offsets(np.c_[xs[i], ys[i]])
    ani = FuncAnimation(fig, animate, frames=num_grounds, interval=50, repeat=False) 
    
    # Ensure the entire plot is visible 
    fig.tight_layout()

    # Save and show animation
    ani.save('./figures/location_ani.gif', writer='imagemagick', fps=24)

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

def test_system_model(): 
    log_file = './logs/system_model.log'
    fig_file_fedl = './figures/plot_synthetic_dy1.png'
    fig_file_netopt = './figures/plot_synthetic_dy2.png'
    plot_fedl(log_file, fig_file_fedl)
    plot_netopt(log_file, fig_file_netopt)
    plot_gains()
    fig_file_time = './figures/plot_synthetic_time.png'
    fig_file_ene = './figures/plot_synthetic_ene.png'
    plot_tien(log_file, fig_file_time, fig_file_ene)
    plot_location_ani()

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

if __name__=='__main__': 
    test_system_model()
    # plot_location_act()
    # test_fixedi()
    # test_server_model()
    # test_combine()
