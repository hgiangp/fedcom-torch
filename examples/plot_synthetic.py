import re 
import matplotlib.pyplot as plt
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
        search_time = re.search(r'At round (.*) t_co: (.*) t_cp: (.*)', line, re.M|re.I)
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
        search_uav = re.search(r'uav_gains_db_mean: (.*)$', line, re.M|re.I)
        if search_uav: 
            uav_gains.append(float(search_uav.group(1)))
        
        search_bs = re.search(r'bs_gains_db_mean: (.*)$', line, re.M|re.I)
        if search_bs: 
            bs_gains.append(float(search_bs.group(1)))
    
    return uav_gains, bs_gains

def parse_location(file_name): 
    xs, ys = [], []
    for line in open(file_name, 'r'): 
        search_xs = re.search(r'xs mean: (.*)$', line, re.M|re.I)
        if search_xs: 
            xs.append(float(search_xs.group(1)))
        
        search_ys = re.search(r'ys mean: (.*)$', line, re.M|re.I)
        if search_ys: 
            ys.append(float(search_ys.group(1)))
    
    return xs, ys  

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

def plot_location(): 
    loc_x, loc_y = parse_location('./logs/system_model.log')
    rounds = np.arange(0, len(loc_x))

    plt.figure(1)
    # plt.plot(loc_x, loc_y, label='(x, y)')
    plt.plot(rounds, loc_x, label='Loc x')
    plt.plot(rounds, loc_y, label='Loc y')
    plt.grid(visible=True, which='both')
    plt.legend()

    plt.savefig('./figures/locations.png')
    plt.show()

def plot_tien(log_file, fig_file_time, fig_file_ene): 
    t_co, t_cp, e_co, e_cp = parse_net_tien(log_file)

    t_co = np.asarray(t_co)
    t_cp = np.asarray(t_cp)
    e_co = np.asarray(e_co)
    e_cp = np.asarray(e_cp)
    rounds = np.arange(0, len(t_co))
    
    # time, energy in each grounds  
    plt.figure(1)
    plt.plot(rounds, t_co, label='temp coms')
    plt.plot(rounds, t_cp, label='temp comp')
    plt.plot(rounds, (t_co + t_cp).cumsum(), label='accumulated')
    plt.grid(visible=True, which='both')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.savefig(fig_file_time)
    plt.show()

    plt.figure(2)
    plt.plot(rounds, e_co, label='temp coms')
    plt.plot(rounds, e_cp, label='temp comp')
    plt.plot(rounds, (e_co + e_cp).cumsum(), label='accumulated')
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
    # plot_fedl(log_file, fig_file_fedl)
    # plot_netopt(log_file, fig_file_netopt)
    # plot_gains()
    # plot_location()
    fig_file_time = './figures/plot_synthetic_time.png'
    fig_file_ene = './figures/plot_synthetic_ene.png'
    plot_tien(log_file, fig_file_time, fig_file_ene)

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
    # test_fixedi()
    # test_server_model()
    # test_combine()
