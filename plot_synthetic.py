import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider 
import os 

from main import read_options 
from parse_log import * 


def plot_fedl(log_file, prefix_figure): 
    rounds, train_acc, test_acc, train_loss, test_loss, sim = parse_fedl_w_test_acc(log_file)

    ys = [[train_acc, test_acc], [train_loss, test_loss], [sim]]
    ylabels = ["Accuracy", "Loss", "Dissimilarity"]
    fignames = ['accuracy.png', 'loss.png', 'dissimilarity.png']
    legends = [['train_acc', 'test_acc'], ['train_loss', 'test_loss'], ['dissimilarity']]
    N = len(ys)
    for t in range(N): 
        fig = plt.figure()
        yvals = ys[t]
        # print(f"len(yvals) = {len(yvals)}")
        legend = legends[t]
        plt.grid('both')
        for j in range(len(yvals)): 
            plt.plot(yvals[j], label=legend[j])
        
        plt.legend()
        plt.ylabel(ylabels[t]) 
        plt.savefig(prefix_figure+fignames[t]) # plt.savefig('plot_mnist.png')
        plt.close()

def plot_netopt(log_file, fig_file): 
    lrounds, grounds, ans, etas = parse_netopt(log_file)
    
    rounds = np.arange(0, len(lrounds), dtype=int)
    ys = [lrounds, grounds, ans, etas]
    ylabels = ["LRounds", "GRounds", "a_n", "eta"]

    N = len(ys)
    fig, ax = plt.subplots(ncols=1, nrows=N)
    for t in range(N):
        ax[t].grid('both')
        ax[t].plot(rounds, ys[t])
        ax[t].set_ylabel(ylabels[t]) 
    
    plt.savefig(fig_file) # plt.savefig('plot_mnist.png')
    # plt.show()
    plt.close()

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
    # plt.show()
    plt.close()



def plot_tien(log_file, fig_file_time, fig_file_ene): 
    t_co, t_cp, _, e_co, e_cp, _ = parse_net_tien(log_file)

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
    # plt.show()
    plt.close()

    plt.figure(2)
    plt.plot(rounds, e_co, label='temp coms')
    plt.plot(rounds, e_cp, label='temp comp')
    # plt.plot(rounds, (e_co + e_cp).cumsum(), label='accumulated')
    # plt.yscale('log')
    plt.grid(visible=True, which='both')
    plt.ylabel('Energy (J)')
    plt.legend()
    plt.savefig(fig_file_ene)
    # plt.show()
    plt.close()

def plot_user_customized(var_value, fig_file_prefix, var_name='freqs', unit='GHz', norm_factor=1e9): 
    num_users, num_grounds = var_value.shape
    rounds = np.arange(0, num_grounds)

    fig_path_dir = fig_file_prefix + var_name
    if not os.path.exists(os.path.join(fig_file_prefix, var_name)):
        os.mkdir(os.path.join(fig_file_prefix, var_name))
    
    # indivisual plots 
    for i in range(num_users): 
        fig = plt.figure(i)
        var_i = var_value[i][:]/norm_factor
        plt.plot(rounds, var_i)
        plt.title(f'user {i}')
        plt.grid(which='both')
        plt.ylabel(f'optimal {var_name} ({unit})')
        plt.savefig(f'{fig_path_dir}/user_{i}.png')
        plt.close()

    # all-user plot 
    labels = [f'user {i}' for i in range(num_users)]

    fig = plt.figure()
    for i in range(num_users): 
        var_i = var_value[i][:]/norm_factor
        plt.plot(rounds, var_i, label=labels[i])
    
    plt.legend()    
    plt.grid(which='both')
    plt.ylabel(f'optimal {var_name} ({unit})')
    
    plt.savefig(f'{fig_file_prefix}{var_name}/all-user-{var_name}.png')
    plt.close()

def plot_users(log_file, fig_file_prefix): 
    freqs, decs, powers = parse_solutions(log_file)
    plot_user_customized(var_value=freqs, fig_file_prefix=fig_file_prefix, var_name='freqs', unit='GHz', norm_factor=1e9)
    plot_user_customized(var_value=powers, fig_file_prefix=fig_file_prefix, var_name='powers', unit='dBm', norm_factor=1)
    
    t_co_s, t_cp_s, e_co_s, e_cp_s = parse_net_tien_array(log_file)
    plot_user_customized(var_value=t_co_s, fig_file_prefix=fig_file_prefix, var_name='time_co', unit='s', norm_factor=1)
    plot_user_customized(var_value=t_cp_s, fig_file_prefix=fig_file_prefix, var_name='time_cp', unit='s', norm_factor=1)
    plot_user_customized(var_value=e_co_s, fig_file_prefix=fig_file_prefix, var_name='energy_co', unit='mJ', norm_factor=1e-3)
    plot_user_customized(var_value=e_cp_s, fig_file_prefix=fig_file_prefix, var_name='energy_cp', unit='mJ', norm_factor=1e-3)
    
def test_fixedi(): 
    log_file = './logs/system_model_fixedi.log'
    fig_file = './figures/plot_synthetic_fixedi.png'
    plot_fedl(log_file, fig_file)

def test_system_model(index=4, dataset='synthetic', tau=15, log_file='system_model.log'):
    log_name=f'system_model_tau{tau}.log'
    log_name=log_file
    log_file = f'./logs/{dataset}/s{index}/{log_name}'
    prefix_figure = f'./figures/{dataset}/s{index}/' 
    if not os.path.exists(prefix_figure): 
        os.makedirs(prefix_figure)

    fig_file_fedl = prefix_figure + 'plot_synthetic_dy1.png'
    fig_file_netopt = prefix_figure + 'plot_synthetic_dy2.png'
    fig_file_gain = prefix_figure + 'channel_gains.png'
    fig_file_time = prefix_figure + 'plot_synthetic_time.png'
    fig_file_ene = prefix_figure + 'plot_synthetic_ene.png'
    plot_fedl(log_file, prefix_figure)
    plot_netopt(log_file, fig_file_netopt)
    plot_gains(log_file, fig_file_gain)
    plot_tien(log_file, fig_file_time, fig_file_ene)
    plot_users(log_file, prefix_figure)

def main(): 
    # options, _ = read_options()
    options = {
        'sce_idx': 4, 
        'dataset': 'mnist',
        'tau': 30
    }
    logfile = 'tau30_gamma100_cn0.7_vec40_optim1.log'
    test_system_model(index=options['sce_idx'], dataset=options['dataset'], tau=options['tau'], log_file=logfile)

if __name__=='__main__':
    # plot_location_act()
    # test_fixedi()
    # test_server_model()
    # test_combine()
    # plot_location_ani('./logs/location_model.log', './figures/location_ani.gif')
    main()
    # test_system_model(index=4, dataset='mnist', log_name='system_model_unoptim.log')