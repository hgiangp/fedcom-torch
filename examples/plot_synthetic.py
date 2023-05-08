import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider 

from system_utils import read_options 
from parse_log import * 
from network_params import x_uav, y_uav, x_bs, y_bs
from network_utils import calc_bs_gains, calc_uav_gains

def plot_fedl(log_file, fig_file): 
    rounds, acc, loss, sim, test_loss = parse_fedl(log_file)

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
    # plt.show()
    plt.close()

def plot_netopt(log_file, fig_file): 
    lrounds, grounds, ans, etas = parse_netopt(log_file)
    
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
    plt.close()

def plot_location_ani(log_file, fig_file): 
    xs, ys = parse_location(log_file) # (num_grounds, num_users)
    num_grounds, num_users = xs.shape
    # labels = [f'user {i}' for i in range(num_users)]

    # plot maps 
    fig, ax = plot_maps()

    # plot users
    colors = plt.get_cmap('viridis', num_users)(np.linspace(0.2, 0.7, num_users))
    sc = ax[0, 0].scatter(xs[0], ys[0], c=colors, alpha=0.5)

    def animate(i):
        fig.suptitle(f'round = {i}')
        sc.set_offsets(np.c_[xs[i], ys[i]])
    ani = FuncAnimation(fig, animate, frames=num_grounds, interval=50, repeat=False)

    # Save and show animation
    ani.save(fig_file, writer='imagemagick', fps=24)

def plot_maps(fig_size=(8, 8), nrows=1, ncols=1, title=''): 
    # plot map 
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=fig_size, squeeze=False)
    # fig, ax = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False)

    # plot road map 
    x11 = [-410, 500]; x12 = [500, 44] 
    x21 = [-500, 500]; x22 = [500, 0]
    x31 = [-500, -500]; x32 = [500, 500]
    x41 = [-444, -500]; x42 = [500, 444]
    xmid11 = [500, 22]; xmid12 = [-455, 500]
    xmid21 = [-472, -500]; xmid22 = [500, 472]

    # plot uav, bs 
    uav_x, uav_y = x_uav, y_uav
    bs_x, bs_y = x_bs, y_bs
    for i in range(nrows):
        for j in range(ncols): 
            ax[i][j].plot([x11[0], x12[0]], [x11[1], x12[1]], color='blue')
            ax[i][j].plot([x21[0], x22[0]], [x21[1], x22[1]], color='blue')
            ax[i][j].plot([x31[0], x32[0]], [x31[1], x32[1]], color='blue')
            ax[i][j].plot([x41[0], x42[0]], [x41[1], x42[1]], color='blue')
            
            ax[i][j].plot([xmid11[0], xmid12[0]], [xmid11[1], xmid12[1]], color='orange', linestyle='--')   
            ax[i][j].plot([xmid21[0], xmid22[0]], [xmid21[1], xmid22[1]], color='orange', linestyle='--')

            ax[i][j].scatter([uav_x], [uav_y], marker="*", s=100, alpha=0.7, c='red')
            ax[i][j].annotate('UAV', (uav_x+10, uav_y+20))
            ax[i][j].scatter([bs_x], [bs_y], marker="p", s=70, alpha=0.7, c='green')
            ax[i][j].annotate('BS', (bs_x-15, bs_y+30))

            ax[i][j].set_xlim(-500, 500)
            ax[i][j].set_ylim(-500, 500)
            ax[i][j].grid(which='both')
    
    fig.suptitle(title)
    # Ensure the entire plot is visible
    # fig.tight_layout()
    return fig, ax 

def test_plot_maps(nrows=1, ncols=1):
    fig, ax = plot_maps(nrows=nrows, ncols=ncols)
    plt.savefig("./figures/maps.png")

def plot_gain_density(): 
    # maps 
    x = np.linspace(-500, 500, 1000)
    y = np.linspace(-500, 500, 1000)
    X, Y = np.meshgrid(x, y)
    
    # calculate gains  
    uav_gains = calc_uav_gains(X, Y)
    bs_gains = calc_bs_gains(X, Y)
    
    uav_gains_db = 10 * np.log10(uav_gains)
    bs_gains_db = 10 * np.log10(bs_gains)

    # plot maps 
    fig, ax = plot_maps(fig_size=(11, 5), nrows=1, ncols=2, title='Gains (dB)')
    
    # calculate max, min of color level 
    cmin = min(uav_gains_db.min(), bs_gains_db.min())
    cmax = max(uav_gains_db.max(), bs_gains_db.max())
    color_levels = np.linspace(start=cmin, stop=cmax, num=20)
    color_map = 'coolwarm'

    # plot data 
    color = ax[0, 0].contourf(X, Y, uav_gains_db, cmap=color_map, levels=color_levels)
    ax[0, 1].contourf(X, Y, bs_gains_db, cmap=color_map, levels=color_levels)

    # adding color bar
    cax = fig.add_axes([0.92, 0.11, 0.02, 0.77]) # rect([left, bottom, width, height])
    cbar = fig.colorbar(color, cax=cax, orientation='vertical')

    # customize the colorbar
    # cbar.ax.set_ylabel('Gain (dB)', fontweight='bold')
    # cbar.ax.set_yticks(ticks=[-0.70, -0.35, 0.00, 0.35, 0.7])
    # cbar.ax.set_yticklabels([-0.70, -0.35, 0.00, 0.35, 0.7], rotation='vertical', va='center')

    plt.savefig('./figures/gain_density.png')
    plt.close()

def plot_SNR(): 
    # maps 
    x = np.linspace(-500, 500, 1000)
    y = np.linspace(-500, 500, 1000)
    X, Y = np.meshgrid(x, y)
    
    # calculate gains  
    uav_gains = calc_uav_gains(X, Y)
    bs_gains = calc_bs_gains(X, Y)
    
    uav_gains_db = 10 * np.log10(uav_gains)
    bs_gains_db = 10 * np.log10(bs_gains)

    # calculate snr 
    noise_dBm = -114 # -174 + 60 
    p_t_dBm = 20 # max power 

    snr_uav_db = p_t_dBm + uav_gains_db - noise_dBm 
    snr_bs_db  = p_t_dBm + bs_gains_db - noise_dBm 

    # plot maps 
    fig, ax = plot_maps(fig_size=(11, 5), nrows=1, ncols=2, title='SNR (dB)')

    # calculate max, min of color level 
    cmin = min(snr_uav_db.min(), snr_bs_db.min())
    cmax = max(snr_uav_db.max(), snr_bs_db.max())

    # color plot setting 
    color_levels = np.linspace(start=cmin, stop=cmax, num=20)
    color_map = 'coolwarm'

    # plot data 
    color = ax[0, 0].contourf(X, Y, snr_uav_db, cmap=color_map, levels=color_levels)
    ax[0, 1].contourf(X, Y, snr_bs_db, cmap=color_map, levels=color_levels)

    # adding color bar
    cax = fig.add_axes([0.92, 0.11, 0.02, 0.77]) # rect([left, bottom, width, height])
    cbar = fig.colorbar(color, cax=cax, orientation='vertical')

    # customize the colorbar
    # cbar.ax.set_ylabel('Gain (dB)', fontweight='bold')
    # cbar.ax.set_yticks(ticks=[-0.70, -0.35, 0.00, 0.35, 0.7])
    # cbar.ax.set_yticklabels([-0.70, -0.35, 0.00, 0.35, 0.7], rotation='vertical', va='center')

    plt.savefig('./figures/snr.png')
    plt.close()

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
    for i in range(num_users): 
        fig = plt.figure(i)
        var_i = var_value[i][:]/norm_factor
        plt.plot(rounds, var_i)
        plt.title(f'user {i}')
        plt.grid(which='both')
        plt.ylabel(f'optimal {var_name} ({unit})')
        plt.savefig(f'{fig_file_prefix}{var_name}/user_{i}.png')
        plt.close()

def plot_users(log_file, fig_file_prefix): 
    freqs, decs, powers = parse_solutions(log_file)
    # plot_user_customized(var_value=freqs, fig_file_prefix=fig_file_prefix, var_name='freqs', unit='GHz', norm_factor=1e9)
    plot_user_customized(var_value=powers, fig_file_prefix=fig_file_prefix, var_name='powers', unit='dBm', norm_factor=1)
    
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
    # plot_fedl(log_file, fig_file_fedl)
    # plot_netopt(log_file, fig_file_netopt)
    # plot_gains(log_file, fig_file_gain)
    # plot_tien(log_file, fig_file_time, fig_file_ene)
    # plot_location_ani(log_file, fig_file_ani)
    plot_users(log_file, prefix_figure)

def test_server_model(): 
    log_file, fig_file = './logs/server_model.log', './figures/plot_synthetic.png'
    plot_fedl(log_file, fig_file)

def test_combine(): 
    rounds_sys, acc_sys, loss_sys, sim_sys, tloss_sys = parse_fedl('./logs/system_model.log')
    rounds_ser, acc_ser, loss_ser, sim_ser, tloss_ser = parse_fedl('./logs/server_model.log')

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
    # plt.show()
    plt.close()

def main(): 
    sce_idx = read_options()['sce_idx']
    test_system_model(index=sce_idx)

if __name__=='__main__':
    # plot_location_act()
    # test_fixedi()
    # test_server_model()
    # test_combine()
    # plot_location_ani('./logs/location_model.log', './figures/location_ani.gif')
    # plot_gain_density()
    # test_plot_maps(1, 1)
    # plot_SNR()
    main()