import os
import numpy as np 
import matplotlib.pyplot as plt
from parse_log import * 
from matplotlib.widgets import Slider 
from matplotlib.animation import FuncAnimation
from src.network_params import x_uav, y_uav, x_bs, y_bs
from src.network_utils import calc_bs_gains, calc_uav_gains
import matplotlib.colors as mcolors

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
    # colors = plt.get_cmap('viridis', num_users)(np.linspace(0.2, 0.7, num_users))
    colors = list(mcolors.TABLEAU_COLORS.keys())[:num_users]
    print(colors, type(colors))
    
    sc = ax[0, 0].scatter(xs[0], ys[0], c=colors, alpha=1)

    def animate(i):
        fig.suptitle(f'round = {i}')
        sc.set_offsets(np.c_[xs[i], ys[i]])
    ani = FuncAnimation(fig, animate, frames=num_grounds, interval=50, repeat=False)

    # Save and show animation
    ani.save(fig_file, writer='imagemagick', fps=20)
    plt.close()

def plot_location_ani_all(log_file, prefix_figure): 
    xs, ys = parse_location(log_file) # (num_grounds, num_users)
    num_grounds, num_users = xs.shape
    colors = list(mcolors.TABLEAU_COLORS.keys())[:num_users]
    
    print(colors, type(colors))
    for user_idx in range(num_users):
        fig, ax = plot_maps(nrows=1, ncols=1)
        sc = ax[0, 0].scatter(xs[0][user_idx], ys[0][user_idx], c=colors[user_idx], alpha=1)
        
        def animate(i):
            fig.suptitle(f'round = {i}')
            sc.set_offsets(np.c_[xs[i][user_idx], ys[i][user_idx]])

        ani = FuncAnimation(fig, animate, frames=num_grounds, interval=50, repeat=False)

        ani.save(f'{prefix_figure}location_ani{user_idx}.gif', writer='imagemagick', fps=20)
        plt.close()

def plot_maps(fig_size=(8, 8), nrows=1, ncols=1, title=''): 
    # plot map 
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=fig_size, squeeze=False)
    # fig, ax = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False)

    # plot road map 
    x11 = [250+22, 500]; x12 = [250+22, -500]; x13 = [250-22, 500]; x14 = [250-22, -500];  xmid11 = [250, 500]; xmid12 = [250, -500]
    x21 = [-250+22, 500]; x22 = [-250+22, -500]; x23 = [-250-22, 500]; x24 = [-250-22, -500];  xmid21 = [-250, 500]; xmid22 = [-250, -500]
    x31 = [-500, 250+22]; x32 = [500, 250+22]; x33 = [-500, 250-22]; x34 = [500, 250-22];  xmid31 = [-500, 250]; xmid32 = [500, 250]
    x41 = [-500, 0+22]; x42 = [500, 0+22]; x43 = [-500, 0-22]; x44 = [500, 0-22];  xmid41 = [-500, 0]; xmid42 = [500, 0]
    x51 = [-500, -250+22]; x52 = [500, -250+22]; x53 = [-500, -250-22]; x54 = [500, -250-22];  xmid51 = [-500, -250]; xmid52 = [500, -250]

    # plot uav, bs 
    uav_x, uav_y = x_uav, y_uav
    bs_x, bs_y = x_bs, y_bs
    for i in range(nrows):
        for j in range(ncols): 
            # first road 
            ax[i][j].plot([x11[0], x12[0]], [x11[1], x12[1]], color='blue')
            ax[i][j].plot([x13[0], x14[0]], [x13[1], x14[1]], color='blue')
            ax[i][j].plot([xmid11[0], xmid12[0]], [xmid11[1], xmid12[1]], color='orange', linestyle='--')   

            # second road 
            ax[i][j].plot([x21[0], x22[0]], [x21[1], x22[1]], color='blue')
            ax[i][j].plot([x23[0], x24[0]], [x23[1], x24[1]], color='blue')
            ax[i][j].plot([xmid21[0], xmid22[0]], [xmid21[1], xmid22[1]], color='orange', linestyle='--')   

            # third road 
            ax[i][j].plot([x31[0], x32[0]], [x31[1], x32[1]], color='blue')
            ax[i][j].plot([x33[0], x34[0]], [x33[1], x34[1]], color='blue')
            ax[i][j].plot([xmid31[0], xmid32[0]], [xmid31[1], xmid32[1]], color='orange', linestyle='--')   

            ax[i][j].plot([x41[0], x42[0]], [x41[1], x42[1]], color='blue')
            ax[i][j].plot([x43[0], x44[0]], [x43[1], x44[1]], color='blue')
            ax[i][j].plot([xmid41[0], xmid42[0]], [xmid41[1], xmid42[1]], color='orange', linestyle='--')   

            ax[i][j].plot([x51[0], x52[0]], [x51[1], x52[1]], color='blue')
            ax[i][j].plot([x53[0], x54[0]], [x53[1], x54[1]], color='blue')
            ax[i][j].plot([xmid51[0], xmid52[0]], [xmid51[1], xmid52[1]], color='orange', linestyle='--')   


            ax[i][j].scatter([uav_x], [uav_y], marker="*", s=100, alpha=0.7, c='red')
            ax[i][j].annotate('UAV', (uav_x+10, uav_y+20))
            ax[i][j].scatter([bs_x], [bs_y], marker="p", s=70, alpha=0.7, c='green')
            ax[i][j].annotate('BS', (bs_x-15, bs_y+30))

            ax[i][j].set_xlim(-500, 500)
            ax[i][j].set_ylim(-500, 500)
            xstick = np.arange(-500, 510, 250)
            ax[i][j].set_xticks(xstick)
            ax[i][j].set_yticks(xstick)
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
    fig, ax = plot_maps(fig_size=(11, 5), nrows=1, ncols=2, title='Gain Density(dB)')
    
    # calculate max, min of color level 
    cmin = min(uav_gains_db.min(), bs_gains_db.min())
    cmax = max(uav_gains_db.max(), bs_gains_db.max())
    color_levels = np.linspace(start=cmin, stop=cmax, num=20)
    color_map = 'coolwarm'

    cs1 = ax[0, 0].contour(X, Y, bs_gains_db, levels=[-112, -110, -108, -105, -100, -90])
    ax[0, 0].clabel(cs1, inline=True, fontsize=14)
    ax[0, 0].set_title('BS')
    cs2 = ax[0, 1].contour(X, Y, uav_gains_db, levels=[-115, -110, -105, -100, -90])
    ax[0, 1].clabel(cs2, inline=True, fontsize=14)
    ax[0, 1].set_title('UAV')

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

def main(index=4, dataset='synthetic', tau=15, log_file='system_model.log'):
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

    log_name=f'tau{tau}_gamma{gamma}_cn{C_n}_vec{velocity}_optim{optim}.log'
    log_file = f'./logs/{dataset}/s{index}/{log_name}'
    prefix_figure = f'./figures/{dataset}/s{index}/' 
    if not os.path.exists(prefix_figure): 
        os.makedirs(prefix_figure)
    fig_file_ani = prefix_figure + 'location_ani.gif'
    plot_location_ani(log_file, fig_file_ani)
    plot_location_ani_all(log_file, prefix_figure)
    return 

if __name__=='__main__':
    main()
    # plot_gain_density()
    # plot_SNR()
    # test_plot_maps()