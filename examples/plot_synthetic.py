import re 
import matplotlib.pyplot as plt
import numpy as np 

def parse_log(file_name): 
    rounds, acc, loss, sim = [], [], [], []
    lrounds, grounds, ans = [], [], []
    etas, energies = [], []
    
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

        search_grad = re.search(r'gradient difference: (.*)', line, re.M|re.I)
        if search_grad: 
            sim.append(float(search_grad.group(1)))

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
        
        search_ene = re.search(r'At round (.*) energy consumption: (.*)', line, re.M|re.I)
        if search_ene: 
            energies.append(float(search_ene.group(2)))
                
    return rounds, acc, loss, sim, lrounds, grounds, ans, etas, energies

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

def parse_locs(file_name): 
    xs, ys = [], []
    for line in open(file_name, 'r'): 
        search_xs = re.search(r'xs mean: (.*)$', line, re.M|re.I)
        if search_xs: 
            xs.append(float(search_xs.group(1)))
        
        search_ys = re.search(r'ys mean: (.*)$', line, re.M|re.I)
        if search_ys: 
            ys.append(float(search_ys.group(1)))
    
    return xs, ys  

def test_parse_log(in_file, out_file1, out_file2): 
    rounds, acc, loss, sim, lrounds, grounds, ans, etas, energies = parse_log(file_name=in_file)

    print(f"acc = \n{acc[-5:]}") 
    print(f"loss = \n{loss[-5:]}") 
    print(f"sim = \n{sim[-5:]}")

    rounds = np.asarray(rounds)
    acc = np.asarray(acc) * 100
    loss = np.asarray(loss)
    sim = np.asarray(sim)
    lrounds = np.asarray(lrounds)
    grounds = np.asarray(grounds)
    ans = np.asarray(ans)
    etas = np.asarray(etas)
    energies = np.asarray(energies)

    plt.figure(1)
    plt.subplot(311)
    plt.plot(rounds, acc)
    plt.ylabel("Accuracy")
    plt.grid(which='both')

    plt.subplot(312)
    plt.plot(rounds, loss)
    plt.ylabel("Loss")
    plt.grid(which='both')
        
    plt.subplot(313)
    plt.plot(rounds, sim)
    plt.ylabel("Dissimilarity")

    plt.grid(which='both')
    plt.savefig(out_file1) # plt.savefig('plot_mnist.png')
    plt.show()

    plt.figure(2)
    plt.subplot(411)
    plt.plot(rounds, lrounds)
    plt.grid(which='both')
    plt.ylabel("Lrounds")

    plt.subplot(412)
    plt.plot(rounds, grounds)
    plt.grid(which='both')
    plt.ylabel("Grounds")
    
    plt.subplot(413)
    plt.plot(rounds, ans)
    plt.grid(which='both')
    plt.ylabel("a_n")

    plt.subplot(414)
    plt.plot(rounds, etas)
    plt.grid(which='both')
    plt.ylabel("eta")

    plt.savefig(out_file2)
    plt.show()

    plt.figure(3)
    plt.plot(rounds, energies)
    plt.ylabel("Energy consumption (J)")
    plt.savefig("./figures/plot_synthetic_ene.png")
    plt.grid(which='both')
    plt.show()

def test_fixedi(in_file='./logs/system_model_fixedi.log'): 
    rounds, acc, loss, sim, _, _, _, _, energies = parse_log(file_name=in_file)

    rounds = np.asarray(rounds)
    acc = np.asarray(acc) * 100
    loss = np.asarray(loss)
    sim = np.asarray(sim)
    energies = np.asarray(energies)

    plt.figure(1)
    plt.subplot(311)
    plt.plot(rounds, acc)
    plt.ylabel("Accuracy")
    
    plt.subplot(312)
    plt.plot(rounds, loss)
    plt.ylabel("Loss")
        
    plt.subplot(313)
    plt.plot(rounds, sim)
    plt.ylabel("Dissimilarity")

    plt.savefig("./figures/plot_synthetic_fixedi.png") # plt.savefig('plot_mnist.png')
    plt.show()

    plt.figure(2)
    plt.plot(rounds[:81], energies[:81])
    plt.ylabel("Energy consumption (J)")
    plt.savefig("./figures/plot_synthetic_ene_fixedi.png")
    plt.show()

def ene_plot():
    rounds_dy, acc_dy, loss_dy, sim_dy, lrounds_dy, grounds_dy, ans_dy, etas_dy, energies_dy = parse_log('./logs/system_model.log')
    rounds, acc, loss, sim, _, _, _, _, energies = parse_log('./logs/system_model_fixedi.log')

    max_round = len(rounds_dy)
    plt.figure(1)
    plt.plot(rounds_dy, energies_dy, label='Dynamic')
    plt.plot(rounds[:max_round], energies[:max_round], label='Fixed i')
    plt.grid(visible=True, which='both')
    plt.legend()
    plt.savefig('./figures/plot_synthetic_ene_compared.png')
    plt.show()

def test_prase_gains(): 
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

def test_parse_loc(): 
    loc_x, loc_y = parse_locs('./logs/system_model.log')
    rounds = np.arange(0, len(loc_x))

    plt.figure(1)
    # plt.plot(loc_x, loc_y, label='(x, y)')
    plt.plot(rounds, loc_x, label='Loc x')
    plt.plot(rounds, loc_y, label='Loc y')
    plt.grid(visible=True, which='both')
    plt.legend()

    plt.savefig('./figures/locations.png')
    plt.show()

def test_system_model(): 
    in_file, out_file1, out_file2 = './logs/system_model.log', './figures/plot_synthetic_dy1.png', './figures/plot_synthetic_dy2.png'
    test_parse_log(in_file, out_file1, out_file2)
    test_prase_gains()
    test_parse_loc()

def test_server_model(): 
    in_file, out_file1, out_file2 = './logs/server_model.log', './figures/plot_synthetic.png', './figures/dumb.png'
    test_parse_log(in_file, out_file1, out_file2)

if __name__=='__main__': 
    # ene_plot()
    test_system_model()
    # test_fixedi()
    # test_server_model()
