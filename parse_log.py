import re 
import numpy as np 
from src.network_utils import to_dB

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

def parse_fedl_w_test_acc(file_name): 
    rounds, train_acc, test_acc, train_loss, sim = [], [], [], [], []
    test_loss = []
    
    for line in open(file_name, 'r'):
        search_train_acc = re.search(r'At round (.*) training accuracy: (.*)', line, re.M|re.I)
        if search_train_acc: 
            rounds.append(int(search_train_acc.group(1)))
            train_acc.append(float(search_train_acc.group(2)))
        
        search_test_acc = re.search(r'At round (.*) test accuracy: (.*)', line, re.M|re.I)
        if search_test_acc: 
            test_acc.append(float(search_test_acc.group(2)))
        
        search_loss = re.search(r'At round (.*) training loss: (.*)', line, re.M|re.I)
        if search_loss: 
            train_loss.append(float(search_loss.group(2)))

        search_test_loss = re.search(r'At round (.*) test loss: (.*)', line, re.M|re.I)
        if search_test_loss: 
            test_loss.append(float(search_test_loss.group(2)))
        
        search_grad = re.search(r'gradient difference: (.*)', line, re.M|re.I)
        if search_grad: 
            sim.append(float(search_grad.group(1)))
    
    rounds = np.asarray(rounds)
    train_acc = np.asarray(train_acc) * 100
    test_acc = np.asarray(test_acc) * 100
    train_loss = np.asarray(train_loss)
    sim = np.asarray(sim)
    test_loss = np.asarray(test_loss)
        
    return rounds, train_acc, test_acc, train_loss, test_loss, sim

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

        search_eta = re.search(r'At round (.*) optimal eta: (.*)', line, re.M|re.I)
        if search_eta: 
            etas.append(float(search_eta.group(2)))
    
    lrounds = np.asarray(lrounds)
    grounds = np.asarray(grounds)
    ans = np.asarray(ans)
    etas = np.asarray(etas)
    
    return lrounds, grounds, ans, etas

def parse_net_tien(file_name): 
    t_co, t_cp, t, e_co, e_cp, e = [], [], [], [], [], []

    for line in open(file_name, 'r'):
        search_time = re.search(r'At round (.*) average t_co: (.*) average t_cp: (.*) t: (.*)', line, re.M|re.I)
        if search_time: 
            t_co.append(float(search_time.group(2)))
            t_cp.append(float(search_time.group(3)))  
            t.append(float(search_time.group(4))) 

        search_ene = re.search(r'At round (.*) average e_co: (.*) average e_cp: (.*) e: (.*)', line, re.M|re.I)
        if search_ene: 
            e_co.append(float(search_ene.group(2)))
            e_cp.append(float(search_ene.group(3)))
            e.append(float(search_ene.group(4))) 

    t_co = np.asarray(t_co)* 1000 # ms
    t_cp = np.asarray(t_cp)* 1000 # ms
    t = np.asarray(t)* 1000 # ms
    e_co = np.asarray(e_co)* 1000 # mJ
    e_cp = np.asarray(e_cp)* 1000 # mJ
    e = np.asarray(e)* 1000 # mJ
    return t_co, t_cp, t, e_co, e_cp, e

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

def parse_solutions(file_name):
    freqs, decs, powers = [], [], []

    for line in open(file_name, 'r'):
        search_freqs = re.search(r'At round (.*) optimal freqs: \[(.*)\]', line, re.M|re.I)
        if search_freqs: 
            freq = np.fromstring(search_freqs.group(2), sep=' ')
            freqs.append(freq) # [num_rounds, (num_users)]

        search_decs = re.search(r'At round (.*) optimal decs: (.*)', line, re.M|re.I)
        if search_decs: 
            dec = np.fromstring(search_decs.group(2), sep=' ', dtype=int)
            decs.append(dec) # [num_rounds, (num_users)]

        search_powers = re.search(r'At round (.*) optimal powers: \[(.*)\]', line, re.M|re.I)
        if search_powers: 
            power = np.fromstring(search_powers.group(2), sep=' ')
            powers.append(to_dB(power)) # [num_rounds, (num_users)] # dBm
    
    freqs = np.asarray(freqs).T # transpose # (num_users, num_rounds)
    powers = np.asarray(powers).T # transpose # (num_users, num_rounds)
    decs = np.asarray(decs).T # transpose # (num_users, num_rounds)
    
    return freqs, decs, powers

def parse_net_tien_array(file_name):
    t_co_s, e_co_s, t_cp_s, e_cp_s = [], [], [], []

    for line in open(file_name, 'r'):
        search_t_co = re.search(r'At round (.*) t_co: \[(.*)\]', line, re.M|re.I)
        if search_t_co: 
            t_co = np.fromstring(search_t_co.group(2), sep=' ')
            t_co_s.append(t_co) # [num_rounds, (num_users)]
        
        search_e_co = re.search(r'At round (.*) e_co: \[(.*)\]', line, re.M|re.I)
        if search_e_co: 
            e_co = np.fromstring(search_e_co.group(2), sep=' ')
            e_co_s.append(e_co) # [num_rounds, (num_users)]

        search_t_cp = re.search(r'At round (.*) t_cp: \[(.*)\]', line, re.M|re.I)
        if search_t_cp: 
            t_cp = np.fromstring(search_t_cp.group(2), sep=' ')
            t_cp_s.append(t_cp) # [num_rounds, (num_users)]

        search_e_cp = re.search(r'At round (.*) e_cp: \[(.*)\]', line, re.M|re.I)
        if search_e_cp: 
            e_cp = np.fromstring(search_e_cp.group(2), sep=' ')
            e_cp_s.append(e_cp) # [num_rounds, (num_users)]
    
    t_co_s = np.asarray(t_co_s).T # transpose # (num_users, num_rounds)
    t_cp_s = np.asarray(t_cp_s).T # transpose # (num_users, num_rounds)
    e_co_s = np.asarray(e_co_s).T # transpose # (num_users, num_rounds)
    e_cp_s = np.asarray(e_cp_s).T # transpose # (num_users, num_rounds)
    
    return t_co_s, t_cp_s, e_co_s, e_cp_s

def parse_num_samples(file_name): 
    samples = []
    for line in open(file_name, 'r'):
        search_samples = re.search(r'num_samples = \[(.*)\]', line, re.M|re.I)
        if search_samples: 
            n_samples = np.fromstring(search_samples.group(1), sep=' ', dtype=int)
            samples.append(n_samples)
            break 
    
    samples = samples[0][:]
    return samples