import re 
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
    
    lrounds = np.asarray(lrounds)
    grounds = np.asarray(grounds)
    ans = np.asarray(ans)
    etas = np.asarray(etas)
    
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