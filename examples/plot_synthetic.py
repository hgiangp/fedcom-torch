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
    
    plt.subplot(312)
    plt.plot(rounds, loss)
    plt.ylabel("Loss")
        
    plt.subplot(313)
    plt.plot(rounds, sim)
    plt.ylabel("Dissimilarity")

    plt.savefig(out_file1) # plt.savefig('plot_mnist.png')
    plt.show()

    plt.figure(2)
    plt.subplot(411)
    plt.plot(rounds, lrounds)
    plt.ylabel("Lrounds")

    plt.subplot(412)
    plt.plot(rounds, grounds)
    plt.ylabel("Grounds")
    
    plt.subplot(413)
    plt.plot(rounds, ans)
    plt.ylabel("a_n")

    plt.subplot(414)
    plt.plot(rounds, etas)
    plt.ylabel("eta")

    plt.savefig(out_file2)
    plt.show()

    plt.figure(3)
    plt.plot(rounds, energies)
    plt.ylabel("Energy consumption (J)")
    plt.savefig("plot_synthetic_ene.png")
    plt.show()

if __name__=='__main__': 
    in_file, out_file1, out_file2 = './logs/system_model.log', './figures/plot_synthetic_dy1.png', './figures/plot_synthetic_dy2.png'  
    # in_file, out_file1, out_file2 = './logs/server_model.log', './figures/plot_synthetic.png', './figures/dump.png' 
    test_parse_log(in_file, out_file1, out_file2)
