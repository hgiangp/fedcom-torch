import re 
import matplotlib.pyplot as plt
import numpy as np 

def parse_log(file_name): 
    rounds, acc, loss, sim = [], [], [], []
    
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
    
    return rounds, acc, loss, sim 

def test_parse_log(in_file, out_file): 
    rounds, acc, loss, sim = parse_log(file_name=in_file)

    print(f"acc = \n{acc[-5:]}") 
    print(f"loss = \n{loss[-5:]}") 
    print(f"sim = \n{sim[-5:]}")

    rounds = np.asarray(rounds)
    acc = np.asarray(acc) * 100
    loss = np.asarray(loss)
    sim = np.asarray(sim)

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
    plt.savefig(out_file) # plt.savefig('plot_mnist.png')
    plt.show()



if __name__=='__main__': 
    in_file, out_file = './logs/system_model.log', './logs/plot_synthetic_dy.png' 
    # in_file, out_file = './logs/server_model.log', './figures/plot_synthetic.png' 
    test_parse_log(in_file, out_file)
