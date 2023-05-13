import numpy as np
import matplotlib.pyplot as plt
import os
import importlib
import torch
from tqdm import trange
from sklearn.manifold import TSNE
import pandas as pd 
import seaborn as sns 

from src.custom_dataset import * 

def load_model(train_data: dict):
    options={'dataset': 'mnist', 'sce_idx': 4, 'model': 'mclr', 'model_params': (784, 10)}
        
    # load selected model 
    # init the saved model 
    model_path = '%s.%s.%s.%s' % ('flearn', 'models', options['dataset'], options['model'])
    model_lib = importlib.import_module(model_path)
    model_mclr = getattr(model_lib, 'Model')(options['model_params'])

    # saved model direction 
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(parent_dir, 'models', options['dataset'], f's{str(options["sce_idx"])}')
    model_path = save_dir + '/'+ 'model_weights.pth' 

    # load the saved model
    model_mclr.model.load_state_dict(torch.load(model_path))
    print('Model loaded!')

    training_data =  CustomDataset(train_data)
    train_dataloader = DataLoader(training_data, batch_size=32, shuffle=False, drop_last=False)

    # evaluate on the train set
    y_pred = model_mclr.predict(train_dataloader, debug=False)

    return y_pred

def view_data(): 
    num_users = 10
    for user_id in trange(num_users): 
        train_data, _ = test_load_data(user_id=user_id, dataset_name='mnist')

        X, y = train_data['x'], train_data['y']
        
        y_pred = load_model(train_data)
        y_pred = np.asarray(y_pred)
        y = np.asarray(y) 
        dist = np.abs(y_pred - y) 

        widx = np.where(dist > 0)[0]
        cidx = np.where(dist == 0)[0]
        idxes = np.concatenate((widx, cidx), axis=0) if widx.size else cidx 

        fig, axs = plt.subplots(nrows=7, ncols=7, figsize=(7, 7))
        for i, ax in enumerate(axs.ravel()):
            idx = idxes[i]
            color = 'blue' if y_pred[idx] == y[idx] else 'red'
            # plot image 
            image = np.asarray(X[idx]).reshape((28, 28))
            ax.imshow(image, cmap=plt.cm.binary)
            ax.set_title(y_pred[idx], color=color, y=0.85)
            ax.axis("off")

        _ = fig.suptitle(f"MNIST user_id = {user_id}")
        plt.savefig(f'./figures/mnist/images/image_user{user_id}.png')
        plt.show()

def view_SNE():
    # configure TSNE model 
    # the number of components = 2
    # default perplexity = 30
    # default learning rate = 200
    # default Maximum number of iterations for the optimization = 1000
    model = TSNE(n_components=2, random_state=0)

    # load data 
    num_users = 10
    X_all, y_pred_all, y_all = np.array([]), np.array([]), np.array([])

    for user in range(num_users):
        test_data, _ = test_load_data(user_id=user, dataset_name='mnist')
        y_pred = load_model(test_data)
        X, y = test_data['x'], test_data['y']
        
        X_reshape = np.array([np.asarray(x) for x in X])
        y = np.array(y)
        y_pred = np.asarray(y_pred)

        X_all = np.concatenate((X_all, X_reshape), axis=0) if X_all.size else X_reshape
        y_pred_all = np.concatenate((y_pred_all, y_pred), axis=0) if y_pred_all.size else y_pred
        y_all = np.concatenate((y_all, y), axis=0) if y_all.size else y 
    
    print(f"X_all.shape = {X_all.shape}")
    print(f"y_all.shape = {y_all.shape}")

    tsne_data = model.fit_transform(X_all)
    print(f"tsne_data.shape = {tsne_data.shape}")

    # set marker size 
    dist = np.abs(y_pred_all - y_all) 
    widx = np.where(dist > 0)[0]
    s = np.ones(y_all.size) * 1
    s[widx] = 3
    
    print(f"s.size = {s.size} widx.shape = {widx.shape}")

    # Creating a new data frame which helps us in ploting the result data
    tsne_data = np.vstack((tsne_data.T, y_all, y_pred_all, s)).T
    columns =(r'${dim}_1$', r'${dim}_2$', "label", "pred", "size")
    tsne_df = pd.DataFrame(data=tsne_data, columns=columns)

    # Ploting the result of tsne
    sns.relplot(x=columns[0], y=columns[1], hue=columns[2], size=columns[4],  sizes=(40, 200), palette="bright", alpha=.8, height=9, legend="auto", data=tsne_df)
    plt.savefig(f'./figures/mnist/images/tSNE_label.png')
    plt.close()

    sns.relplot(x=columns[0], y=columns[1], hue=columns[3], size=columns[4],  sizes=(40, 200), palette="bright", alpha=.8, height=9, legend="auto", data=tsne_df)
    # legends = [str(i) for i in range(10)]
    # plt.legend(frameon=False, labels=legends, loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig(f'./figures/mnist/images/tSNE_pred.png')
    plt.close()
    # plt.show()

if __name__=='__main__': 
    # view_data()
    # load_model()
    view_SNE()