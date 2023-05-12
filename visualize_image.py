import numpy as np
import matplotlib.pyplot as plt
import os
import importlib
import torch
from tqdm import trange

from src.custom_dataset import test_load_data, load_dataloader

def view_data(): 
    num_users = 10
    for user_id in trange(num_users): 
        train_data, test_data = test_load_data(user_id=user_id, dataset_name='mnist')
        X, y = train_data['x'], train_data['y']
        y_pred = load_model(user_id)

        fig, axs = plt.subplots(nrows=7, ncols=7, figsize=(7, 7))
        for idx, ax in enumerate(axs.ravel()):
            # select color of title 
            color = 'blue' if y_pred[idx] == y[idx] else 'red'
            # plot image 
            image = np.asarray(X[idx]).reshape((28, 28))
            ax.imshow(image, cmap=plt.cm.binary)
            ax.set_title(y_pred[idx], color=color, y=0.85)
            ax.axis("off")

        _ = fig.suptitle(f"MNIST user_id = {user_id}")
        plt.savefig(f'./figures/mnist/images/image_user{user_id}.png')
        # plt.show()

def load_model(user_id=1):
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

    train_data, test_data = test_load_data(user_id=user_id, dataset_name='mnist')
    train_dataloader, test_dataloader = load_dataloader(train_data, test_data, shuffle=False)

    # evaluate on the train set
    y_pred = model_mclr.predict(train_dataloader, debug=False)

    return y_pred

if __name__=='__main__': 
    view_data()
    # load_model()