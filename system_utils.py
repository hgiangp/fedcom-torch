import importlib
import os
import argparse 

from custom_dataset import read_data

def load_model(mod_name='CustomLogisticRegression'): 
    r""" Load the client model
    Args: 
    Return: 
    """ 
    model_path = '%s' % ('custom_model')
    mod = importlib.import_module(model_path)
    model = getattr(mod, mod_name)
    return model 

def load_data(dataset_name='synthetic'): 
    r""" Load the dataset
    Args: 
        dataset_name = ['synthetic', 'mnist']
    Return: 
    """
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_dir = os.path.join(parent_dir, 'data', dataset_name, 'data', 'train')
    test_dir = os.path.join(parent_dir, 'data', dataset_name, 'data', 'test')
    dataset = read_data(train_dir, test_dir)
    return dataset

# GLOBAL PARAMETERS
SCENARIO_IDXES = [1, 2, 3, 4]
DATASETS = ['synthetic', 'mnist']

def read_options(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--sce_idx',
                    help='index of simulation scenario;',
                    type=int,
                    choices=SCENARIO_IDXES,
                    default=4)
    
    parser.add_argument('--tau',
                help='deadline of federated learning process;',
                type=float,
                default=40)

    parser.add_argument('--dataset',
            help='name of dataset;',
            type=str,
            choices=DATASETS,
            default='synthetic')
    
    try: parsed = vars(parser.parse_args())
    except IOError as msg: parser.error(str(msg))

    return parsed