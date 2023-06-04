import argparse 
import importlib
import os

from src.custom_dataset import read_data
from src.system_model import SystemModel

# GLOBAL PARAMETERS
SCENARIO_IDXES = [1, 2, 3, 4]
DATASETS = ['synthetic', 'mnist', 'cifar10']
MODEL_PARAMS = {
    'synthetic.mclr': (5, 3), # num_inputs, num_classes,
    'mnist.mclr': (784, 10), # num_classes
    'cifar10.mclr': (3, 32, 32) # 32x32x3 color channel
}
OPTIM_OPTIONS = [1, 2, 3, 4]

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
    parser.add_argument('--model',
                        help='name of model;',
                        type=str,
                        default='mclr')
    parser.add_argument('--learning_rate',
                        help='learning rate of local model;',
                        type=float,
                        default=0.001)
    parser.add_argument('--optim',
                        help='power and freq is optimized;',
                        type=int,
                        choices=OPTIM_OPTIONS,
                        default=1)
    parser.add_argument('--xi_factor',
                        help='deadline of federated learning process;',
                        type=float,
                        default=1)
    
    try: parsed = vars(parser.parse_args())
    except IOError as msg: parser.error(str(msg))

    # load selected model 
    model_path = '%s.%s.%s.%s' % ('flearn', 'models', parsed['dataset'], parsed['model'])
    mod = importlib.import_module(model_path)
    model = getattr(mod, 'Model')

    # load model parameters 
    parsed['model_params'] = MODEL_PARAMS['.'.join(model_path.split('.')[2:])]

    # print and return 
    maxLen = max([len(i) for i in parsed.keys()])
    fmtString = '\t%' + str(maxLen) + 's : %s'

    print('Arguments:')
    for keyPair in sorted(parsed.items()): print(fmtString % keyPair)

    return parsed, model

def main(): 
    options, model = read_options()
    # read data 
    train_dir = os.path.join('data', options['dataset'], 'data', 'train')
    test_dir = os.path.join('data', options['dataset'], 'data', 'test')
    dataset = read_data(train_dir, test_dir)
    
    # call system model 
    sys_mod = SystemModel(options, model, dataset)
    sys_mod.run()
    sys_mod.save_model()

if __name__=='__main__':
    main()
