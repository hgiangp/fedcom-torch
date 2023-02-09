import importlib
import os 

from custom_dataset import read_data 

class BaseFederated(object): 
    def __init__(self, params, learner, dataset):
        print("BaseFederated generated!")


def test(): 

    # load the client model 
    model_path = '%s' % ('custom_model')
    mod = importlib.import_module(model_path)
    learner = getattr(mod, 'CustomModel')

    # load the dataset 
    dataset_name = 'synthetic'
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_dir = os.path.join(parent_dir, 'data', dataset_name, 'data', 'train')
    test_dir = os.path.join(parent_dir, 'data', dataset_name, 'data', 'test')
    dataset = read_data(train_dir, test_dir)

    # TODO: check params 
    params = {}
    t = BaseFederated(params, learner, dataset)
    
    pass

if __name__=="__main__": 
    test()

