import argparse 

# GLOBAL PARAMETERS
SCENARIO_IDXES = [1, 2, 3, 4]

def read_options(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--sce_idx',
                    help='index of simulation scenario;',
                    type=int,
                    choices=SCENARIO_IDXES,
                    default=4)
    
    try: parsed = vars(parser.parse_args())
    except IOError as msg: parser.error(str(msg))

    return parsed