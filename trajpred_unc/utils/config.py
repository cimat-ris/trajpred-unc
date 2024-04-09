import argparse
import yaml, os, sys
from trajpred_unc.utils.constants import SUBDATASETS_NAMES

CONFIG_PATH = "cfg/"

def get_model_filename(config,ensemble_id=0):
	return config["train"]["model_name"]+"_{}_{}.pth".format(SUBDATASETS_NAMES[config["dataset"]["id_dataset"]][config["dataset"]["id_test"]],ensemble_id)

# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    # Parser arguments
    args = get_args(argv=sys.argv[1:])
    config["dataset"]["id_dataset"] = args.id_dataset
    config["dataset"]["id_test"]    = args.id_test
    config["dataset"]["pickle"]     = args.pickle 
    config["misc"]["log_level"]     = args.log_level
    config["misc"]["log_file"]      = args.log_file
    config["misc"]["seed"]          = args.seed
    return config

arg_lists = []

def get_args(argv=None):
    # Parser arguments
    parser = argparse.ArgumentParser(description='')
    def add_argument_group(name):
        arg = parser.add_argument_group(name)
        arg_lists.append(arg)
        return arg

	# Data arguments
    data_args = add_argument_group('Data')
    data_args.add_argument('--id-dataset',
						type=str, default=0, metavar='N',
						help='id of the dataset to use. 0 is ETH-UCY, 1 is SDD (default: 0)')
    data_args.add_argument('--id-test',
						type=int, default=2, metavar='N',
						help='id of the dataset to use as test in LOO (default: 2)')
    data_args.add_argument('--pickle',
						action='store_true',
						help='use previously made pickle files')

    misc_args = add_argument_group('Misc')
    misc_args.add_argument('--seed',type=int, default=1,help='Random seed for all randomized functions')
    misc_args.add_argument('--log-level',type=int, default=20,help='Log level (default: 20)')
    misc_args.add_argument('--log-file',default='',help='Log file (default: standard output)')

    args = parser.parse_args(argv)
    return args