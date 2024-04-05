import argparse
import yaml, os
from trajpred_unc.utils.constants import SUBDATASETS_NAMES

CONFIG_PATH = "cfg/"

def get_model_filename(config,ensemble_id=0):
	return config["train"]["model_name"]+"_{}_{}.pth".format(SUBDATASETS_NAMES[config["dataset"]["id_dataset"]][config["dataset"]["id_test"]],ensemble_id)

# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config