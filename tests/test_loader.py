# Imports
import sys,os,logging
import torch
from trajpred_unc.utils.datasets_utils import get_dataset
from trajpred_unc.utils.config import load_config


def test_get_dataset():
	# Parser arguments
	config = load_config("deterministic_ethucy.yaml")
	# Loggin format
	logging.basicConfig(format='%(levelname)s: %(message)s',level=config["misc"]["log_level"])
	# Device
	if torch.cuda.is_available():
		logging.info(torch.cuda.get_device_name(torch.cuda.current_device()))
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# Get the data
	batched_train_data,batched_val_data,batched_test_data,homography,reference_image = get_dataset(config["dataset"])
