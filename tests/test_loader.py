# Imports
import sys,os,logging
import torch
sys.path.append('.')
from utils.datasets_utils import get_dataset
from utils.config import get_config


def test_get_dataset():
	# Parser arguments
	config = get_config(argv=['--id-test', '0'])
	# Loggin format
	logging.basicConfig(format='%(levelname)s: %(message)s',level=config.log_level)
	# Device
	if torch.cuda.is_available():
		logging.info(torch.cuda.get_device_name(torch.cuda.current_device()))
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# Get the data
	batched_train_data,batched_val_data,batched_test_data,homography,reference_image = get_dataset(config)
