# Gaussian variables used for tests:
# train_torch_deterministic_gaussian: IsotonicReg with gaussian=True
# train_torch_ensembles_calibration: IsotonicReg with gaussian=True
# train_torch_dropout_calibration: IsotonicReg with gaussian=False, Conformal with gaussian=True
# train_torch_bitrap_BT: IsotonicReg with gaussian=False, Conformal with gaussian=False

import argparse
import logging, sys, random
import numpy as np
import torch

from trajpred_unc.uncertainties.calibration_utils import get_data_for_calibration
from trajpred_unc.uncertainties.calibration import generate_metrics_calibration_all
from trajpred_unc.utils.constants import SUBDATASETS_NAMES, AGENTFORMER, BITRAP, BITRAP_BT_SDD, DETERMINISTIC_GAUSSIAN, DETERMINISTIC_GAUSSIAN_SDD, DROPOUT, DROPOUT_SDD, ENSEMBLES, ENSEMBLES_GAUSSIAN, ENSEMBLES_SDD, VARIATIONAL, VARIATIONAL_SDD
from trajpred_unc.utils.config import load_config

# Parser arguments
parser = argparse.ArgumentParser(description='')
parser.add_argument('--calibration-conformal', action='store_true', help='generates metrics using calibration conformal')
parser.add_argument('--absolute-coords', action='store_true', help='')
parser.add_argument('--gaussian-output', default=True, action='store_true', help='gaussian var to be used to compute calibration metrics')
parser.add_argument('--test-name', type=str, default='deterministicGaussian', help='Test data to be load (default: deterministic gaussian test)')
parser.add_argument('--seed',type=int, default=1,help='Random seed for all randomized functions')
args = parser.parse_args()

# Load configuration file (conditional model)
config = load_config("deterministic_gaussian_ethucy.yaml")

def get_names(config,ensemble=False):
	"""
	Args:
	"""
	if ensemble:
		method_name          = config["train"]["model_name"]+"_ensemble_"
	else:
		method_name          = config["train"]["model_name"]
	calibration_filename = method_name+str(SUBDATASETS_NAMES[config["dataset"]["id_dataset"]][config["dataset"]["id_test"]])+"_calibration"
	return method_name, calibration_filename

def compute_calibration_metrics():
	"""
	Evaluation of calibration metrics and calibration methods
	"""
	method_name,calibration_filename = get_names(config,True)
	logging.info('Uncertainty calibration with '+SUBDATASETS_NAMES[config["dataset"]["id_dataset"]][config["dataset"]["id_test"]]+' as test dataset')
	# Load data for calibration compute
	predictions_calibration,predictions_test,observations_calibration,observations_test,groundtruth_calibration, groundtruth_test,sigmas_samples,sigmas_samples_full,id_test = get_data_for_calibration(calibration_filename)

	# Resampling parameter
	kde_size      = 1500
	resample_size = 1000
	kde_size      = 15
	resample_size = 10

	# Calibrate and evaluate metrics for the three methods, and for all positions
	# 0: Conformal
	# 1: Conformal with relative density
	# 2: Regresion Isotonica
	if args.gaussian_output:
		generate_metrics_calibration_all(method_name,predictions_calibration,observations_calibration,groundtruth_calibration,predictions_test,observations_test,groundtruth_test,methods=[0,1,2],kde_size=kde_size,relative_coords_flag=not args.absolute_coords,resample_size=resample_size,gaussian=[sigmas_samples, sigmas_samples_full])
	else:
		generate_metrics_calibration_all(method_name,predictions_calibration,observations_calibration,groundtruth_calibration,predictions_test,observations_test,groundtruth_test,methods=[0,1,2],kde_size=kde_size,relative_coords_flag=not args.absolute_coords,resample_size=resample_size,gaussian=[None, None])

if __name__ == "__main__":
	# Choose seed
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)
	np.random.seed(args.seed)
	random.seed(args.seed)
	# Loggin format
	logging.basicConfig(format='%(levelname)s: %(message)s',level=config["misc"]["log_level"])
	compute_calibration_metrics()
