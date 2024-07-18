# Gaussian variables used for tests:
# train_torch_deterministic_gaussian: IsotonicReg with gaussian=True
# train_torch_ensembles_calibration: IsotonicReg with gaussian=True
# train_torch_dropout_calibration: IsotonicReg with gaussian=False, Conformal with gaussian=True
# train_torch_bitrap_BT: IsotonicReg with gaussian=False, Conformal with gaussian=False

import logging, random
import numpy as np
import torch

from trajpred_unc.uncertainties.calibration_utils import get_data_for_calibration
from trajpred_unc.uncertainties.calibration import generate_calibration_metrics
from trajpred_unc.utils.constants import SUBDATASETS_NAMES, AGENTFORMER, BITRAP, BITRAP_BT_SDD, DETERMINISTIC_GAUSSIAN, DETERMINISTIC_GAUSSIAN_SDD, DROPOUT, DROPOUT_SDD, ENSEMBLES, ENSEMBLES_GAUSSIAN, ENSEMBLES_SDD, VARIATIONAL, VARIATIONAL_SDD
from trajpred_unc.utils.config import load_config

# Load configuration file (conditional model)
config = load_config("deterministic_gaussian_ethucy.yaml")
config = load_config("deterministic_dropout_ethucy.yaml")
#config = load_config("bitrap_ethucy.yaml")
#config = load_config("socialvae_ethucy.yaml")

def get_names(config):
	"""
	Args:
	"""
	if config["misc"]["ensemble"]:
		method_name          = config["train"]["model_name"]+"_ensemble"
	else:
		method_name          = config["train"]["model_name"]
	calibration_filename = method_name+"_"+str(SUBDATASETS_NAMES[config["dataset"]["id_dataset"]][config["dataset"]["id_test"]])+"_calibration"
	return method_name, calibration_filename

def compute_calibration_metrics():
	"""
	Evaluation of calibration metrics and calibration methods
	"""
	method_name,calibration_filename = get_names(config)
	logging.info('Evaluation of '+method_name+' model')
	logging.info('Uncertainty calibration with '+SUBDATASETS_NAMES[config["dataset"]["id_dataset"]][config["dataset"]["id_test"]]+' as test dataset')
	# Load data for calibration compute
	predictions_calibration,predictions_test,observations_calibration,observations_test,groundtruth_calibration, groundtruth_test,sigmas_calibration,sigmas_test,__ = get_data_for_calibration(calibration_filename)

	# Resampling parameter
	kde_size      = 150
	resample_size = 100

	# Calibrate and evaluate metrics for the three methods, and for all positions
	# 0: Conformal
	# 1: Conformal with relative density
	# 2: Regresion Isotonica
	if sigmas_calibration is not None:
		generate_calibration_metrics(method_name,predictions_calibration,observations_calibration,groundtruth_calibration,predictions_test,observations_test,groundtruth_test,methods=[0,1,2],kde_size=kde_size,resample_size=resample_size,gaussian=[sigmas_calibration, sigmas_test])
	else:
		generate_calibration_metrics(method_name,predictions_calibration,observations_calibration,groundtruth_calibration,predictions_test,observations_test,groundtruth_test,methods=[0,1,2],kde_size=kde_size,resample_size=resample_size,gaussian=[None, None])

if __name__ == "__main__":
	# Choose seed
	torch.manual_seed(config["misc"]["seed"])
	torch.cuda.manual_seed(config["misc"]["seed"])
	np.random.seed(config["misc"]["seed"])
	random.seed(config["misc"]["seed"])
	# Loggin format
	logging.basicConfig(format='%(levelname)s: %(message)s',level=config["misc"]["log_level"])
	compute_calibration_metrics()
