# Gaussian variables used for tests:
# train_torch_deterministic_gaussian: IsotonicReg with gaussian=True
# train_torch_ensembles_calibration: IsotonicReg with gaussian=True
# train_torch_dropout_calibration: IsotonicReg with gaussian=False, Conformal with gaussian=True
# train_torch_bitrap_BT: IsotonicReg with gaussian=False, Conformal with gaussian=False

import argparse
import logging, sys, random
import numpy as np
import torch
sys.path.append('.')

from utils.calibration_utils import get_data_for_calibration
from utils.calibration import generate_metrics_calibration, generate_metrics_calibration_all
from utils.constants import SUBDATASETS_NAMES, BITRAP, BITRAP_BT_SDD, DETERMINISTIC_GAUSSIAN, DETERMINISTIC_GAUSSIAN_SDD, DROPOUT, DROPOUT_SDD, ENSEMBLES, ENSEMBLES_SDD, VARIATIONAL, VARIATIONAL_SDD


# Parser arguments
parser = argparse.ArgumentParser(description='')
parser.add_argument('--calibration-conformal', action='store_true', help='generates metrics using calibration conformal')
parser.add_argument('--absolute-coords', action='store_true', help='')
parser.add_argument('--gaussian-output', default=False, action='store_true', help='gaussian var to be used to compute calibration metrics')
parser.add_argument('--show-plot', default=False, action='store_true', help='show the calibration plots')
parser.add_argument('--nll', default=False, action='store_true', help='Compute the values of GT negative log likelihood before and after calibration')
parser.add_argument('--test-name', type=str, default='deterministicGaussian', help='Test data to be load (default: deterministic gaussian test)')
parser.add_argument('--id-dataset',type=str, default=0, metavar='N',
					help='id of the dataset to use. 0 is ETH-UCY, 1 is SDD (default: 0)')
parser.add_argument('--id-test',type=int, default=2, metavar='N',
					help='id of the subdataset to use as test (default: 2)')
parser.add_argument('--seed',type=int, default=1,help='Random seed for all randomized functions')
parser.add_argument('--log-level',type=int, default=20,help='Log level (default: 20)')
parser.add_argument('--log-file',default='',help='Log file (default: standard output)')
args = parser.parse_args()

valid_test_names = {
	"deterministicGaussian": DETERMINISTIC_GAUSSIAN,
	"ensembles":             ENSEMBLES,
	"dropout":               DROPOUT,
	"bitrap":                BITRAP,
	"variational":           VARIATIONAL,
	"deterministicGaussianSDD": DETERMINISTIC_GAUSSIAN_SDD,
	"ensemblesSDD":             ENSEMBLES_SDD,
	"dropoutSDD":               DROPOUT_SDD,
	"bitrapSDD":                BITRAP_BT_SDD,
	"variationalSDD":           VARIATIONAL_SDD
}

def get_test_name():
	"""
	Args:
	Returns:
		- test_name
	"""
	if args.test_name not in valid_test_names.keys():
		return "ERROR: INVALID TEST NAME!!"
	return valid_test_names[args.test_name]+"_"+str(SUBDATASETS_NAMES[args.id_dataset][args.id_test])+"_calibration"


def compute_calibration_metrics():
	"""
	Evaluation of calibration metrics and calibration methods
	"""
	test_name = get_test_name()
	logging.info('Uncertainty calibration with '+SUBDATASETS_NAMES[args.id_dataset][args.id_test]+' as test dataset')
	# Load data for calibration compute
	predictions_calibration,predictions_test,observations_calibration,observations_test,groundtruth_calibration, groundtruth_test,__,__,sigmas_samples,sigmas_samples_full,id_test = get_data_for_calibration(test_name)

	# Resampling parameter
	kde_size      = 1500
	resample_size =  200

	# Calibrate and evaluate metrics for the three methods, and for all positions
	# 0: Conformal
	# 1: Conformal con densidad relativa
	# 2: Regresion Isotonica
	method_name = valid_test_names[args.test_name]+"_"+str(SUBDATASETS_NAMES[args.id_dataset][args.id_test])
	#generate_metrics_calibration(method_name,predictions_calibration,observations_calibration,groundtruth_calibration, predictions_test,observations_test,groundtruth_test, methods=[0,1,2],kde_size=kde_size,resample_size=resample_size,gaussian=[sigmas_samples, sigmas_samples_full])
	if args.gaussian_output:
		generate_metrics_calibration_all(method_name,predictions_calibration,observations_calibration,groundtruth_calibration, predictions_test,observations_test,groundtruth_test,kde_size=kde_size,relative_coords_flag=not args.absolute_coords,resample_size=resample_size,gaussian=[sigmas_samples, sigmas_samples_full])
	else:
		generate_metrics_calibration_all(method_name,predictions_calibration,observations_calibration,groundtruth_calibration, predictions_test,observations_test,groundtruth_test,kde_size=kde_size,relative_coords_flag=not args.absolute_coords,resample_size=resample_size,gaussian=[None, None])

if __name__ == "__main__":
	# Choose seed
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)
	np.random.seed(args.seed)
	random.seed(args.seed)
	# Loggin format
	logging.basicConfig(format='%(levelname)s: %(message)s',level=args.log_level)
	compute_calibration_metrics()
