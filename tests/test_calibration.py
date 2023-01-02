# Gaussian variables used for tests:
# train_torch_deterministic_gaussian: IsotonicReg with gaussian=True
# train_torch_ensembles_calibration: IsotonicReg with gaussian=True
# train_torch_dropout_calibration: IsotonicReg with gaussian=False, Conformal with gaussian=True
# train_torch_bitrap_BT: IsotonicReg with gaussian=False, Conformal with gaussian=False

import argparse
import logging, sys
import numpy as np
import torch
sys.path.append('.')

from utils.calibration_utils import get_data_for_calibration
from utils.calibration import generate_metrics_calibration_IsotonicReg, generate_metrics_calibration_conformal
from utils.constants import BITRAP_BT, BITRAP_BT_SDD, DETERMINISTIC_GAUSSIAN, DETERMINISTIC_GAUSSIAN_SDD, DROPOUT, DROPOUT_SDD, ENSEMBLES, ENSEMBLES_SDD, VARIATIONAL, VARIATIONAL_SDD

from utils.calibration import generate_metrics_calibration

# Parser arguments
parser = argparse.ArgumentParser(description='')
parser.add_argument('--calibration-conformal', action='store_true', help='generates metrics using calibration conformal')
parser.add_argument('--gaussian-output', default=False, action='store_true', help='gaussian var to be used to compute calibration metrics')
parser.add_argument('--show-plot', default=False, action='store_true', help='show the calibration plots')
parser.add_argument('--nll', default=False, action='store_true', help='Compute the values of GT negative log likelihood before and after calibration')
parser.add_argument('--test-name', type=str, default='deterministicGaussian', help='Test data to be load (default: deterministic gaussian test)')
parser.add_argument('--log-level',type=int, default=20,help='Log level (default: 20)')
parser.add_argument('--log-file',default='',help='Log file (default: standard output)')
args = parser.parse_args()

def get_test_name():
    """
    Args:
    Returns:
        - test_name
    """
    valid_test_names = {
        "deterministicGaussian": DETERMINISTIC_GAUSSIAN,
        "ensembles":             ENSEMBLES,
        "dropout":               DROPOUT,
        "bitrap":                BITRAP_BT,
        "variational":           VARIATIONAL,
        "deterministicGaussianSDD": DETERMINISTIC_GAUSSIAN_SDD,
        "ensemblesSDD":             ENSEMBLES_SDD,
        "dropoutSDD":               DROPOUT_SDD,
        "bitrapSDD":                BITRAP_BT_SDD,
        "variationalSDD":           VARIATIONAL_SDD
        }
    if args.test_name not in valid_test_names.keys():
        return "ERROR: INVALID TEST NAME!!"
    return valid_test_names[args.test_name]

def compute_calibration_metrics():
    """
    Compute Isotonic Regression (by default) and conformal calibration metrics (if provided argument)
    """
    test_name = get_test_name()
    # Load data for calibration compute
    tpred_samples, tpred_samples_full, data_test, data_test_full, target_test, target_test_full, targetrel_test, targetrel_test_full, sigmas_samples, sigmas_samples_full, id_test = get_data_for_calibration(test_name)
    if args.calibration_conformal:
        logging.info("*************************************")
        logging.info("******* Conformal Calibration *******")
        logging.info("*************************************")
        generate_metrics_calibration_conformal(tpred_samples, data_test, targetrel_test, target_test, sigmas_samples, id_test, gaussian=args.gaussian_output, tpred_samples_test=tpred_samples_full, data_test=data_test_full, targetrel_test=targetrel_test_full, target_test=target_test_full, sigmas_samples_test=sigmas_samples_full,show_plot=args.show_plot)

    # ---------------------------------- Calibration HDR -------------------------------------------------
    logging.info("*******************************************")
    logging.info("***** Isotonic Regression Calibration *****")
    logging.info("*******************************************")
    generate_metrics_calibration_IsotonicReg(tpred_samples, data_test, target_test, sigmas_samples, id_test, gaussian=args.gaussian_output, tpred_samples_test=tpred_samples_full, data_test=data_test_full, target_test=target_test_full, sigmas_samples_test=sigmas_samples_full,compute_nll=args.nll,show_plot=args.show_plot)


def compute_calibration_metrics2():
    """
    Compute Isotonic Regression (by default) and conformal calibration metrics (if provided argument)
    """
    test_name = get_test_name()
    # Load data for calibration compute
    tpred_samples, tpred_samples_full, data_test, data_test_full, target_test, target_test_full, targetrel_test, targetrel_test_full, sigmas_samples, sigmas_samples_full, id_test = get_data_for_calibration(test_name)

    # Size del remuestreo
    resample_size = 1000
    
    # Conjunto de calibracion
    data_pred = tpred_samples
    data_obs = data_test
    data_gt = target_test
    # Conjunto de evaluacion
    data_pred = tpred_samples_full
    data_obs = data_test_full
    data_gt = target_test_full
    
    # Calculamos los tres metodos
    # 0: Conformal
    # 1: Conformal con densidad relativa
    # 2: Regresion Isotonica
    generate_metrics_calibration(data_pred, data_obs, data_gt, data_pred, data_obs, data_gt, methods=[0,1,2], resample_size=resample_size)
    #generate_metrics_calibration(data_pred, data_obs, data_gt, data_pred, data_obs, data_gt, methods=[0,1,2], resample_size=resample_size, gaussian=[sigmas_samples, sigmas_samples_full])


if __name__ == "__main__":
    # Loggin format
    logging.basicConfig(format='%(levelname)s: %(message)s',level=args.log_level)
    compute_calibration_metrics()
