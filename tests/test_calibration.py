import argparse
import logging, sys
sys.path.append('.')

# Local constants
from utils.constants import TEST_DETERMINISTIC_GAUSSIAN
from utils.calibration_utils import get_data_for_calibration
from utils.calibration import generate_metrics_calibration_IsotonicReg, generate_metrics_calibration_conformal

# Parser arguments
parser = argparse.ArgumentParser(description='')
parser.add_argument('--calibration-conformal', action='store_true', help='generates metrics using calibration conformal')
args = parser.parse_args()

def compute_calibration_metrics():
    """
    Compute Isotonic Regression (by default) and conformal calibration metrics (if provided argument)
    """
    # Load data for calibration compute
    tpred_samples, tpred_samples_full, data_test, data_test_full, target_test, target_test_full, targetrel_test, targetrel_test_full, sigmas_samples, sigmas_samples_full, id_test, gaussian = get_data_for_calibration(TEST_DETERMINISTIC_GAUSSIAN)

    # ---------------------------------- Calibration HDR -------------------------------------------------
    logging.info("*******************************************")
    logging.info("***** Isotonic Regression Calibration *****")
    logging.info("*******************************************")
    generate_metrics_calibration_IsotonicReg(tpred_samples, data_test, target_test, sigmas_samples, id_test, gaussian=gaussian, tpred_samples_test=tpred_samples_full, data_test=data_test_full, target_test=target_test_full, sigmas_samples_test=sigmas_samples_full)

    if args.calibration_conformal:
        logging.info("*************************************")
        logging.info("******* Conformal Calibration *******")
        logging.info("*************************************")
        generate_metrics_calibration_conformal(tpred_samples, data_test, targetrel_test, target_test, sigmas_samples, args.id_test, gaussian=True, tpred_samples_test=tpred_samples_full, data_test=data_test_full, targetrel_test=targetrel_test_full, target_test=target_test_full, sigmas_samples_test=sigmas_samples_full)

if __name__ == "__main__":
    compute_calibration_metrics()