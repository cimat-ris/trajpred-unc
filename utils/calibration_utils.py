import logging
import os
import pickle
# Local constants
from utils.constants import PICKLE_DIR
from utils.directory_utils import mkdir_p

def save_data_for_calibration(test_name, tpred_samples, tpred_samples_full, data_test, data_test_full, target_test, target_test_full, targetrel_test, targetrel_test_full, sigmas_samples, sigmas_samples_test, id_test):
    """
    Pickle provided data for future calibration compute
	Args:
        - test_name
        - tpred_samples
        - tpred_samples_full
        - data_test
        - data_test_full
        - target_test
        - target_test_full
        - targetrel_test
        - targetrel_test_full
        - sigmas_samples
        - sigmas_samples_test
        - id_test
	Returns:
	"""
    # make one data object
    data_for_calibration = {
        "TPRED_SAMPLES":        tpred_samples,
        "TPRED_SAMPLES_FULL":   tpred_samples_full,
        "DATA_TEST":            data_test,
        "DATA_TEST_FULL":       data_test_full,
        "TARGET_TEST":          target_test,
        "TARGET_TEST_FULL":     target_test_full,
        "TARGETREL_TEST":       targetrel_test,
        "TARGETREL_TEST_FULL":  targetrel_test_full,
        "SIGMAS_SAMPLES":       sigmas_samples,
        "SIGMAS_SAMPLES_TEST":  sigmas_samples_test,
        "ID_TEST":              id_test
    }
    # Creates pickle directory if does not exists
    mkdir_p(PICKLE_DIR)
    pickle_out_name = os.path.join(PICKLE_DIR, test_name+".pickle")
    pickle_out = open(pickle_out_name, "wb")
    pickle.dump(data_for_calibration, pickle_out, protocol=2)
    pickle_out.close()
    logging.info("Pickling data for calibration compute...")

def get_data_for_calibration(test_name):
    """
    Unpickle data for future calibration compute
    Args:
        - test_name
	Returns:
        - tpred_samples
        - tpred_samples_full
        - data_test
        - data_test_full
        - target_test
        - target_test_full
        - targetrel_test
        - targetrel_test_full
        - sigmas_samples
        - sigmas_samples_test
        - id_test
    """
    logging.info("Unpickling data for calibration compute...")
    pickle_in_name = os.path.join(PICKLE_DIR, test_name+".pickle")
    pickle_in = open(pickle_in_name, "rb")
    data_for_calibration = pickle.load(pickle_in)
    return data_for_calibration.values()