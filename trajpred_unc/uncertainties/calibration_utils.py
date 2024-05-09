import logging
import os
import random
import pickle
# Local constants
from trajpred_unc.utils.constants import PICKLE_DIR
from trajpred_unc.utils.directory_utils import mkdir_p

def save_data_for_uncertainty_calibration(file_name,prediction_samples,observations,targets,sigmas,id_test):
	"""
	Pickle provided data for future calibration compute
	Args:
		- test_name
		- prediction_samples
		- observations
		- targets
		- sigmas
		- id_test
	Returns:
	"""
	# make one data object
	data_for_calibration = {
		"PREDICTION_SAMPLES":   prediction_samples,
		"OBSERVATIONS":         observations,
		"TARGETS":              targets,
		"SIGMAS":               sigmas,
		"ID_TEST":              id_test
	}
	# Creates pickle directory if does not exists
	mkdir_p(PICKLE_DIR)
	pickle_out_name = os.path.join(PICKLE_DIR, file_name+"_calibration.pickle")
	logging.info("Writing data for uncertainty calibration into: "+pickle_out_name)
	pickle_out = open(pickle_out_name, "wb")
	pickle.dump(data_for_calibration, pickle_out, protocol=2)
	pickle_out.close()
	logging.info("Pickling data for uncertainty calibration...")

def get_data_for_calibration(test_name,calibration_proportion=0.8):
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
		- sigmas_samples
		- sigmas_samples_test
		- id_test
	"""
	logging.info("Unpickling data for uncertainty calibration...")
	pickle_in_name       = os.path.join(PICKLE_DIR, test_name+".pickle")
	pickle_in            = open(pickle_in_name, "rb")
	data_for_calibration = pickle.load(pickle_in)
	predictions,observations,groundtruth,sigmas_samples,id_test = data_for_calibration.values()
	# Split data randomly into calibration and test
	n_samples = predictions.shape[0]
	n_calibration = int(n_samples*calibration_proportion)
	indices       = random.sample(range(n_samples),n_calibration)
	# Calibration data
	predictions_calibration = predictions[indices]
	observations_calibration = observations[indices]
	groundtruth_calibration = groundtruth[indices]
	if sigmas_samples is not None:
		sigmas_samples_calibration = sigmas_samples[indices]
	else:
		sigmas_samples_calibration = None	
	# Test data: the rest
	predictions_test = predictions[[i for i in range(n_samples) if i not in indices]]
	observations_test = observations[[i for i in range(n_samples) if i not in indices]]
	groundtruth_test = groundtruth[[i for i in range(n_samples) if i not in indices]]
	if sigmas_samples is not None:
		sigmas_samples_test = sigmas_samples[[i for i in range(n_samples) if i not in indices]]
	else:
		sigmas_samples_test = None
	return predictions_calibration,predictions_test,observations_calibration,observations_test,groundtruth_calibration,groundtruth_test,sigmas_samples_calibration,sigmas_samples_test,id_test
