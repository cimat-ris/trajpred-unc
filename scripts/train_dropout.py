#!/usr/bin/env python
# coding: utf-8
# Autor: Mario Xavier Canche Uc
# Centro de Investigación en Matemáticas, A.C.
# mario.canche@cimat.mx

# Cargamos las librerias
import sys,os,logging

''' TF_CPP_MIN_LOG_LEVEL
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printeds
3 = INFO, WARNING, and ERROR messages are not printed
'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append('bayesian-torch')

import random,numpy as np
import matplotlib.pyplot as plt
import torch

# Local models
from trajpred_unc.models.bayesian_models_gaussian_loss import lstm_encdec_MCDropout
from trajpred_unc.utils.datasets_utils import get_dataset
from trajpred_unc.utils.plot_utils import plot_traj_world,plot_cov_world
from trajpred_unc.utils.train_utils import train, evaluation_minadefde
from trajpred_unc.utils.config import load_config,get_model_filename
from trajpred_unc.uncertainties.calibration import generate_uncertainty_evaluation_dataset
from trajpred_unc.uncertainties.calibration_utils import save_data_for_calibration

# Local constants
from trajpred_unc.utils.constants import IMAGES_DIR, DROPOUT, TRAINING_CKPT_DIR, SUBDATASETS_NAMES

# Load configuration file (conditional model)
config = load_config("deterministic_dropout_ethucy.yaml")

def main():
	# Printing parameters
	torch.set_printoptions(precision=2)
	# Loggin format
	logging.basicConfig(format='%(levelname)s: %(message)s',level=config["misc"]["log_level"])
	# Device
	if torch.cuda.is_available():
		logging.info(torch.cuda.get_device_name(torch.cuda.current_device()))
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	batched_train_data,batched_val_data,batched_test_data,homography,reference_image = get_dataset(config["dataset"])

	# Choose seed
	torch.manual_seed(config["misc"]["seed"])
	torch.cuda.manual_seed(config["misc"]["seed"])
	np.random.seed(config["misc"]["seed"])
	random.seed(config["misc"]["seed"])

	# Instantiate the model
	model = lstm_encdec_MCDropout(config["model"])
	model.to(device)

	if config["train"]["no_retrain"]==False:
		# Train the model
		train(model,device,0,batched_train_data,batched_val_data,config)

	# Load the previously trained model
	model_filename = config["train"]["save_dir"]+get_model_filename(config)
	logging.info("Loading {}".format(model_filename))
	model.load_state_dict(torch.load(model_filename))
	model.eval()

	# Testing
	ind_sample = np.random.randint(config["dataset"]["batch_size"])
	for batch_idx, (observations_vel,__,observations_abs,target_abs,__,__,__) in enumerate(batched_test_data):
		__, ax = plt.subplots(1,1,figsize=(12,12))
		if ind_sample>observations_vel.shape[0]:
			continue
		# Generate samples from the model
		for ind in range(config["misc"]["model_samples"]):
			if torch.cuda.is_available():
				observations_vel  = observations_vel.to(device)
			predicted_positions,sigmas_positions = model.predict(observations_vel)
			# Plotting
			ind = np.minimum(ind_sample,predicted_positions.shape[0]-1)
			plot_traj_world(predicted_positions[ind,:,:],observations_abs[ind,:,:],target_abs[ind,:,:],ax)
			plot_cov_world(predicted_positions[ind,:,:],sigmas_positions[ind,:,:],observations_abs[ind,:,:],ax)
		plt.legend()
		plt.title('Trajectory samples')
		if config["misc"]["show_test"]:
			plt.show()
		# Not display more than config.examples
		if batch_idx==config["misc"]["samples_test"]-1:
			break

	#------------------ Obtenemos el batch unico de test para las curvas de calibracion ---------------------------
	#------------------ Generates testing sub-dataset for uncertainty calibration and evaluation ---------------------------
	__,__,observations_abs_e,target_abs_e,predictions_e,sigmas_e = generate_uncertainty_evaluation_dataset(batched_test_data,model,config,device=device,type="dropout")
	# TODO: the samples should be sampled from the Gaussian mixture, not only the mean
	evaluation_minadefde(predictions_e, observations_abs_e, target_abs_e,config["train"]["model_name"]+"_"+SUBDATASETS_NAMES[config["dataset"]["id_dataset"]][config["dataset"]["id_test"]])
	#---------------------------------------------------------------------------------------------------------------

	# TODO: make this into a function (as the one above, generate_uncertainty_evaluation_dataset)
	# Testing
	for batch_idx, (observations_vel_c,__,observations_abs_c,target_abs_c,__,__,__) in enumerate(batched_test_data):

		tpred_samples = []
		sigmas_samples = []
		# Sampling from inference dropout
		for ind in range(config["misc"]["model_samples"]):
			if torch.cuda.is_available():
				observations_vel_c  = observations_vel_c.to(device)
			pred, sigmas = model.predict(observations_vel_c, dim_pred=12)
			tpred_samples.append(pred)
			sigmas_samples.append(sigmas)

		predictions_c = np.array(tpred_samples)
		sigmas_c      = np.array(sigmas_samples)
		break	
	# Save these testing data for uncertainty calibration
	pickle_filename = config["train"]["model_name"]+"_"+SUBDATASETS_NAMES[config["dataset"]["id_dataset"]][config["dataset"]["id_test"]]
	save_data_for_calibration(pickle_filename,predictions_c,predictions_e, observations_abs_c,observations_abs_e,target_abs_c,target_abs_e,sigmas_c,sigmas_e,config["dataset"]["id_test"])

if __name__ == "__main__":
	main()
