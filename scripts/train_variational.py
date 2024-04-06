#!/usr/bin/env python
# coding: utf-8
# Autor: Mario Xavier Canche Uc
# Centro de Investigación en Matemáticas, A.C.
# mario.canche@cimat.mx

# Imports
import time
import sys,os,logging, argparse
''' TF_CPP_MIN_LOG_LEVEL
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printeds
3 = INFO, WARNING, and ERROR messages are not printed
'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append('bayesian-torch')


import math,numpy as np,random
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

import torch
# Local models
from trajpred_unc.models.bayesian_models_gaussian_loss import lstm_encdec_variational
from trajpred_unc.utils.datasets_utils import get_dataset
from trajpred_unc.utils.train_utils import train_variational
from trajpred_unc.utils.plot_utils import plot_traj_world, plot_cov_world
# Local constants
from trajpred_unc.utils.constants import IMAGES_DIR, VARIATIONAL, TRAINING_CKPT_DIR, SUBDATASETS_NAMES
from trajpred_unc.utils.config import load_config,get_model_filename
from trajpred_unc.uncertainties.calibration import generate_uncertainty_evaluation_dataset
from trajpred_unc.uncertainties.calibration_utils import save_data_for_calibration

# Parser arguments
config = load_config("deterministic_variational_ethucy.yaml")

def main():
	# Printing parameters
	torch.set_printoptions(precision=2)
	logging.basicConfig(format='%(levelname)s: %(message)s',level=config["misc"]["log_level"])
	# Set seed
	torch.manual_seed(config["misc"]["seed"])
	torch.cuda.manual_seed(config["misc"]["seed"])
	np.random.seed(config["misc"]["seed"])
	random.seed(config["misc"]["seed"])
	# Device
	if torch.cuda.is_available():
		logging.info(torch.cuda.get_device_name(torch.cuda.current_device()))
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	batched_train_data,batched_val_data,batched_test_data,homography,reference_image = get_dataset(config["dataset"])
	model_name    = "deterministic_variational"

	# TODO: Agregar la los argumentos
	prior_mu = 0.0
	prior_sigma = 1.0
	posterior_mu_init = 0.0
	posterior_rho_init = -4

	# Instanciate the model
	model = lstm_encdec_variational(2,128,256,2,prior_mu,prior_sigma,posterior_mu_init,posterior_rho_init)
	model.to(device)

	if config["train"]["no_retrain"]==False:
		# Train the model
		train_variational(model,device,batched_train_data,batched_val_data,config)

	# Load the previously trained model
	model_filename = config["train"]["save_dir"]+get_model_filename(config)
	logging.info("Loading {}".format(model_filename))
	model.load_state_dict(torch.load(model_filename))
	model.eval()

	# Creamos la carpeta donde se guardaran las imagenes
	if not os.path.exists(IMAGES_DIR):
		os.makedirs(IMAGES_DIR)

	# Testing
	ind_sample = np.random.randint(config["dataset"]["batch_size"])
	for batch_idx, (observations_vel,__,observations_abs,target_abs,__,__,__) in enumerate(batched_test_data):
		__, ax = plt.subplots(1,1,figsize=(12,12))

		# For each element of the ensemble
		for ind in range(config["train"]["num_mctrain"]):

			if torch.cuda.is_available():
				observations_vel  = observations_vel.to(device)

			predictions,__,sigmas = model.predict(observations_vel)

			# Plotting
			plot_traj_world(predictions[ind_sample,:,:], observations_abs[ind_sample,:,:], target_abs[ind_sample,:,:], ax)
			plot_cov_world(predictions[ind_sample,:,:],sigmas[ind_sample,:,:],observations_abs[ind_sample,:,:], ax)
		plt.legend()
		plt.title('Trajectory samples {}'.format(batch_idx))
		plt.savefig(IMAGES_DIR+"/pred_variational.pdf")
		if config["misc"]["show_test"]:
			plt.show()
		plt.close()
		# Solo aplicamos a un elemento del batch
		break


	# ## Calibramos la incertidumbre
	draw_ellipse = True

	#------------------ Generates sub-dataset for calibration evaluation ---------------------------
	__,__,observations_abs_e,target_abs_e,predictions_e,sigmas_e = generate_uncertainty_evaluation_dataset(batched_test_data, model,config,device=device,type="variational")
	#---------------------------------------------------------------------------------------------------------------

	# Testing
	cont = 0
	for batch_idx, (observations_vel_c,__,observations_abs_c,target_abs_c,__,__,__) in enumerate(batched_test_data):

		predictions_c = []
		sigmas_c = []
		# Muestreamos con cada modelo
		for ind in range(config["train"]["num_mctrain"]):

			if torch.cuda.is_available():
				observations_vel_c  = observations_vel_c.to(device)

			predictions, kl, sigmas = model.predict(observations_vel_c)

			predictions_c.append(predictions)
			sigmas_c.append(sigmas)

		predictions_c = np.array(predictions_c)
		sigmas_c = np.array(sigmas_c)

		# Save these testing data for uncertainty calibration
		pickle_filename = config["train"]["model_name"]+"_"+SUBDATASETS_NAMES[config["dataset"]["id_dataset"]][config["dataset"]["id_test"]]
		#save_data_for_calibration(pickle_filename, tpred_samples, tpred_samples_full, data_test, data_test_full, target_test, target_test_full, sigmas_samples, sigmas_samples_full, config.id_test)
		save_data_for_calibration(pickle_filename,predictions_c,predictions_e, observations_abs_c,observations_abs_e,target_abs_c,target_abs_e,sigmas_c,sigmas_e,config["dataset"]["id_test"])


		# Solo se ejecuta para un batch
		break


if __name__ == "__main__":
	main()
