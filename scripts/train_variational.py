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
sys.path.append('.')

from utils.calibration_utils import save_data_for_calibration

import math,numpy as np,random
import matplotlib as mpl
#mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
import pandas as pd

import torch
from torchvision import transforms

# Local models
from models.bayesian_models_gaussian_loss import lstm_encdec_variational
from utils.datasets_utils import get_dataset
from utils.train_utils import train_variational
from utils.plot_utils import plot_traj_img, plot_traj_world, plot_cov_world
from utils.calibration import generate_uncertainty_evaluation_dataset
from utils.directory_utils import mkdir_p
from utils.config import get_config

# Local constants
from utils.constants import IMAGES_DIR, VARIATIONAL, TRAINING_CKPT_DIR, SUBDATASETS_NAMES

# Parser arguments
config = get_config(variational=True)

def main():
	# Printing parameters
	torch.set_printoptions(precision=2)
	logging.basicConfig(format='%(levelname)s: %(message)s',level=config.log_level)
	# Set seed
	logging.info("Seed: {}".format(config.seed))
	torch.manual_seed(config.seed)
	random.seed(config.seed)
	np.random.seed(config.seed)
	# Device
	if torch.cuda.is_available():
		logging.info(torch.cuda.get_device_name(torch.cuda.current_device()))
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	batched_train_data,batched_val_data,batched_test_data,homography,reference_image = get_dataset(config)
	model_name    = "deterministic_variational"

	# TODO: Agregar la los argumentos
	prior_mu = 0.0
	prior_sigma = 1.0
	posterior_mu_init = 0.0
	posterior_rho_init = -4

	# Instanciate the model
	model = lstm_encdec_variational(2,128,256,2,prior_mu,prior_sigma,posterior_mu_init,posterior_rho_init)
	model.to(device)

	if config.no_retrain==False:
		# Train the model
		train_variational(model,device,config.id_test,batched_train_data,batched_val_data,config,model_name)
		if config.plot_losses:
			plt.savefig(IMAGES_DIR+"/loss_"+str(config.id_test)+".pdf")
			plt.show()


	# Load the previously trained model
	model_filename = TRAINING_CKPT_DIR+"/"+model_name+"_"+str(SUBDATASETS_NAMES[config.id_dataset][config.id_test])+"_0.pth"
	logging.info("Loading {}".format(model_filename))
	model.load_state_dict(torch.load(model_filename))
	model.eval()

	# Creamos la carpeta donde se guardaran las imagenes
	if not os.path.exists(IMAGES_DIR):
		os.makedirs(IMAGES_DIR)

	# Testing
	ind_sample = np.random.randint(config.batch_size)
	for batch_idx, (datarel_test, targetrel_test, data_test, target_test) in enumerate(batched_test_data):
		fig, ax = plt.subplots(1,1,figsize=(12,12))

		# For each element of the ensemble
		for ind in range(config.num_mctest):

			if torch.cuda.is_available():
				  datarel_test  = datarel_test.to(device)

			pred, kl, sigmas = model.predict(datarel_test, dim_pred=12)

			# Plotting
			plot_traj_world(pred[ind_sample,:,:], data_test[ind_sample,:,:], target_test[ind_sample,:,:], ax)
			plot_cov_world(pred[ind_sample,:,:],sigmas[ind_sample,:,:],data_test[ind_sample,:,:], ax)
		plt.legend()
		plt.title('Trajectory samples {}'.format(batch_idx))
		plt.savefig(IMAGES_DIR+"/pred_variational.pdf")
		if config.show_plot:
			plt.show()
		plt.close()
		# Solo aplicamos a un elemento del batch
		break


	# ## Calibramos la incertidumbre
	draw_ellipse = True

	#------------------ Generates sub-dataset for calibration evaluation ---------------------------
	datarel_test_full, targetrel_test_full, data_test_full, target_test_full, tpred_samples_full, sigmas_samples_full = generate_uncertainty_evaluation_dataset(batched_test_data, model, 1, model_name, config, type="variational",device=device)
	#---------------------------------------------------------------------------------------------------------------

	# Testing
	cont = 0
	for batch_idx, (datarel_test, targetrel_test, data_test, target_test) in enumerate(batched_test_data):

		tpred_samples = []
		sigmas_samples = []
		# Muestreamos con cada modelo
		for ind in range(config.num_mctest):

			if torch.cuda.is_available():
				  datarel_test  = datarel_test.to(device)

			pred, kl, sigmas = model.predict(datarel_test, dim_pred=12)

			tpred_samples.append(pred)
			sigmas_samples.append(sigmas)

		tpred_samples = np.array(tpred_samples)
		sigmas_samples = np.array(sigmas_samples)

		# Save these testing data for uncertainty calibration
		pickle_filename = model_name+"_"+str(SUBDATASETS_NAMES[config.id_dataset][config.id_test])
		save_data_for_calibration(pickle_filename, tpred_samples, tpred_samples_full, data_test, data_test_full, target_test, target_test_full, targetrel_test, targetrel_test_full, sigmas_samples, sigmas_samples_full, config.id_test)


		# Solo se ejecuta para un batch
		break


if __name__ == "__main__":
	main()
