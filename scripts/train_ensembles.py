#!/usr/bin/env python
# coding: utf-8
# Autor: Mario Xavier Canche Uc
# Centro de Investigación en Matemáticas, A.C.
# mario.canche@cimat.mx

# Imports
import time
import sys,os,logging,argparse,random

sys.path.append('bayesian-torch')
sys.path.append('.')
from utils.calibration_utils import save_data_for_calibration

import math,numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import torch
from torchvision import transforms

# Local models
from models.lstm_encdec import lstm_encdec_gaussian
from utils.datasets_utils import get_dataset
from utils.train_utils import train
from utils.plot_utils import plot_traj_img,plot_traj_world,plot_cov_world
from utils.calibration import generate_uncertainty_evaluation_dataset
from utils.config import get_config
# Local constants
from utils.constants import REFERENCE_IMG, ENSEMBLES, TRAINING_CKPT_DIR, SUBDATASETS_NAMES

# Parser arguments
config = get_config(argv=sys.argv[1:],ensemble=True)


def main():
	# Printing parameters
	torch.set_printoptions(precision=2)
	logging.basicConfig(format='%(levelname)s: %(message)s',level=config.log_level)
	# Set the seed
	logging.info("Seed: {}".format(config.seed))
	torch.manual_seed(config.seed)
	np.random.seed(config.seed)
	random.seed(config.seed)
	# Device
	if torch.cuda.is_available():
		logging.info(torch.cuda.get_device_name(torch.cuda.current_device()))
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Get the ETH-UCY data
	batched_train_data,batched_val_data,batched_test_data,homography,reference_image = get_dataset(config)
	model_name    = 'deterministic_gaussian_ensemble'

	# Select random seeds
	seeds = np.random.choice(99999999, config.num_ensembles , replace=False)
	logging.info("Seeds: {}".format(seeds))

	if config.no_retrain==False:
		# Train model for each seed
		for ind, seed in enumerate(seeds):
			# Seed added
			torch.manual_seed(seed)
			np.random.seed(seed)
			random.seed(seed)

			# Instanciate the model
			model = lstm_encdec_gaussian(in_size=2, embedding_dim=128, hidden_dim=256, output_size=2)
			model.to(device)

			# Train the model
			logging.info("Training for seed: {}\t\t {}/{} ".format(seed,ind,len(seeds)))
			train(model,device,ind,batched_train_data,batched_val_data,config,model_name)

	# Instanciate the model
	model = lstm_encdec_gaussian(in_size=2, embedding_dim=128, hidden_dim=256, output_size=2)
	model.to(device)


	ind_sample = np.random.randint(config.batch_size)

	# Testing
	for batch_idx, (datarel_test, targetrel_test, data_test, target_test) in enumerate(batched_test_data):
		fig, ax = plt.subplots(1,1,figsize=(12,12))

		# For each element of the ensemble
		for ind in range(config.num_ensembles):
			# Load the previously trained model
			model_filename = TRAINING_CKPT_DIR+"/"+model_name+"_"+str(SUBDATASETS_NAMES[config.id_dataset][config.id_test])+"_"+str(ind)+".pth"
			logging.info("Loading {}".format(model_filename))
			model.load_state_dict(torch.load(model_filename))
			model.eval()

			if torch.cuda.is_available():
				  datarel_test  = datarel_test.to(device)

			pred, sigmas = model.predict(datarel_test, dim_pred=12)
			# Plotting
			plot_traj_world(pred[ind_sample,:,:],data_test[ind_sample,:,:],target_test[ind_sample,:,:],ax)
			plot_cov_world(pred[ind_sample,:,:],sigmas[ind_sample,:,:],data_test[ind_sample,:,:],ax)
		plt.legend()
		plt.title('Trajectory samples {}'.format(batch_idx))
		if config.show_plot:
			plt.show()
		# We just use the first batch to test
		break

	#------------------ Generates sub-dataset for calibration evaluation ---------------------------
	datarel_test_full, targetrel_test_full, data_test_full, target_test_full, tpred_samples_full, sigmas_samples_full = generate_uncertainty_evaluation_dataset(batched_test_data, model, config.num_ensembles, model_name, config, device=device)
	#---------------------------------------------------------------------------------------------------------------

	# Testing
	cont = 0
	for batch_idx, (datarel_test, targetrel_test, data_test, target_test) in enumerate(batched_test_data):

		tpred_samples = []
		sigmas_samples = []
		# Muestreamos con cada modelo
		for ind in range(config.num_ensembles):

			# Cargamos el Modelo
			model_filename = TRAINING_CKPT_DIR+"/"+model_name+"_"+str(SUBDATASETS_NAMES[config.id_dataset][config.id_test])+"_"+str(ind)+".pth"
			logging.info("Loading {}".format(model_filename))
			model.load_state_dict(torch.load(model_filename))
			model.eval()

			if torch.cuda.is_available():
				  datarel_test  = datarel_test.to(device)

			pred, sigmas = model.predict(datarel_test, dim_pred=12)

			tpred_samples.append(pred)
			sigmas_samples.append(sigmas)

		tpred_samples = np.array(tpred_samples)
		sigmas_samples = np.array(sigmas_samples)
		# Save these testing data for uncertainty calibration
		pickle_filename = ENSEMBLES+"_"+str(SUBDATASETS_NAMES[config.id_dataset][config.id_test])
		save_data_for_calibration(pickle_filename, tpred_samples, tpred_samples_full, data_test, data_test_full, target_test, target_test_full, sigmas_samples, sigmas_samples_full, config.id_test)

		# Only the first batch is used as the calibration dataset
		break


if __name__ == "__main__":
	main()
