#!/usr/bin/env python
# coding: utf-8
# Autor: Mario Xavier Canche Uc
# Centro de Investigación en Matemáticas, A.C.
# mario.canche@cimat.mx

# Imports
import sys,random,logging
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
import torch

# Local models
from models.lstm_encdec import lstm_encdec_gaussian
from utils.datasets_utils import get_dataset
from utils.calibration import generate_uncertainty_evaluation_dataset
from utils.calibration_utils import save_data_for_calibration
from utils.plot_utils import plot_traj_img,plot_traj_world,plot_cov_world
from utils.train_utils import train
from utils.config import load_config,get_model_name
from utils.constants import SUBDATASETS_NAMES

# Load configuration file (conditional model)
config = load_config("deterministic_gaussian_ethucy.yaml")

def main():
	# Printing parameters
	torch.set_printoptions(precision=2)
	# Loggin format
	logging.basicConfig(format='%(levelname)s: %(message)s',level=config["misc"]["log_level"])
	# Device
	if torch.cuda.is_available():
		logging.info(torch.cuda.get_device_name(torch.cuda.current_device()))
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	batched_train_data,batched_val_data,batched_test_data,__,__ = get_dataset(config["dataset"])

	# Select random seeds
	seeds = np.random.choice(999999, config["misc"]["model_samples"],replace=False)
	logging.info("Seeds: {}".format(seeds))

	if config["train"]["no_retrain"]==False:
		# Train model for each seed
		for ind, seed in enumerate(seeds):
			# Seed added
			torch.manual_seed(seed)
			np.random.seed(seed)
			random.seed(seed)

			# Instanciate the model
			model = lstm_encdec_gaussian(config["model"])
			model.to(device)

			# Train the model
			logging.info("Training for seed: {}\t\t {}/{} ".format(seed,ind,len(seeds)))
			train(model,device,ind,batched_train_data,batched_val_data,config)

	# Instanciate the model
	model = lstm_encdec_gaussian(config["model"])
	model.to(device)

	ind_sample = np.random.randint(config["dataset"]["batch_size"])
	# Testing
	for batch_idx, (observations_vel,__,observations_abs,target_abs,__,__,__) in enumerate(batched_test_data):
		__, ax = plt.subplots(1,1,figsize=(12,12))

		# For each element of the ensemble
		for ind in range(config["misc"]["model_samples"]):
			# Load the previously trained model
			model_filename = config["train"]["save_dir"]+get_model_name(config,ensemble_id=ind)
			logging.info("Loading {}".format(model_filename))
			model.load_state_dict(torch.load(model_filename))
			model.eval()

			if torch.cuda.is_available():
				observations_vel  = observations_vel.to(device)
			predictions, sigmas = model.predict(observations_vel)
			# Plotting
			plot_traj_world(predictions[ind_sample,:,:],observations_abs[ind_sample,:,:],target_abs[ind_sample,:,:],ax)
			plot_cov_world(predictions[ind_sample,:,:],sigmas[ind_sample,:,:],observations_abs[ind_sample,:,:],ax)
		plt.title('Trajectory samples {}'.format(batch_idx))
		if config["misc"]["show_test"]:
			plt.show()
		# We just use the first batch to test
		break

	#------------------ Generates sub-dataset for calibration evaluation ---------------------------
	__,__,observations_abs_e,target_abs_e,predictions_e,sigmas_e = generate_uncertainty_evaluation_dataset(batched_test_data,model,config,device=device,type="ensemble")
	#---------------------------------------------------------------------------------------------------------------

	# Testing
	for batch_idx, (observations_vel_c,__,observations_abs_c,target_abs_c,__,__,__) in enumerate(batched_test_data):

		tpred_samples = []
		sigmas_samples = []
		# Muestreamos con cada modelo
		for ind in range(config["misc"]["model_samples"]):
			# Load the model from the ensemble
			model_filename = config["train"]["save_dir"]+get_model_name(config,ensemble_id=ind)
			logging.info("Loading {}".format(model_filename))
			model.load_state_dict(torch.load(model_filename))
			model.eval()

			if torch.cuda.is_available():
				observations_vel_c  = observations_vel_c.to(device)

			predictions, sigmas = model.predict(observations_vel_c)
			tpred_samples.append(predictions)
			sigmas_samples.append(sigmas)

		predictions_c = np.array(tpred_samples)
		sigmas_c      = np.array(sigmas_samples)
		break
	
	pickle_filename = config["train"]["model_name"]+"_ensemble_"+SUBDATASETS_NAMES[config["dataset"]["id_dataset"]][config["dataset"]["id_test"]]
	save_data_for_calibration(pickle_filename,predictions_c,predictions_e, observations_abs_c,observations_abs_e,target_abs_c,target_abs_e,sigmas_c,sigmas_e,config["dataset"]["id_test"])

if __name__ == "__main__":
	main()
