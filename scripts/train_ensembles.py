#!/usr/bin/env python
# coding: utf-8
# Autor: Mario Xavier Canche Uc
# Centro de Investigación en Matemáticas, A.C.
# mario.canche@cimat.mx

# Imports
import random,logging
import numpy as np
import matplotlib.pyplot as plt
import torch

# Local models
from trajpred_unc.models.lstm_encdec import lstm_encdec_gaussian
from trajpred_unc.utils.datasets_utils import get_dataset
from trajpred_unc.uncertainties.calibration import generate_uncertainty_evaluation_dataset
from trajpred_unc.uncertainties.calibration_utils import save_data_for_uncertainty_calibration
from trajpred_unc.utils.plot_utils import plot_traj_world,plot_cov_world
from trajpred_unc.utils.train_utils import train
from trajpred_unc.utils.config import load_config,get_model_filename
from trajpred_unc.utils.constants import SUBDATASETS_NAMES

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
	seeds = np.random.choice(9999, config["misc"]["model_samples"],replace=False)
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
	for batch_idx, (observations,target,__,__,__) in enumerate(batched_test_data):
		__, ax = plt.subplots(1,1,figsize=(12,12))

		# For each element of the ensemble
		for ind in range(config["misc"]["model_samples"]):
			# Load the previously trained model
			model_filename = config["train"]["save_dir"]+get_model_filename(config,ensemble_id=ind)
			logging.info("Loading {}".format(model_filename))
			model.load_state_dict(torch.load(model_filename))
			model.eval()

			if torch.cuda.is_available():
				observations  = observations.to(device)
			predictions, sigmas = model.predict(observations[:,:,2:4],observations[:,:,:2])
			# Plotting
			plot_traj_world(predictions[ind_sample,:,:],observations[ind_sample,:,:2].cpu(),target[ind_sample,:,:2].cpu(),ax)
			plot_cov_world(predictions[ind_sample,:,:],sigmas[ind_sample,:,:],observations[ind_sample,:,:2].cpu(),ax)
		plt.title('Trajectory samples {}'.format(batch_idx))
		if config["misc"]["show_test"]:
			plt.show()
		# We just use the first batch to test
		break

	#------------------ Generates sub-dataset for calibration evaluation ---------------------------
	observations,target,predictions,sigmas = generate_uncertainty_evaluation_dataset(batched_test_data,model,config,device=device,type="ensemble")
	#---------------------------------------------------------------------------------------------------------------
	
	# Save these testing data for uncertainty calibration
	pickle_filename = config["train"]["model_name"]+"_ensemble_"+SUBDATASETS_NAMES[config["dataset"]["id_dataset"]][config["dataset"]["id_test"]]
	save_data_for_uncertainty_calibration(pickle_filename,predictions,observations,target,sigmas,config["dataset"]["id_test"])

if __name__ == "__main__":
	main()
