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
from trajpred_unc.utils.train_utils import train
from trajpred_unc.utils.evaluation import evaluation_minadefde
from trajpred_unc.utils.config import load_config,get_model_filename
from trajpred_unc.uncertainties.calibration import generate_uncertainty_evaluation_dataset
from trajpred_unc.uncertainties.calibration_utils import save_data_for_uncertainty_calibration

# Local constants
from trajpred_unc.utils.constants import SUBDATASETS_NAMES

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
	for batch_idx, (observations,target,__,__,__) in enumerate(batched_test_data):
		__, ax = plt.subplots(1,1,figsize=(12,12))
		if ind_sample>observations.shape[0]:
			continue
		# Generate samples from the model
		for ind in range(config["misc"]["model_samples"]):
			if torch.cuda.is_available():
				observations  = observations.to(device)
			predicted_positions,sigmas_positions = model.predict(observations[:,:,2:4],observations[:,:,:2])
			# Plotting
			ind = np.minimum(ind_sample,predicted_positions.shape[0]-1)
			plot_traj_world(predicted_positions[ind,:,:],observations[ind,:,:2].cpu(),target[ind,:,:2].cpu(),ax)
			plot_cov_world(predicted_positions[ind,:,:],sigmas_positions[ind,:,:],observations[ind,:,:2].cpu(),ax)
		plt.legend()
		plt.title('Trajectory samples')
		if config["misc"]["show_test"]:
			plt.show()
		# Not display more than config.examples
		if batch_idx==config["misc"]["samples_test"]-1:
			break

	#------------------ Generates testing sub-dataset for uncertainty calibration and evaluation ---------------------------
	observations,target,predictions,sigmas = generate_uncertainty_evaluation_dataset(batched_test_data,model,config,device=device,type="dropout")
	# Evaluates the minade and minfde based on means of the samples
	evaluation_minadefde(predictions,target,config["train"]["model_name"]+"_"+SUBDATASETS_NAMES[config["dataset"]["id_dataset"]][config["dataset"]["id_test"]])
	
	#---------------------------------------------------------------------------------------------------------------
	# Save these testing data for uncertainty calibration
	pickle_filename = config["train"]["model_name"]+"_"+SUBDATASETS_NAMES[config["dataset"]["id_dataset"]][config["dataset"]["id_test"]]
	save_data_for_uncertainty_calibration(pickle_filename,predictions,observations,target,sigmas,config["dataset"]["id_test"])

if __name__ == "__main__":
	main()
