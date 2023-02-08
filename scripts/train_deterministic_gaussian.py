#!/usr/bin/env python
# coding: utf-8
# Autor: Mario Xavier Canche Uc
# Centro de Investigación en Matemáticas, A.C.
# mario.canche@cimat.mx

# Imports
import time
import sys,os,logging

sys.path.append('.')

import math,numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
import torch
from torchvision import transforms

# Local models
from models.lstm_encdec import lstm_encdec_gaussian
from utils.datasets_utils import get_dataset
from utils.train_utils import train
from utils.plot_utils import plot_traj_img, plot_traj_world, plot_cov_world
from utils.calibration import generate_uncertainty_evaluation_dataset
from utils.calibration_utils import save_data_for_calibration
from utils.directory_utils import mkdir_p
from utils.config import get_config
# Local constants
from utils.constants import IMAGES_DIR,TRAINING_CKPT_DIR, DETERMINISTIC_GAUSSIAN, SUBDATASETS_NAMES


# Parser arguments
config = get_config(argv=sys.argv[1:])

def main():
	# Printing parameters
	torch.set_printoptions(precision=2)
	# Loggin format
	logging.basicConfig(format='%(levelname)s: %(message)s',level=config.log_level)
	# Choose seed
	logging.info("Seed: {}".format(config.seed))
	torch.manual_seed(config.seed)
	torch.cuda.manual_seed(config.seed)
	random.seed(config.seed)
	np.random.seed(config.seed)
	# Device
	if torch.cuda.is_available():
		logging.info(torch.cuda.get_device_name(torch.cuda.current_device()))
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Get the data
	batched_train_data,batched_val_data,batched_test_data,homography,reference_image = get_dataset(config)
	model_name    = DETERMINISTIC_GAUSSIAN

	# Training
	if config.no_retrain==False:
		# Instanciate the model
		model = lstm_encdec_gaussian(in_size=2, embedding_dim=128, hidden_dim=256, output_size=2)
		model.to(device)
		# Train the model
		train(model,device,0,batched_train_data,batched_val_data,config,model_name)

	# Model instantiation
	model = lstm_encdec_gaussian(in_size=2, embedding_dim=128, hidden_dim=256, output_size=2)
	# Load the previously trained model
	model_filename = TRAINING_CKPT_DIR+"/"+model_name+"_"+str(SUBDATASETS_NAMES[config.id_dataset][config.id_test])+"_0.pth"
	logging.info("Loading {}".format(model_filename))
	model.load_state_dict(torch.load(model_filename))
	model.eval()
	model.to(device)


	output_dir = os.path.join(IMAGES_DIR)
	mkdir_p(output_dir)

	# Testing a random trajectory index in all batches
	ind_sample = np.random.randint(config.batch_size)
	for batch_idx, (datarel_test, targetrel_test, data_test, target_test) in enumerate(batched_test_data):
		fig, ax = plt.subplots(1,1,figsize=(12,12))

		if torch.cuda.is_available():
			datarel_test  = datarel_test.to(device)

		pred, sigmas = model.predict(datarel_test, dim_pred=12)
		# Plotting
		ind = np.minimum(ind_sample,pred.shape[0]-1)
		plot_traj_world(pred[ind,:,:],data_test[ind,:,:],target_test[ind,:,:],ax)
		plot_cov_world(pred[ind,:,:],sigmas[ind,:,:],data_test[ind,:,:],ax)
		plt.legend()
		plt.savefig(os.path.join(output_dir , "pred_dropout"+".pdf"))
		if config.show_plot:
			plt.show()
		plt.close()
		# Not display more than config.examples
		if batch_idx==config.examples-1:
			break

	#------------------ Generates sub-dataset for calibration evaluation ---------------------------
	datarel_test_full, targetrel_test_full, data_test_full, target_test_full, tpred_samples_full, sigmas_samples_full = generate_uncertainty_evaluation_dataset(batched_test_data, model, 1, model_name, config, device=device)
	#---------------------------------------------------------------------------------------------------------------

	# Producing data for uncertainty calibration
	for batch_idx, (datarel_test, targetrel_test, data_test, target_test) in enumerate(batched_test_data):

		tpred_samples  = []
		sigmas_samples = []

		if torch.cuda.is_available():
			datarel_test  = datarel_test.to(device)

		pred, sigmas = model.predict(datarel_test, dim_pred=12)
		tpred_samples.append(pred)
		sigmas_samples.append(sigmas)
		tpred_samples = np.array(tpred_samples)
		sigmas_samples = np.array(sigmas_samples)
		# Save these testing data for uncertainty calibration
		pickle_filename = model_name+"_"+str(SUBDATASETS_NAMES[config.id_dataset][config.id_test])
		save_data_for_calibration(pickle_filename, tpred_samples, tpred_samples_full, data_test, data_test_full, target_test, target_test_full, targetrel_test, targetrel_test_full, sigmas_samples, sigmas_samples_full, config.id_test)
		# Only the first batch is used as the calibration dataset
		break
if __name__ == "__main__":
	main()
