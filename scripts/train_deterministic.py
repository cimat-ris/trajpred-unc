#!/usr/bin/env python
# coding: utf-8

# Imports
import sys,random,logging
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
import torch

# Local models
from models.lstm_encdec import lstm_encdec
from utils.datasets_utils import get_dataset
from utils.plot_utils import plot_traj_img
from utils.train_utils import train
from utils.config import load_config,get_model_name

# Load configuation file (conditional model)
config = load_config("deterministic_ethucy.yaml")

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

	# Model instantiation
	model = lstm_encdec(config["model"])
	model.to(device)

	# May not have to retrain the model
	if config["train"]["no_retrain"]==False:
		# Train the model
		train(model,device,0,batched_train_data,batched_val_data,config)

	# Load the previously trained model
	model_filename = config["train"]["save_dir"]+get_model_name(config)
	logging.info("Loading {}".format(model_filename))
	model.load_state_dict(torch.load(model_filename))
	model.to(device)
	model.eval()

	ind_sample = 1

	# Testing
	for batch_idx, (datarel_test,__,data_test,target_test,__,__,__) in enumerate(batched_test_data):
		if torch.cuda.is_available():
			datarel_test  = datarel_test.to(device)

		# Prediction
		pred = model.predict(datarel_test, dim_pred=12)

		# Plotting
		plt.figure(figsize=(12,12))
		plt.imshow(reference_image)
		plot_traj_img(pred[ind_sample,:,:],data_test[ind_sample,:,:],target_test[ind_sample,:,:],homography,reference_image)
		plt.legend()
		plt.title('Trajectory samples {}'.format(batch_idx))
		if config["misc"]["show_test"]:
			plt.show()
		# Not display more than args.examples
		if batch_idx==config["misc"]["samples_test"]-1:
			break

	# Testing: Quantitative
	ade  = fde = total = 0
	for batch_idx, (datavel_test,__, data_test, target_test,__,__,__) in enumerate(batched_test_data):
		if torch.cuda.is_available():
			datavel_test  = datavel_test.to(device)
		total += len(datavel_test)
		# prediction
		init_pos  = np.expand_dims(data_test[:,-1,:],axis=1)
		pred_test = model.predict(datavel_test, dim_pred=12) + init_pos
		ade    += np.average(np.sqrt(np.square(target_test-pred_test).sum(2)),axis=1).sum()
		fde    += (np.sqrt(np.square(target_test[:,-1,:]-pred_test[:,-1,:]).sum(1))).sum()
	logging.info("Test ade : {:.4f} ".format(ade/total))
	logging.info("Test fde : {:.4f} ".format(fde/total))
if __name__ == "__main__":
	main()
