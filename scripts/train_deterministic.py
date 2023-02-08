#!/usr/bin/env python
# coding: utf-8

# Imports
import time, random
import sys,os,logging, argparse
sys.path.append('.')

import math,numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import torch
from torchvision import transforms

# Local models
from models.lstm_encdec import lstm_encdec
from utils.datasets_utils import get_dataset
from utils.plot_utils import plot_traj_img
from utils.train_utils import train
from utils.config import get_config

# Local constants
from utils.constants import TRAINING_CKPT_DIR,SUBDATASETS_NAMES


# Parser arguments
config = get_config(argv=sys.argv[1:])

def main():
	# Printing parameters
	torch.set_printoptions(precision=2)
	# Loggin format
	logging.basicConfig(format='%(levelname)s: %(message)s',level=config.log_level)
	# Device
	if torch.cuda.is_available():
		logging.info(torch.cuda.get_device_name(torch.cuda.current_device()))
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	batched_train_data,batched_val_data,batched_test_data,homography,reference_image = get_dataset(config)
	model_name    = 'deterministic'

	# Choose seed
	torch.manual_seed(config.seed)
	torch.cuda.manual_seed(config.seed)
	np.random.seed(config.seed)
	random.seed(config.seed)

	# Model instantiation
	model = lstm_encdec(in_size=2, embedding_dim=128, hidden_dim=128, output_size=2)
	model.to(device)

	# Seed for RNG
	if config.no_retrain==False:
		# Train the model
		train(model,device,0,batched_train_data,batched_val_data,config,model_name)

	# Load the previously trained model
	model_filename = TRAINING_CKPT_DIR+"/"+model_name+"_"+str(SUBDATASETS_NAMES[config.id_dataset][config.id_test])+"_0.pth"
	logging.info("Loading {}".format(model_filename))
	model.load_state_dict(torch.load(model_filename))
	model.to(device)
	model.eval()

	ind_sample = 1

	# Testing
	for batch_idx, (datarel_test, targetrel_test, data_test, target_test) in    enumerate(batched_test_data):
		if torch.cuda.is_available():
			datarel_test  = datarel_test.to(device)

		# Prediction
		pred = model.predict(datarel_test, dim_pred=12)

		# Plotting
		plt.figure(figsize=(12,12))
		plt.imshow(reference_image)
		plot_traj_img(pred[ind_sample,:,:], data_test[ind_sample,:,:], target_test[ind_sample,:,:], homography, reference_image)
		plt.legend()
		plt.title('Trajectory samples {}'.format(batch_idx))
		if config.show_plot:
			plt.show()
		# Not display more than args.examples
		if batch_idx==config.examples-1:
			break

	# Testing: Quantitative
	ade  = 0
	fde  = 0
	total= 0
	for batch_idx, (datavel_test, targetvel_test, data_test, target_test) in enumerate(batched_test_data):
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
