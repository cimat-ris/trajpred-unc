#!/usr/bin/env python
# coding: utf-8

# Imports
import time
import sys,os,logging, argparse
sys.path.append('.')
from os.path import exists

import math,numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

import torch
from torchvision import transforms
import torch.optim as optim
import cv2
# Local models
from models.lstm_encdec import lstm_encdec
from utils.datasets_utils import setup_loo_experiment,Experiment_Parameters,traj_dataset
from utils.plot_utils import plot_traj_img
from utils.train_utils import train

# Local constants
from utils.constants import OBS_TRAJ_VEL, PRED_TRAJ_VEL, OBS_TRAJ, PRED_TRAJ, TRAINING_CKPT_DIR, REFERENCE_IMG,FRAMES_IDS, SUBDATASETS_NAMES,DATASETS_DIR
from utils.config import get_config



# Parser arguments
config = get_config()

# Gets a testing batch of trajectories starting at the same frame (for visualization)
def get_testing_batch(testing_data,testing_data_path):
	# A trajectory id
	randomtrajId     = np.random.randint(len(testing_data),size=1)[0]
	# Last observed frame id for a random trajectory in the testing dataset
	frame_id         = testing_data.Frame_Ids[randomtrajId][7]
	idx              = np.where((testing_data.Frame_Ids[:,7]==frame_id))[0]
	# Get the video corresponding to the testing
	cap   = cv2.VideoCapture(testing_data_path+'/video.avi')
	frame = 0
	while(cap.isOpened()):
		ret, test_bckgd = cap.read()
		if frame == frame_id:
			break
		frame = frame + 1
	# Form the batch
	return frame_id, traj_dataset(*(testing_data[idx])), test_bckgd

def main():
	# Printing parameters
	torch.set_printoptions(precision=2)
	# Loggin format
	logging.basicConfig(format='%(levelname)s: %(message)s',level=config.log_level)
	# Device
	if torch.cuda.is_available():
		logging.info(torch.cuda.get_device_name(torch.cuda.current_device()))
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Load the default parameters
	experiment_parameters = Experiment_Parameters()

	# Load the dataset and perform the split
	training_data, validation_data, testing_data, test_homography = setup_loo_experiment(DATASETS_DIR[config.id_dataset],SUBDATASETS_NAMES[config.id_dataset],config.id_test,experiment_parameters,pickle_dir='pickle',use_pickled_data=config.pickle, validation_proportion=config.validation_proportion, compute_neighbors=not config.no_neighbors)
	test_data = traj_dataset(testing_data[OBS_TRAJ_VEL ], testing_data[PRED_TRAJ_VEL], testing_data[OBS_TRAJ], testing_data[PRED_TRAJ], testing_data[FRAMES_IDS])

	model_name = 'deterministic'

	# Model instantiation
	model = lstm_encdec(in_size=2, embedding_dim=128, hidden_dim=128, output_size=2)
	# Load the previously trained model
	model_filename = TRAINING_CKPT_DIR+"/"+model_name+"_"+str(SUBDATASETS_NAMES[config.id_dataset][config.id_test])+"_0.pth"
	if (not os.path.exists(model_filename)):
		return
	logging.info("Loading {}".format(model_filename))
	model.load_state_dict(torch.load(model_filename))
	model.to(device)
	model.eval()

	ind_sample = 1
	frame_id, batch, test_bckgd = get_testing_batch(test_data,DATASETS_DIR[config.id_dataset]+SUBDATASETS_NAMES[config.id_dataset][config.id_test])
	# Form batches
	batched_test_data  = torch.utils.data.DataLoader(batch,batch_size=len(batch))
	n_trajs            = len(batch)
	# Testing: Qualitative. Should be only one batch
	for (datarel_test, targetrel_test, data_test, target_test) in batched_test_data:
		if torch.cuda.is_available():
			datarel_test  = datarel_test.to(device)
		# Prediction
		pred = model.predict(datarel_test, dim_pred=12)
		# Plotting
		plt.figure(figsize=(12,12))
		plt.imshow(test_bckgd)
		for k in range(n_trajs):
			plot_traj_img(pred[k,:,:], data_test[k,:,:], target_test[k,:,:], test_homography, test_bckgd)
		plt.title('Trajectory samples, frame {}'.format(frame_id))
		plt.show()

	# Testing: Quantitative. Over the batch defined above
	ade  = 0
	fde  = 0
	total= 0
	for batch_idx, (datavel_test, targetvel_test, data_test, target_test) in    enumerate(batched_test_data):
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
