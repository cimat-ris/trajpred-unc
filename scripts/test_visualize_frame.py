#!/usr/bin/env python
# coding: utf-8

# Imports
import os,logging

import math,numpy as np
import matplotlib.pyplot as plt

import torch
import cv2
# Local models
from trajpred_unc.models.lstm_encdec import lstm_encdec_gaussian
from trajpred_unc.utils.datasets_utils import setup_loo_experiment,traj_dataset,get_testing_batch,collate_fn_padd
from trajpred_unc.utils.plot_utils import plot_traj_img
from trajpred_unc.utils.train_utils import train

# Local constants
from trajpred_unc.utils.constants import OBS_TRAJ_VEL, PRED_TRAJ_VEL, OBS_TRAJ, PRED_TRAJ, TRAINING_CKPT_DIR, REFERENCE_IMG,FRAMES_IDS, SUBDATASETS_NAMES,DATASETS_DIR
from trajpred_unc.utils.config import load_config, get_model_filename



# Parser arguments
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


	# Load the dataset and perform the split
	__,__,testing_data, test_homography = setup_loo_experiment(config["dataset"])
	test_data = traj_dataset(testing_data[OBS_TRAJ_VEL ], testing_data[PRED_TRAJ_VEL], testing_data[OBS_TRAJ], testing_data[PRED_TRAJ], Frame_Ids=testing_data[FRAMES_IDS])
	frame_id, batch, test_bckgd = get_testing_batch(test_data,config["dataset"])

	# Model instantiation
	model = lstm_encdec_gaussian(config["model"])
	# Load the previously trained model
	model_filename = config["train"]["save_dir"]+get_model_filename(config,ensemble_id=0)
	if (not os.path.exists(model_filename)):
		return
	logging.info("Loading {}".format(model_filename))
	model.load_state_dict(torch.load(model_filename))
	model.to(device)
	model.eval()

	ind_sample = 1
	# Form batches
	batched_test_data  = torch.utils.data.DataLoader(batch,batch_size=len(batch),collate_fn=collate_fn_padd)
	n_trajs            = len(batch)
	# Testing: Qualitative. Should be only one batch
	for batch_idx, (observations_vel_c,__,observations_abs_c,target_abs_c,__,__,__) in enumerate(batched_test_data):
		if torch.cuda.is_available():
			observations_vel_c  = observations_vel_c.to(device)
		# Prediction
		predictions,__ = model.predict(observations_vel_c)
		# Plotting
		plt.figure(figsize=(12,12))
		plt.imshow(test_bckgd)
		for k in range(n_trajs):
			plot_traj_img(predictions[k,:,:], observations_abs_c[k,:,:], target_abs_c[k,:,:], test_homography, test_bckgd)
		plt.title('Trajectory samples, frame {}'.format(frame_id))
		plt.show()

	# Testing: Quantitative. Over the batch defined above
	ade  = 0
	fde  = 0
	total= 0
	for batch_idx, (observations_vel_c,__,observations_abs_c,target_abs_c,__,__,__) in enumerate(batched_test_data):
		if torch.cuda.is_available():
			observations_vel_c  = observations_vel_c.to(device)
		total += len(observations_vel_c)
		# prediction
		init_pos         = np.expand_dims(observations_abs_c[:,-1,:],axis=1)
		predictions,__   = model.predict(observations_vel_c) 
		predictions     += init_pos
		ade    += np.average(np.sqrt(np.square(target_abs_c-predictions).sum(2)),axis=1).sum()
		fde    += (np.sqrt(np.square(target_abs_c[:,-1,:]-predictions[:,-1,:]).sum(1))).sum()
	logging.info("Test ade : {:.4f} ".format(ade/total))
	logging.info("Test fde : {:.4f} ".format(fde/total))
if __name__ == "__main__":
	main()
