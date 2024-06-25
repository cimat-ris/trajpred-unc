#!/usr/bin/env python
# coding: utf-8

# Imports
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch

import os,sys
sys.path.append('../')
sys.path.append('../SocialVAE')

from SocialVAE.social_vae import SocialVAE
from SocialVAE.data import Dataloader
from SocialVAE.utils import ADE_FDE, FPC, seed, get_rng_state, set_rng_state


# Local models
from trajpred_unc.utils.datasets_utils import setup_loo_experiment,get_testing_batch,get_dataset,traj_dataset,collate_fn_padd
from trajpred_unc.utils.train_utils import train
from trajpred_unc.uncertainties.kde import plot_kde_img
from trajpred_unc.uncertainties.calibration import generate_uncertainty_evaluation_dataset,regression_isotonic_fit,recalibrate_and_test
from trajpred_unc.uncertainties.calibration_utils import save_data_for_uncertainty_calibration,get_data_for_calibration
from trajpred_unc.utils.config import load_config, get_model_filename
# Local constants
from trajpred_unc.utils.constants import (
	FRAMES_IDS, KEY_IDX, OBS_NEIGHBORS, OBS_TRAJ, PRED_TRAJ,FRAMES_IDS)

# Load configuration file (conditional model)
config = load_config("socialvae_ethucy.yaml")

def main():
	# Printing parameters
	torch.set_printoptions(precision=2)
	# Loggin format
	logging.basicConfig(format='%(levelname)s: %(message)s',level=config["misc"]["log_level"])
	# Device
	if torch.cuda.is_available():
		logging.info(torch.cuda.get_device_name(torch.cuda.current_device()))
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	__, __, testing_data, test_homography = setup_loo_experiment(config["dataset"])
	test_data = traj_dataset(testing_data[OBS_TRAJ], testing_data[PRED_TRAJ], Frame_Ids=testing_data[FRAMES_IDS])
	frame_id, batch, test_bckgd = get_testing_batch(test_data,config["dataset"])
	batched_test_data  = torch.utils.data.DataLoader(batch,batch_size=len(batch),collate_fn=collate_fn_padd)

	# Get the homography
	homography_to_img = np.linalg.inv(test_homography)

	logging.info("Loading model")
	model = SocialVAE(horizon=config["dataset"]["pred_len"], ob_radius=2.0, hidden_dim=256)
	model.to(device)
	path_ckpt = "../SocialVAE/models/zara01/"
	ckpt = os.path.join(path_ckpt, "ckpt-best")	
	logging.info("Load from ckpt: {}".format(ckpt))
	if os.path.exists(ckpt):
		state_dict = torch.load(ckpt, map_location=device)
		model.load_state_dict(state_dict["model"])

	# Cycle over the batch
	for (observations,targets,__,__,__) in batched_test_data:
		logging.info("Trajectories {}".format(len(observations)))
		# Cycle over the trajectories of this batch
		for traj_idx in range(len(observations)):
			# Output for each element of the ensemble
			predictions = []
			sigmas      = []
			for idx in range(config["misc"]["model_samples"]):
				if torch.cuda.is_available():
					observations = observations.to(device)
				# TODO
				print(observations.shape)
				model(observations, neighbor, n_predictions=config.PRED_SAMPLES)
				#prediction, sigma = models[idx].predict(observations_vel,observations_pos)
				#predictions.append(prediction[traj_idx]),sigmas.append(sigma[traj_idx])
			# Sampling 1000 samples from the mixture
			#xs = []
			#ys = []
			#for i in range(1000):
			#	k      = np.random.randint(config["misc"]["model_samples"])
			#	mean   = predictions[k][11]
			#	cov    = np.array([[sigmas[k][11,0],sigmas[k][#11,2]],[sigmas[k][11,2],sigmas[k][11,1]]])
			#	sample = np.random.multivariate_normal(mean, cov, 1)[0]
			#	xs.append(sample[0])
			#	ys.append(sample[1])

			# Plot the KDEs on two subplots, with level sets			
			#__, axs = plt.subplots(1,2,figsize=(24,12),constrained_layout = True)
			#plot_kde_img(observations_pos[traj_idx],target_pos[traj_idx],xs,ys,test_bckgd,test_homography,axs[0])	
			#plt.show()



if __name__ == "__main__":
	main()
