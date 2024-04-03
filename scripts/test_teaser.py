#!/usr/bin/env python
# coding: utf-8
# Autor: Mario Xavier Canche Uc
# Centro de Investigación en Matemáticas, A.C.
# mario.canche@cimat.mx
from sklearn.isotonic import IsotonicRegression

# Imports
import time
import sys,os,logging, argparse
sys.path.append('bayesian-torch')
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
from torchvision import transforms
import torch.optim as optim
import scipy.stats as st

# Local models
from models.lstm_encdec import lstm_encdec_gaussian
from utils.datasets_utils import setup_loo_experiment,get_testing_batch,get_dataset,traj_dataset,collate_fn_padd
from utils.train_utils import train
from utils.plot_utils import world_to_image_xy
from utils.hdr import get_alpha,get_falpha
from utils.kde import sort_sample,samples_to_alphas,plot_kde_img
from utils.calibration import generate_uncertainty_evaluation_dataset,regression_isotonic_fit,calibrate_and_test
from utils.config import load_config, get_model_name
import torch.optim as optim
# Local constants
from utils.constants import (
	FRAMES_IDS, KEY_IDX, OBS_NEIGHBORS, OBS_TRAJ, OBS_TRAJ_VEL, OBS_TRAJ_ACC, OBS_TRAJ_THETA, PRED_TRAJ, PRED_TRAJ_VEL, PRED_TRAJ_ACC,FRAMES_IDS,
	TRAIN_DATA_STR, TEST_DATA_STR, VAL_DATA_STR, IMAGES_DIR, MUN_POS_CSV, DATASETS_DIR, SUBDATASETS_NAMES, TRAINING_CKPT_DIR
)

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
	seeds = np.random.choice(9999, config["misc"]["model_samples"] , replace=False)
	logging.info("Seeds: {}".format(seeds))

	if config["train"]["no_retrain"]==False:
		# Train model for each seed
		for ind, seed in enumerate(seeds):
			# Seed added
			torch.manual_seed(seed)
			torch.cuda.manual_seed(seed)

			# Instanciate the model
			model = lstm_encdec_gaussian(config["model"])
			model.to(device)

			# Train the model
			logging.info(" Training for seed: {} \t\t {}/{}".format(seed,ind,len(seeds)))
			train(model,device,ind,batched_train_data,batched_val_data,config)
			# Testing: Quantitative
			ade  = 0
			fde  = 0
			total= 0
			model.eval()
			for batch_idx, (observations_vel,__,observations_abs,target_abs,__,__,__) in enumerate(batched_test_data):
				if torch.cuda.is_available():
					observations_vel  = observations_vel.to(device)
				total += len(observations_vel)
				# prediction
				init_pos  = np.expand_dims(observations_abs[:,-1,:],axis=1)
				pred_abs  = model.predict(observations_vel)[0] + init_pos
				ade      += np.average(np.sqrt(np.square(target_abs-pred_abs).sum(2)),axis=1).sum()
				fde      += (np.sqrt(np.square(target_abs[:,-1,:]-pred_abs[:,-1,:]).sum(1))).sum()
			logging.info("Test ade : {:.4f} ".format(ade/total))
			logging.info("Test fde : {:.4f} ".format(fde/total))

	# Instanciate the models
	models= []
	# For each element of the ensemble
	for ind in range(config["misc"]["model_samples"] ):
		model = lstm_encdec_gaussian(config["model"])
		model.to(device)
		# Load the previously trained model
		model_filename = config["train"]["save_dir"]+get_model_name(config,ensemble_id=ind)
		model.load_state_dict(torch.load(model_filename))
		models.append(model)


	#----------------
	__,__,observations_abs_e,target_abs_e,predictions_e,sigmas_e = generate_uncertainty_evaluation_dataset(batched_test_data,model,config,device=device,type="ensemble")

	#---------------------------------------------------------------------------------------------------------------
	# Testing
	cont = 0
	for batch_idx, (observations_vel_c,__,observations_abs_c,target_abs_c,__,__,__) in enumerate(batched_test_data):

		predictions_c = []
		sigmas_c      = []
		# For each model of the ensemble
		for ind in range(config["misc"]["model_samples"]):
			# Load the model
			model_filename = config["train"]["save_dir"]+get_model_name(config,ensemble_id=ind)
			model.load_state_dict(torch.load(model_filename))
			model.eval()
			if torch.cuda.is_available():
				observations_vel_c  = observations_vel_c.to(device)
			predictions, sigmas = model.predict(observations_vel_c)
			# Keep moments
			predictions_c.append(predictions)
			sigmas_c.append(sigmas)
		# Stack the means and covariances
		predictions_c      = np.array(predictions_c)
		sigmas_c           = np.array(sigmas_c)
		observations_abs_c = observations_abs_c.numpy()
		# Calibrate
		# Convert it to absolute (starting from the last observed position)
		predictions_c= predictions_c[:,:,11,:]+observations_abs_c[:,-1,:]
		sigmas_c     = sigmas_c[:,:,11,:]
		target_abs_c = target_abs_c[:,11,:]
		# Uncertainty calibration
		logging.info("Calibration at position: {}".format(11))
		conf_levels,cal_pcts,unc_pcts,__,__= calibrate_and_test(predictions_c,target_abs_c,None,None,2,gaussian=(sigmas_c,None))

		# Isotonic regression: Gives a mapping from predicted alpha to corrected alpha
		iso_reg, iso_inv = regression_isotonic_fit(predictions_c,target_abs_c,kde_size=1000,resample_size=100,sigmas_prediction=sigmas_c)

		# Plot calibration curves (before/after calibration)
		plt.gca().set_aspect('equal')
		plt.plot(conf_levels,unc_pcts,'purple',label=r'$\hat{P}_{\alpha}$ (uncalibrated)')
		plt.plot(conf_levels,cal_pcts,'red',label=r'$\hat{P}_{\alpha}$ (calibrated)')
		plt.xlabel(r'$\alpha$', fontsize=10)
		plt.legend(fontsize=10)
		plt.show()
		#plt.plot(conf_levels,iso_reg.transform(conf_levels),'green',label=r'$a_\alpha$')
		break

	training_data, validation_data, testing_data, test_homography = setup_loo_experiment(config["dataset"])
	test_data = traj_dataset(testing_data[OBS_TRAJ_VEL ], testing_data[PRED_TRAJ_VEL], testing_data[OBS_TRAJ], testing_data[PRED_TRAJ], Frame_Ids=testing_data[FRAMES_IDS])
	frame_id, batch, test_bckgd = get_testing_batch(test_data,config["dataset"])
	batched_test_data  = torch.utils.data.DataLoader(batch,batch_size=len(batch),collate_fn=collate_fn_padd)

	# Get the homography
	homography_to_img = np.linalg.inv(test_homography)

	for (observations_vel,__,observations_abs,target_abs,__,__,__) in batched_test_data:
		logging.info("Trajectories {}".format(len(observations_vel)))
		# Cycle over the trajectories of this batch
		for traj_idx in range(len(observations_vel)):
			# Output for each element of the ensemble
			predictions = []
			sigmas      = []
			for idx in range(config["misc"]["model_samples"]):
				if torch.cuda.is_available():
					observations_vel = observations_vel.to(device)
				prediction, sigma = models[idx].predict(observations_vel)
				predictions.append(prediction[traj_idx]),sigmas.append(sigma[traj_idx])
			# Sampling 1000 samples from the mixture
			xs = []
			ys = []
			for i in range(1000):
				k      = np.random.randint(config["misc"]["model_samples"])
				mean   = predictions[k][11]
				cov    = np.array([[sigmas[k][11,0],sigmas[k][11,2]],[sigmas[k][11,2],sigmas[k][11,1]]])
				sample = np.random.multivariate_normal(mean, cov, 1)[0]+ np.array([observations_abs[traj_idx,-1].numpy()])
				xs.append(sample[0,0])
				ys.append(sample[0,1])

			# Plot the KDEs on two subplots, with level sets			
			__, axs = plt.subplots(1,2,figsize=(24,12),constrained_layout = True)
			plot_kde_img(observations_abs[traj_idx],target_abs[traj_idx],xs,ys,test_bckgd,test_homography,axs[0])	
			plot_kde_img(observations_abs[traj_idx],target_abs[traj_idx],xs,ys,test_bckgd,test_homography,axs[1],iso_inv)	
			plt.show()


if __name__ == "__main__":
	main()
