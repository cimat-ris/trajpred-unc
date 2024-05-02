#!/usr/bin/env python
# coding: utf-8
# Autor: Mario Xavier Canche Uc
# Centro de Investigación en Matemáticas, A.C.
# mario.canche@cimat.mx

# Imports
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch

# Local models
from trajpred_unc.models.lstm_encdec import lstm_encdec_gaussian
from trajpred_unc.utils.datasets_utils import setup_loo_experiment,get_testing_batch,get_dataset,traj_dataset,collate_fn_padd
from trajpred_unc.utils.train_utils import train
from trajpred_unc.uncertainties.kde import plot_kde_img
from trajpred_unc.uncertainties.calibration import generate_uncertainty_evaluation_dataset,regression_isotonic_fit,recalibrate_and_test
from trajpred_unc.utils.config import load_config, get_model_filename
# Local constants
from trajpred_unc.utils.constants import (
	FRAMES_IDS, KEY_IDX, OBS_NEIGHBORS, OBS_TRAJ, OBS_TRAJ_VEL, OBS_TRAJ_ACC, OBS_TRAJ_THETA, PRED_TRAJ, PRED_TRAJ_VEL, PRED_TRAJ_ACC,FRAMES_IDS)

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
			for batch_idx, (observations_vel,__,observations_pos,target_pos,__,__,__) in enumerate(batched_test_data):
				if torch.cuda.is_available():
					observations_vel  = observations_vel.to(device)
				total += len(observations_vel)
				# prediction
				predictions  = model.predict(observations_vel,observations_pos)[0]
				ade         += np.average(np.sqrt(np.square(target_pos-predictions).sum(2)),axis=1).sum()
				fde         += (np.sqrt(np.square(target_pos[:,-1,:]-predictions[:,-1,:]).sum(1))).sum()
			logging.info("Test ade : {:.4f} ".format(ade/total))
			logging.info("Test fde : {:.4f} ".format(fde/total))

	# Instanciate the models
	models= []
	# For each element of the ensemble
	for ind in range(config["misc"]["model_samples"] ):
		model = lstm_encdec_gaussian(config["model"])
		model.to(device)
		# Load the previously trained model
		model_filename = config["train"]["save_dir"]+get_model_filename(config,ensemble_id=ind)
		model.load_state_dict(torch.load(model_filename))
		models.append(model)


	#----------------
	__,__,observations_abs_e,target_abs_e,predictions_e,sigmas_e = generate_uncertainty_evaluation_dataset(batched_test_data,model,config,device=device,type="ensemble")

	#---------------------------------------------------------------------------------------------------------------
	# Testing
	cont = 0
	for batch_idx, (observations_vel_c,__,observations_pos_c,target_pos_c,__,__,__) in enumerate(batched_test_data):

		predictions_c = []
		sigmas_c      = []

		# For each model of the ensemble
		for ind in range(config["misc"]["model_samples"]):
			# Load the model
			model_filename = config["train"]["save_dir"]+get_model_filename(config,ensemble_id=ind)
			model.load_state_dict(torch.load(model_filename))
			model.eval()
			if torch.cuda.is_available():
				observations_vel_c  = observations_vel_c.to(device)
			predictions, sigmas = model.predict(observations_vel_c,observations_pos_c)
			# Keep moments
			predictions_c.append(predictions)
			sigmas_c.append(sigmas)
		# Stack the means and covariances
		predictions_c      = np.array(predictions_c)
		predictions_c      = np.swapaxes(predictions_c,0,1)[:,:,11,:]
		sigmas_c           = np.array(sigmas_c)
		sigmas_c           = np.swapaxes(sigmas_c,0,1)[:,:,11,:]
		target_pos_c	   = target_pos_c.numpy()[:,11,:]
		# Uncertainty calibration
		logging.info("Calibration at position: {}".format(11))
		method  = 2
		results = recalibrate_and_test(predictions_c,target_pos_c,None,None,[method],kde_size=150,resample_size=100,gaussian=(sigmas_c,None))		
		# Isotonic regression: Gives a mapping from predicted alpha to corrected alpha
		iso_reg, iso_inv = regression_isotonic_fit(predictions_c,target_pos_c,kde_size=1000,resample_size=100,sigmas_prediction=sigmas_c)
		# Plot calibration curves (before/after calibration)
		plt.gca().set_aspect('equal')
		plt.plot(results[method]["confidence_levels"],results[method]["raw"]["onCalibrationData"],'purple',label=r'$\hat{P}_{\alpha}$ (uncalibrated)')
		plt.plot(results[method]["confidence_levels"],results[method]["recalibrated"]["onCalibrationData"],'red',label=r'$\hat{P}_{\alpha}$ (calibrated)')
		plt.xlabel(r'$\alpha$', fontsize=10)
		plt.legend(fontsize=10)
		plt.show()
		break

	training_data, validation_data, testing_data, test_homography = setup_loo_experiment(config["dataset"])
	test_data = traj_dataset(testing_data[OBS_TRAJ_VEL ], testing_data[PRED_TRAJ_VEL], testing_data[OBS_TRAJ], testing_data[PRED_TRAJ], Frame_Ids=testing_data[FRAMES_IDS])
	frame_id, batch, test_bckgd = get_testing_batch(test_data,config["dataset"])
	batched_test_data  = torch.utils.data.DataLoader(batch,batch_size=len(batch),collate_fn=collate_fn_padd)

	# Get the homography
	homography_to_img = np.linalg.inv(test_homography)

	for (observations_vel,__,observations_pos,target_abs,__,__,__) in batched_test_data:
		logging.info("Trajectories {}".format(len(observations_vel)))
		# Cycle over the trajectories of this batch
		for traj_idx in range(len(observations_vel)):
			# Output for each element of the ensemble
			predictions = []
			sigmas      = []
			for idx in range(config["misc"]["model_samples"]):
				if torch.cuda.is_available():
					observations_vel = observations_vel.to(device)
				prediction, sigma = models[idx].predict(observations_vel,observations_pos)
				predictions.append(prediction[traj_idx]),sigmas.append(sigma[traj_idx])
			# Sampling 1000 samples from the mixture
			xs = []
			ys = []
			for i in range(1000):
				k      = np.random.randint(config["misc"]["model_samples"])
				mean   = predictions[k][11]
				cov    = np.array([[sigmas[k][11,0],sigmas[k][11,2]],[sigmas[k][11,2],sigmas[k][11,1]]])
				sample = np.random.multivariate_normal(mean, cov, 1)[0]
				xs.append(sample[0])
				ys.append(sample[1])

			# Plot the KDEs on two subplots, with level sets			
			__, axs = plt.subplots(1,2,figsize=(24,12),constrained_layout = True)
			plot_kde_img(observations_pos[traj_idx],target_abs[traj_idx],xs,ys,test_bckgd,test_homography,axs[0])	
			plot_kde_img(observations_pos[traj_idx],target_abs[traj_idx],xs,ys,test_bckgd,test_homography,axs[1],iso_inv)	
			plt.show()


if __name__ == "__main__":
	main()
