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
from trajpred_unc.uncertainties.calibration_utils import save_data_for_uncertainty_calibration,get_data_for_calibration
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


	models= []
	# Cycle over the models of the ensemble
	for ind, seed in enumerate(seeds):
		# Seed added
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)

		# Instanciate the model
		model = lstm_encdec_gaussian(config["model"])
		model.to(device)

		# Train the model if necessary
		if config["train"]["no_retrain"]==False:
			logging.info(" Training for seed: {} \t\t {}/{}".format(seed,ind,len(seeds)))
			train(model,device,ind,batched_train_data,batched_val_data,config)

		# Load the previously trained model
		model_filename = config["train"]["save_dir"]+get_model_filename(config,ensemble_id=ind)
		model.load_state_dict(torch.load(model_filename))
		model.to(device)
		models.append(model)

		# Testing: Quantitative
		ade  = 0
		fde  = 0
		total= 0
		model.eval()
		for (observations,targets,__,__,__) in batched_test_data:
			if torch.cuda.is_available():
				observations  = observations.to(device)
			total += len(observations)
			# Prediction
			targets      = targets[:,:,:2]
			predictions  = model.predict(observations[:,:,2:4],observations[:,:,:2])[0]
			ade         += np.average(np.sqrt(np.square(targets-predictions).sum(2)),axis=1).sum()
			fde         += (np.sqrt(np.square(targets[:,-1,:]-predictions[:,-1,:]).sum(1))).sum()
		logging.info("Model {} Test ade : {:.4f} ".format(ind,ade/total))
		logging.info("Model {} Test fde : {:.4f} ".format(ind,fde/total))


	#----------------
	# Use the test data to generate the calibration data
	observations,targets,predictions,sigmas = generate_uncertainty_evaluation_dataset(batched_test_data,model,config,device=device,type="ensemble")
	save_data_for_uncertainty_calibration("tmp_calibration",predictions,observations,targets,sigmas,config["dataset"]["id_test"])
	predictions_calibration,__,__,__,groundtruth_calibration,__,sigmas_calibration,__,__ = get_data_for_calibration("tmp_calibration")

	#---------------------------------------------------------------------------------------------------------------
	# Uncertainty calibration at a given position
	position= 11
	logging.info("Calibration at position: {}".format(position))
	method  = 1
	results = recalibrate_and_test(predictions_calibration[:,:,position,:2],groundtruth_calibration[:,position,:2],None,None,[method],kde_size=150,resample_size=100,gaussian=(sigmas_calibration[:,:,position,:],None))		
	# Isotonic regression: Gives a mapping from predicted alpha to corrected alpha
	__, iso_inv = regression_isotonic_fit(predictions_calibration[:,:,position,:2],groundtruth_calibration[:,position,:2],kde_size=1000,resample_size=100,sigmas_prediction=sigmas_calibration[:,:,position,:])

	# Plot calibration curves (before/after calibration)
	plt.gca().set_aspect('equal')
	plt.plot(results[method]["confidence_levels"],results[method]["raw"]["onCalibrationData"],'purple',label=r'$\hat{P}_{\alpha}$ (uncalibrated)')
	plt.plot(results[method]["confidence_levels"],results[method]["recalibrated"]["onCalibrationData"],'red',label=r'$\hat{P}_{\alpha}$ (calibrated)')
	plt.xlabel(r'$\alpha$', fontsize=10)
	plt.legend(fontsize=10)
	plt.show()
	

	#---------------------------------------------------------------------------------------------------------------
	__, __, testing_data, test_homography = setup_loo_experiment(config["dataset"])
	test_data = traj_dataset(testing_data[OBS_TRAJ],testing_data[PRED_TRAJ],Frame_Ids=testing_data[FRAMES_IDS])
	frame_id, batch, test_bckgd = get_testing_batch(test_data,config["dataset"])
	batched_test_data  = torch.utils.data.DataLoader(batch,batch_size=len(batch),collate_fn=collate_fn_padd)

	# Get the homography
	homography_to_img = np.linalg.inv(test_homography)

	for (observations,targets,__,__,__) in batched_test_data:
		logging.info("Trajectories {}".format(len(observations)))
		# Cycle over the trajectories of this batch
		for traj_idx in range(len(observations)):
			print(traj_idx)
			# Output for each element of the ensemble
			predictions = []
			sigmas      = []
			for idx in range(config["misc"]["model_samples"]):
				if torch.cuda.is_available():
					observations = observations.to(device)
				prediction, sigma = models[idx].predict(observations[:,:,2:4],observations[:,:,:2])
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
			observations_ = observations[:,:,:2].cpu()
			targets_      = targets[:,:,:2].cpu()
			# Plot the KDEs on two subplots, with level sets			
			__, axs = plt.subplots(1,2,figsize=(24,12),constrained_layout = True)
			plot_kde_img(observations_[traj_idx],targets_[traj_idx],xs,ys,test_bckgd,test_homography,axs[0])	
			plot_kde_img(observations_[traj_idx],targets_[traj_idx],xs,ys,test_bckgd,test_homography,axs[1],iso_inv)	
			plt.show()


if __name__ == "__main__":
	main()
