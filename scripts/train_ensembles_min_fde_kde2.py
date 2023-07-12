#!/usr/bin/env python
# coding: utf-8
# Autor: Mario Xavier Canche Uc
# Centro de Investigación en Matemáticas, A.C.
# mario.canche@cimat.mx

# Imports
import time
import sys,os,logging,argparse,random, tqdm
import pandas as pd

sys.path.append('bayesian-torch')
sys.path.append('.')
from utils.calibration_utils import save_data_for_calibration

import math,numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as st

import torch
from torchvision import transforms

# Local models
from models.lstm_encdec import lstm_encdec_gaussian
from utils.datasets_utils import get_dataset
from utils.train_utils import train
from utils.plot_utils import plot_traj_img,plot_traj_world,plot_cov_world
from utils.calibration import generate_uncertainty_evaluation_dataset, regression_isotonic_fit, calibrate_and_test
from utils.config import get_config
# Local constants
from utils.constants import REFERENCE_IMG, ENSEMBLES, TRAINING_CKPT_DIR, SUBDATASETS_NAMES
from utils.hdr import get_alpha,get_falpha,sort_sample,samples_to_alphas

# Parser arguments
config = get_config(argv=sys.argv[1:],ensemble=True)


def main():
	# Printing parameters
	torch.set_printoptions(precision=2)
	logging.basicConfig(format='%(levelname)s: %(message)s',level=config.log_level)
	# Set the seed
	logging.info("Seed: {}".format(config.seed))
	torch.manual_seed(config.seed)
	np.random.seed(config.seed)
	random.seed(config.seed)
	# Device
	if torch.cuda.is_available():
		logging.info(torch.cuda.get_device_name(torch.cuda.current_device()))
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Get the ETH-UCY data
	batched_train_data,batched_val_data,batched_test_data,homography,reference_image = get_dataset(config)
	model_name    = 'deterministic_gaussian_ensemble'

	# Select random seeds
	seeds = np.random.choice(99999999, config.num_ensembles , replace=False)
	logging.info("Seeds: {}".format(seeds))

	if config.no_retrain==False:
		# Train model for each seed
		for ind, seed in enumerate(seeds):
			# Seed added
			torch.manual_seed(seed)
			np.random.seed(seed)
			random.seed(seed)

			# Instanciate the model
			model = lstm_encdec_gaussian(in_size=2, embedding_dim=128, hidden_dim=256, output_size=2)
			model.to(device)

			# Train the model
			logging.info("Training for seed: {}\t\t {}/{} ".format(seed,ind,len(seeds)))
			train(model,device,ind,batched_train_data,batched_val_data,config,model_name)

	# Instanciate the model
	model = lstm_encdec_gaussian(in_size=2, embedding_dim=128, hidden_dim=256, output_size=2)
	model.to(device)


	ind_sample = np.random.randint(config.batch_size)

	# Testing
	for batch_idx, (datarel_test, targetrel_test, data_test, target_test) in enumerate(batched_test_data):
		fig, ax = plt.subplots(1,1,figsize=(12,12))

		# For each element of the ensemble
		for ind in range(config.num_ensembles):
			# Load the previously trained model
			model_filename = TRAINING_CKPT_DIR+"/"+model_name+"_"+str(SUBDATASETS_NAMES[config.id_dataset][config.id_test])+"_"+str(ind)+".pth"
			logging.info("Loading {}".format(model_filename))
			model.load_state_dict(torch.load(model_filename))
			model.eval()

			if torch.cuda.is_available():
				  datarel_test  = datarel_test.to(device)

			pred, sigmas = model.predict(datarel_test, dim_pred=12)
			# Plotting
			plot_traj_world(pred[ind_sample,:,:],data_test[ind_sample,:,:],target_test[ind_sample,:,:],ax)
			plot_cov_world(pred[ind_sample,:,:],sigmas[ind_sample,:,:],data_test[ind_sample,:,:],ax)
		plt.legend()
		plt.title('Trajectory samples {}'.format(batch_idx))
		if config.show_plot:
			plt.show()
		# We just use the first batch to test
		break

	#------------------ Generates sub-dataset for calibration evaluation ---------------------------
	datarel_test_full, targetrel_test_full, data_test_full, target_test_full, tpred_samples_full, sigmas_samples_full = generate_uncertainty_evaluation_dataset(batched_test_data, model, config.num_ensembles, model_name, config, device=device)
	#---------------------------------------------------------------------------------------------------------------

	# Testing
	cont = 0
	for batch_idx, (datarel_test, targetrel_test, data_test, target_test) in enumerate(batched_test_data):
		print(datarel_test.shape)
		tpred_samples0 = []
		sigmas_samples = []
		# Muestreamos con cada modelo
		for ind in range(config.num_ensembles):

			# Cargamos el Modelo
			model_filename = TRAINING_CKPT_DIR+"/"+model_name+"_"+str(SUBDATASETS_NAMES[config.id_dataset][config.id_test])+"_"+str(ind)+".pth"
			logging.info("Loading {}".format(model_filename))
			model.load_state_dict(torch.load(model_filename))
			model.eval()

			if torch.cuda.is_available():
				  datarel_test  = datarel_test.to(device)

			pred, sigmas = model.predict(datarel_test, dim_pred=12)

			tpred_samples0.append(pred)
			sigmas_samples.append(sigmas)

		tpred_samples0 = np.array(tpred_samples0)
		sigmas_samples = np.array(sigmas_samples)
		
		# Convert it to absolute (starting from the last observed position)
		tpred_samples0[:,:,11,:] = tpred_samples0[:,:,11,:] + data_test[:,-1,:].numpy()
		
		# Save these testing data for uncertainty calibration
		pickle_filename = ENSEMBLES+"_"+str(SUBDATASETS_NAMES[config.id_dataset][config.id_test])
		save_data_for_calibration(pickle_filename, tpred_samples0, tpred_samples_full, data_test, data_test_full, target_test, target_test_full, sigmas_samples, sigmas_samples_full, config.id_test)

		# Only the first batch is used as the calibration dataset
		break
		
	
	tpred_samples_full[:,:,11,:] = tpred_samples_full[:,:,11,:] + data_test_full[:,-1,:].numpy()
	#print(tpred_samples_full)
	#print(target_test_full)
	#aaaa
		
	#---------------------------------------------------------------------------------------------------------------------
	
	""" Select sub-dataset of 256 trajectories for uncertainty calibration """
	#select = np.random.permutation(gt_trajectories.shape[0])
	tpred_samples   = np.concatenate([tpred_samples0, tpred_samples_full], axis=1) #tpred_samples[:,select,:,:]
	gt_trajectories = np.concatenate([target_test, target_test_full], axis=0) #gt_trajectories[select,:,:]
	obs_trajectories= np.concatenate([data_test, data_test_full], axis=0) #obs_trajectories[select,:,:]
	tpred_samples_cal = tpred_samples0 #tpred_samples[:,:256,:,:]
	tpred_samples_test= tpred_samples_full #tpred_samples[:,256:,:,:]

	data_cal          = data_test #obs_trajectories[:256,:,:]
	data_test         = data_test_full #obs_trajectories[256:,:,:]
	target_cal        = target_test #gt_trajectories[:256,:,:]
	target_test       = target_test_full #gt_trajectories[256:,:,:]
	pickle_filename = model_name+"_"+str(SUBDATASETS_NAMES[0][config.id_test])
	#save_data_for_calibration(pickle_filename, tpred_samples_cal, tpred_samples_test, data_cal, data_test, target_cal, target_test, None, None, config.id_test)

	"""  Evaluate minFDE before calibration."""
	minfdes = []
	for k in tqdm.tqdm(range(tpred_samples.shape[1])):
		xs = []
		ys = []
		#tpred = tpred_samples[:,k,11,:].reshape((-1,2,1))
		gt    = gt_trajectories[k,11,:].reshape((2,1))
		# Prediction samples
		world_samples   = tpred_samples[:,k,11,:].reshape((-1,2))
		if world_samples.shape[0]>0:
			# Build a Kernel Density Estimator with these samples
			kde         = st.gaussian_kde(world_samples.T)
			sub_samples = kde.resample(20).T.reshape((-1,2,1))
			fdes        = np.sqrt(np.sum(np.square(sub_samples - gt),axis=1))
			minfdes.append(np.amin(fdes))
	#allmfdes_before.append(np.mean(minfdes))
	minfdes_antes = np.mean(minfdes)
	print("minFDE: {:.4f}".format(minfdes_antes))


	""" Calibration."""
	# Uncertainty calibration
	logging.info("Calibration at position: {}".format(11))
	# Isotonic regression: Gives a mapping from predicted alpha to corrected alpha
	iso_reg, iso_inv = regression_isotonic_fit(tpred_samples_cal[:,:,11,:],target_cal,11,kde_size=1000,resample_size=100,sigmas_prediction=sigmas_samples)

	"""  Evaluate minFDE after calibration."""
	minfdes = []
	for k in tqdm.tqdm(range(tpred_samples.shape[1])):
		xs = []
		ys = []
		#tpred = tpred_samples[:,k,11,:].reshape((-1,2,1))
		gt    = gt_trajectories[k,11,:].reshape((2,1))
		# Prediction samples
		world_samples   = tpred_samples[:,k,11,:].reshape((-1,2))
		if world_samples.shape[0]>0:
			# Build a Kernel Density Estimator with these samples
			kde         = st.gaussian_kde(world_samples.T)
			# Evaluate our samples on it
			alphas_samples, fs_samples, sorted_samples = samples_to_alphas(kde,world_samples.T)
			modified_alphas = iso_inv.transform(alphas_samples)
			fs_samples_new  = []
			for alpha in modified_alphas:
				fs_samples_new.append(get_falpha(sorted_samples,alpha))
			fs_samples_new    = np.array(fs_samples_new)
			importance_weights= fs_samples_new/fs_samples
			kde               = st.gaussian_kde(world_samples.T,weights=importance_weights)
			sub_samples = kde.resample(20).T.reshape((-1,2,1))
			fdes        = np.sqrt(np.sum(np.square(sub_samples - gt),axis=1))
			minfdes.append(np.amin(fdes))
	#allmfdes_after.append(np.mean(minfdes))
	minfdes_despues = np.mean(minfdes)
	print("minFDE: {:.4f}".format(minfdes_despues))
	
	# Guardamos con un data frame
	df = pd.DataFrame([[minfdes_antes, minfdes_despues]], columns=["minfde_antes", "minfde_despues"])
	output_csv_name = "images/calibration/"+pickle_filename + "_" + str(config.num_ensembles) + "_minFDE_kde.csv"
	df.to_csv(output_csv_name, mode='a', header=not os.path.exists(output_csv_name))


if __name__ == "__main__":
	main()
	
