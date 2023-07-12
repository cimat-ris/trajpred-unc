#!/usr/bin/env python
# coding: utf-8
# Autor: Mario Xavier Canche Uc
# Centro de Investigación en Matemáticas, A.C.
# mario.canche@cimat.mx

# Cargamos las librerias
import time
import sys,os,logging

''' TF_CPP_MIN_LOG_LEVEL
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printeds
3 = INFO, WARNING, and ERROR messages are not printed
'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append('bayesian-torch')
sys.path.append('.')

import math,random,numpy as np
import matplotlib as mpl
#mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
import pandas as pd

import torch
from torchvision import transforms

import math,numpy as np
import matplotlib as mpl
import matplotlib.patches as patches
#mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import cv2
import torch
from torchvision import transforms
import torch.optim as optim
import scipy.stats as st

# Local models
from models.bayesian_models_gaussian_loss import lstm_encdec_MCDropout
from utils.datasets_utils import get_dataset, setup_loo_experiment,Experiment_Parameters,traj_dataset
from utils.plot_utils import plot_traj_img,plot_traj_world,plot_cov_world,world_to_image_xy
from utils.calibration import generate_uncertainty_evaluation_dataset,regression_isotonic_fit,calibrate_and_test
from utils.calibration_utils import save_data_for_calibration
from utils.train_utils import train, evaluation_minadefde
from utils.config import get_config
from utils.hdr import get_alpha,get_falpha,sort_sample,samples_to_alphas

# Local constants
from utils.constants import IMAGES_DIR, DROPOUT, TRAINING_CKPT_DIR, SUBDATASETS_NAMES
from utils.constants import (
	FRAMES_IDS, KEY_IDX, OBS_NEIGHBORS, OBS_TRAJ, OBS_TRAJ_VEL, OBS_TRAJ_ACC, OBS_TRAJ_THETA, PRED_TRAJ, PRED_TRAJ_VEL, PRED_TRAJ_ACC,FRAMES_IDS,
	TRAIN_DATA_STR, TEST_DATA_STR, VAL_DATA_STR, IMAGES_DIR, MUN_POS_CSV, DATASETS_DIR, SUBDATASETS_NAMES, TRAINING_CKPT_DIR
)

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
	
# Parser arguments
config = get_config(argv=sys.argv[1:],dropout=True)
model_name = 'deterministic_dropout'

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
	#batched_train_data,batched_val_data,batched_test_data,homography,reference_image = get_dataset(config)
	
	# Load the dataset and perform the split
	experiment_parameters = Experiment_Parameters()
	training_data, validation_data, test_data, test_homography = setup_loo_experiment(DATASETS_DIR[0],SUBDATASETS_NAMES[0],config.id_test,experiment_parameters,pickle_dir='pickle',use_pickled_data=config.pickle)

	# Torch dataset
	train_data  = traj_dataset(training_data[OBS_TRAJ_VEL ], training_data[PRED_TRAJ_VEL],training_data[OBS_TRAJ], training_data[PRED_TRAJ])
	val_data    = traj_dataset(validation_data[OBS_TRAJ_VEL ], validation_data[PRED_TRAJ_VEL],validation_data[OBS_TRAJ], validation_data[PRED_TRAJ])
	testing_data= traj_dataset(test_data[OBS_TRAJ_VEL ], test_data[PRED_TRAJ_VEL], test_data[OBS_TRAJ], test_data[PRED_TRAJ],test_data[FRAMES_IDS])

	# Form batches
	batched_train_data = torch.utils.data.DataLoader(train_data,batch_size=config.batch_size,shuffle=False)
	batched_val_data   = torch.utils.data.DataLoader(val_data,batch_size=config.batch_size,shuffle=False)
	batched_test_data  = torch.utils.data.DataLoader(testing_data,batch_size=config.batch_size,shuffle=True)

	# Instantiate the model
	model = lstm_encdec_MCDropout(2,128,256,2,dropout_rate=config.dropout_rate)
	model.to(device)

	if config.no_retrain==False:
		# Train the model
		train(model,device,0,batched_train_data,batched_val_data,config,model_name)

		if config.plot_losses:
			plt.savefig(IMAGES_DIR+"/loss_"+str(config.id_test)+".pdf")
			plt.show()

	# Load the previously trained model
	file_name = TRAINING_CKPT_DIR+"/"+model_name+"_"+str(SUBDATASETS_NAMES[config.id_dataset][config.id_test])+"_0.pth"
	model.load_state_dict(torch.load(file_name))
	model.eval()

	# Testing
	ind_sample = np.random.randint(config.batch_size)
	for batch_idx, (datarel_test, targetrel_test, data_test, target_test) in enumerate(batched_test_data):
		fig, ax = plt.subplots(1,1,figsize=(12,12))
		if ind_sample>data_test.shape[0]:
			continue
		# Generate samples from the model
		for ind in range(config.dropout_samples):
			if torch.cuda.is_available():
				datarel_test  = datarel_test.to(device)
			pred, sigmas = model.predict(datarel_test, dim_pred=12)
			# Plotting
			plot_traj_world(pred[ind_sample,:,:],data_test[ind_sample,:,:],target_test[ind_sample,:,:],ax,nolabel=False if ind==config.dropout_samples-1 else True)
		plt.legend()
		plt.title('Trajectory samples')
		if config.show_plot:
			plt.show()

	# ## Calibramos la incertidumbre
	draw_ellipse = True

	#------------------ Obtenemos el batch unico de test para las curvas de calibracion ---------------------------
	datarel_test_full, targetrel_test_full, data_test_full, target_test_full, tpred_samples_full, sigmas_samples_full = generate_uncertainty_evaluation_dataset(batched_test_data, model, config.dropout_samples, model_name, config, device=device, type="dropout")
	
	print("-->>", datarel_test_full.shape, targetrel_test_full.shape, data_test_full.shape, target_test_full.shape, tpred_samples_full.shape, sigmas_samples_full.shape)
	evaluation_minadefde( model, tpred_samples_full, data_test_full, target_test_full, model_name+"_min_adefde_kde_")

	#---------------------------------------------------------------------------------------------------------------

	# Testing
	cont = 0
	for batch_idx, (datarel_test, targetrel_test, data_test, gt_calibration) in enumerate(batched_test_data):

		predictions_calibration = []
		sigmas_calibration      = []
		# Sampling from inference dropout
		for ind in range(config.dropout_samples):

			if torch.cuda.is_available():
				datarel_test  = datarel_test.to(device)

			pred, sigmas = model.predict(datarel_test, dim_pred=12)

			predictions_calibration.append(pred)
			sigmas_calibration.append(sigmas)

		# Stack the means and covariances
		predictions_calibration   = np.array(predictions_calibration)
		sigmas_calibration        = np.array(sigmas_calibration)
		observations_calibration  = data_test.numpy()
		# Calibrate
		# Convert it to absolute (starting from the last observed position)
		predictions_calibration= predictions_calibration[:,:,11,:]+observations_calibration[:,-1,:]

		# Uncertainty calibration
		logging.info("Calibration at position: {}".format(11))
		conf_levels,cal_pcts,unc_pcts,__,__= calibrate_and_test(predictions_calibration,gt_calibration,None,None,11,2,gaussian=(sigmas_calibration,None))

		# Isotonic regression: Gives a mapping from predicted alpha to corrected alpha
		iso_reg, iso_inv = regression_isotonic_fit(predictions_calibration,gt_calibration,11,kde_size=1000,resample_size=100,sigmas_prediction=sigmas_calibration)

		# Plot calibration curves (before/after calibration)
		#plt.gca().set_aspect('equal')
		#plt.plot(conf_levels,unc_pcts,'purple',label=r'$\hat{P}_{\alpha}$ (uncalibrated)')
		#plt.plot(conf_levels,cal_pcts,'red',label=r'$\hat{P}_{\alpha}$ (calibrated)')
		#plt.plot(conf_levels,iso_reg.transform(conf_levels),'green',label=r'$a_\alpha$')
		#plt.xlabel(r'$\alpha$', fontsize=10)
		#plt.legend(fontsize=10)
		#plt.show()
		break



	frame_id, batch, test_bckgd = get_testing_batch(testing_data,DATASETS_DIR[0]+SUBDATASETS_NAMES[0][config.id_test])
	# Form batches
	batched_test_data  = torch.utils.data.DataLoader(batch,batch_size=len(batch))
	# Get the homography
	homography_to_img = np.linalg.inv(test_homography)

	print("-->>", datarel_test_full.shape, targetrel_test_full.shape, data_test_full.shape, target_test_full.shape, tpred_samples_full.shape, sigmas_samples_full.shape)
	
	l2dis = []
	print(len(batched_test_data))
	#for batch_idx, (observations_rel_test, targetrel_test,observations_test, target_test) in enumerate(batched_test_data):
	for batch_idx, (observations_rel_test0, targetrel_test0, observations_test0, target_test0) in enumerate(batched_test_data):
		observations_rel_test = datarel_test_full
		targetrel_test = targetrel_test_full
		observations_test = data_test_full
		target_test = target_test_full

		#print(observations_rel_test.shape)
		
		logging.info("Trajectories {}".format(len(observations_rel_test)))
		logging.info("Trajectories 2. {}".format(len(observations_rel_test0)))
		# Cycle over the trajectories of this batch
		for traj_idx in range(len(observations_rel_test)):
			##print("---- traj_idx: ", traj_idx)
			# Output for each element of the ensemble
			predictions = []
			sigmas      = []
			#print("observations_test: ")
			#print(observations_test.shape)
			#print(observations_rel_test.shape)
			#print(observations_rel_test0.shape)
			#print(observations_rel_test[0])
			#print(observations_test[0])
			for idx in range(config.dropout_samples):
				if torch.cuda.is_available():
					observations_rel_test  = observations_rel_test.to(device)
				prediction, sigma = model.predict(observations_rel_test, dim_pred=12)
				#print("preds --> ", prediction.shape)
				predictions.append(prediction[traj_idx]),sigmas.append(sigma[traj_idx])
			# Sampling 1000 samples from the mixture
			xs = []
			ys = []
			for i in range(1000):
				k      = np.random.randint(config.num_ensembles)
				mean   = predictions[k][11]

				cov    = np.array([[sigmas[k][11,0],sigmas[k][11,2]],[sigmas[k][11,2],sigmas[k][11,1]]])
				sample = np.random.multivariate_normal(mean, cov, 1)[0]+ np.array([observations_test[traj_idx,-1].numpy()])
				xs.append(sample[0,0])
				ys.append(sample[0,1])

			xmin = 0
			xmax = test_bckgd.shape[1]
			ymin = 0
			ymax = test_bckgd.shape[0]
			xx, yy = np.mgrid[xmin:xmax:100j,ymin:ymax:100j]

			# Testing/visualization uncalibrated KDE
			image_grid      = np.vstack([xx.ravel(), yy.ravel()])
			world_grid      = world_to_image_xy(np.transpose(image_grid),test_homography,flip=False)
			# Prediction samples
			world_samples   = np.vstack([xs, ys])
			image_samples   = world_to_image_xy(np.transpose(world_samples),homography_to_img,flip=False)
			# Build a Kernel Density Estimator with these samples
			kde             = st.gaussian_kde(world_samples)
			# Evaluate our samples on it
			alphas_samples, fs_samples, sorted_samples = samples_to_alphas(kde,world_samples)

			# Visualization of the uncalibrated KDE with its level curves
			alphas = np.linspace(1.0,0.0,num=5,endpoint=False)
			levels = []
			for alpha in alphas:
				level = get_falpha(sorted_samples,alpha)
				levels.append(level)
			# Apply the KDE on the points of the world grid
			f_unc        = np.reshape(kde(np.transpose(world_grid)).T, xx.shape)
			transparency = np.rot90(f_unc)/np.max(f_unc)

			## Or kernel density estimate plot instead of the contourf plot
			#figs, axs = plt.subplots(1,2,figsize=(24,12),constrained_layout = True)
			#axs[0].legend_ = None
			#axs[0].imshow(test_bckgd)
			observations = world_to_image_xy(observations_test[traj_idx,:,:], homography_to_img, flip=False)
			groundtruth  = world_to_image_xy(target_test[traj_idx,:,:], homography_to_img, flip=False)
			# Contour plot
			#cset = axs[0].contour(xx, yy, f_unc, colors='darkgreen',levels=levels[1:],linewidths=0.75)
			#cset.levels = np.array(alphas[1:])
			#axs[0].clabel(cset, cset.levels,fontsize=8)
			#axs[0].plot(observations[:,0],observations[:,1],color='blue')
			#axs[0].plot([observations[-1,0],groundtruth[0,0]],[observations[-1,1],groundtruth[0,1]],color='red')
			#axs[0].plot(groundtruth[:,0],groundtruth[:,1],color='red')
			#axs[0].set_xlim(xmin,xmax)
			#axs[0].set_ylim(ymax,ymin)
			#axs[0].axes.xaxis.set_visible(False)
			#axs[0].axes.yaxis.set_visible(False)
			# Plot the pdf
			#axs[0].imshow(transparency,alpha=np.sqrt(transparency),cmap=plt.cm.Greens_r,extent=[xmin, xmax, ymin, ymax])

			# Testing/visualization **calibrated** KDE
			modified_alphas = iso_inv.transform(alphas_samples)

			# New values for f
			fs_samples_new  = []
			for alpha in modified_alphas:
				fs_samples_new.append(get_falpha(sorted_samples,alpha))
			fs_samples_new    = np.array(fs_samples_new)
			sorted_samples_new= sort_sample(fs_samples_new)
			importance_weights= fs_samples_new/fs_samples



			#-------------------------------------------------------------------------------------
			#print("usado para la kde:")
			#print(world_samples)
			kde               = st.gaussian_kde(world_samples,weights=importance_weights) # 10 veces
			
			# Calculamos el minFDE
			sample_pos12 = kde.resample(config.dropout_samples)
			#print("muestreado:")
			#print(sample_pos12.reshape(-1,2).shape)
			#print("GT:")
			#print(traj_idx)
			#print(target_test.shape)
			#print(target_test[traj_idx].numpy())
			#print(target_test[traj_idx][-1,:].numpy().reshape(1,2))
			
			diff = sample_pos12 - target_test[traj_idx][-1,:].numpy().reshape(2,1)
			#print(diff.shape)
			#print(diff)
			diff = diff**2
			diff = np.sqrt(np.sum(diff, axis=0))
			#print("diferencias:")
			#print(diff)
			


			normin = 999999999999.0
			diffmin= None
			for diff_indx in range(diff.shape[0]):
				# To keep the min
				if np.linalg.norm(diff[diff_indx])<normin:
					normin  = np.linalg.norm(diff[diff_indx])
					diffmin = diff[diff_indx]
			
			#print(diffmin)
			
			l2dis.append(diffmin)

			##print("---- l2dis: ", diffmin)
			
			#print(kde.shape)
			#-------------------------------------------------------------------------------------




			alphas_samples, fs_samples, sorted_samples = samples_to_alphas(kde,world_samples)
			f_cal             = np.reshape(kde(np.transpose(world_grid)).T, xx.shape)
			norm_f_cal        = np.rot90(f_cal)/np.max(f_unc)
			transparency      = np.minimum(norm_f_cal,1.0)
			# Visualization of the calibrated KDE
			alphas = np.linspace(1.0,0.0,num=5,endpoint=False)
			levels = []
			for alpha in alphas:
				level = get_falpha(sorted_samples,alpha)
				levels.append(level)
			#cset = axs[1].contour(xx, yy, f_cal, colors='darkgreen',levels=levels[1:],linewidths=0.75)
			#cset.levels = np.array(alphas[1:])
			#axs[1].clabel(cset, cset.levels,fontsize=8)
			#axs[1].plot(observations[:,0],observations[:,1],color='blue')
			#axs[1].plot([observations[-1,0],groundtruth[0,0]],[observations[-1,1],groundtruth[0,1]],color='red')
			#axs[1].plot(groundtruth[:,0],groundtruth[:,1],color='red')
			#axs[1].imshow(test_bckgd)
			#axs[1].set_xlim(xmin,xmax)
			#axs[1].set_ylim(ymax,ymin)
			#axs[1].axes.xaxis.set_visible(False)
			#axs[1].axes.yaxis.set_visible(False)
			#axs[1].imshow(norm_f_cal,alpha=np.sqrt(transparency),cmap=plt.cm.Greens_r, extent=[xmin, xmax, ymin, ymax])
			#plt.show()

		break


	results = [["mADE", "mFDE"], [0, np.mean(l2dis)]]
    
	output_csv_name = "images/calibration/" + model_name +"_min_fde_kde.csv"
	df = pd.DataFrame(results)
	df.to_csv(output_csv_name, mode='a', header=not os.path.exists(output_csv_name))
	print(df)

		
		

if __name__ == "__main__":
	main()
