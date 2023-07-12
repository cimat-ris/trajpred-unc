import pdb
import os
import sys

sys.path.append(os.path.realpath('.'))
sys.path.append('../bidireaction-trajectory-prediction/')
sys.path.append('../bidireaction-trajectory-prediction/datasets')
import torch
from torch import nn, optim
from torch.nn import functional as F
from utils.calibration_utils import save_data_for_calibration

import pickle as pkl
from datasets import make_dataloader
#import make_dataloader
from bitrap.modeling import make_model
from bitrap.utils.dataset_utils import restore
from bitrap.engine.utils import print_info, post_process

from utils.datasets_utils import setup_loo_experiment,Experiment_Parameters,traj_dataset,get_testing_batch,get_testing_batch_bitrap,traj_dataset_bitrap
import logging
# Local constants
from utils.constants import DATASETS_DIR,SUBDATASETS_NAMES,FRAMES_IDS,OBS_TRAJ,PRED_TRAJ_VEL,OBS_TRAJ_VEL,OBS_TRAJ_ACC,OBS_NEIGHBORS,PRED_TRAJ,BITRAP
from utils.plot_utils import plot_traj_img,plot_traj_world,plot_cov_world,world_to_image_xy
from utils.hdr import get_alpha,get_falpha,sort_sample,samples_to_alphas
from utils.calibration import generate_uncertainty_evaluation_dataset,regression_isotonic_fit,calibrate_and_test

import argparse
from configs import cfg
from termcolor import colored
import numpy as np
import tqdm
import pdb
import matplotlib.pyplot as plt
import scipy.stats as st
import random

config_files = ["cfg/bitrap_np_hotel.yml","cfg/bitrap_np_eth.yml","cfg/bitrap_np_zara1.yml","cfg/bitrap_np_zara2.yml","cfg/bitrap_np_univ.yml"]

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
	parser.add_argument('--gpu', default='0', type=str)
	parser.add_argument('--seed', default=2, type=int)
	parser.add_argument('--id-test',
						type=int, default=4, metavar='N',
						help='id of the dataset to use as test in LOO (default: 2)')
	parser.add_argument('--batch-size', '--b',
						type=int, default=256, metavar='N',
						help='input batch size for training (default: 256)')
	parser.add_argument('--pickle',
						action='store_true',
						help='use previously made pickle files')
	parser.add_argument(
		"opts",
		help="Modify config options using the command-line",
		default=None,
		nargs=argparse.REMAINDER,
	)
	parser.add_argument('--log-level',type=int, default=20,help='Log level (default: 20)')
	parser.add_argument('--log-file',default='',help='Log file (default: standard output)')
	args = parser.parse_args()
	# Loggin format
	logging.basicConfig(format='%(levelname)s: %(message)s',level=args.log_level)

	# Load the default parameters
	experiment_parameters = Experiment_Parameters()

	#### Data: our way
	model_name = BITRAP
	flipImage  = False
	#### Data: BitTrap way
	cfg.merge_from_file(config_files[args.id_test])
	cfg.merge_from_list(args.opts)
	cfg.BATCH_SIZE = 1
	cfg.TEST.BATCH_SIZE = 1
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	# Get dataloader
	test_dataloader = make_dataloader(cfg, 'test')
	env             = test_dataloader.dataset.dataset.env

	# Load a model
	cfg.CKPT_DIR = 'training_checkpoints/bitrap/bitrap-{}-{:02d}.pth'.format(cfg.DATASET.NAME,args.seed)
	# Build model, optimizer and scheduler
	model = make_model(cfg)
	model = model.to(cfg.DEVICE)
	if os.path.isfile(cfg.CKPT_DIR):
	   model.load_state_dict(torch.load(cfg.CKPT_DIR))
	   logging.info(colored('Loaded checkpoint:{}'.format(cfg.CKPT_DIR), 'blue', 'on_green'))
	else:
	   logging.info(colored('The cfg.CKPT_DIR id not a file: {}'.format(cfg.CKPT_DIR), 'green', 'on_red'))
	model.K = 100


	##################################################################
	# With our data (to show how to use BitTrap normalization)
	# Load the dataset and perform the split
	__, __, test_data, test_homography = setup_loo_experiment(DATASETS_DIR[0],SUBDATASETS_NAMES[0],args.id_test,experiment_parameters,pickle_dir='pickle',use_pickled_data=args.pickle, compute_neighbors=True)
	testing_data= traj_dataset(test_data[OBS_TRAJ_VEL ], test_data[PRED_TRAJ_VEL], test_data[OBS_TRAJ], test_data[PRED_TRAJ],test_data[FRAMES_IDS])

	# Calibration
	X_test            = np.concatenate([test_data[OBS_TRAJ],test_data[OBS_TRAJ_VEL],test_data[OBS_TRAJ_ACC]],axis=2)
	test_data_bitrap  = traj_dataset_bitrap(X_test,test_data[OBS_NEIGHBORS],test_data[PRED_TRAJ],Frame_Ids=test_data[FRAMES_IDS])
	# Form batches
	batched_test_data  = torch.utils.data.DataLoader(test_data_bitrap,batch_size=1,shuffle=False)
	pred_traj  = np.zeros((model.K, len(batched_test_data), 12, 2))
	obs_traj   = np.zeros((len(batched_test_data), 8,2))
	gt_traj    = np.zeros((len(batched_test_data),12,2))
	gt_traj_rel= np.zeros((len(batched_test_data),12,2))

	for batch_idx, (observations_test, neighbors_test, target_test) in enumerate(batched_test_data):
		# Cycle over the trajectories of this batch
		for traj_idx in range(len(observations_test)):
			# Input: batch_sizex8x6 (un-normalized)
			X_global     = torch.Tensor(observations_test[traj_idx:traj_idx+1]).to(cfg.DEVICE)
			# This is the GT: batch_sizex12x2 (un-normalized)
			y_global     = torch.Tensor(target_test[traj_idx:traj_idx+1])
			# Input: batch_sizex8x6 (**normalized**)
			node_type = env.NodeType[0]
			state     = {'position':['x','y'], 'velocity':['x','y'], 'acceleration':['x','y']}
			_, std    = env.get_standardize_params(state, node_type)
			std[0:2]  = env.attention_radius[(node_type, node_type)]
			# Reference point: the last observed position, batch_size x 6
			observations_reference       = np.zeros_like(observations_test[traj_idx:traj_idx+1,0])
			observations_reference[:,0:2]= np.array(observations_test)[traj_idx:traj_idx+1,-1, 0:2]
			observations_reference       = np.expand_dims(observations_reference,axis=1)
			# Normalize the inputs
			input_x                      = env.standardize(observations_test[traj_idx:traj_idx+1],state,node_type,mean=observations_reference,std=std)
			input_x                      = torch.tensor(input_x,dtype=torch.float).to(cfg.DEVICE)
			# Neighbors
			input_neighbors = {}
			input_neighbors[(node_type,node_type)] = [[]]
			n_neighbors = len(neighbors_test)
			for neighbor in neighbors_test:
				neighbor_normalized = env.standardize(neighbor, state, node_type, mean=observations_reference, std=std)
				input_neighbors[(node_type,node_type)][0].append(neighbor_normalized)
			# Should be a list of x tensors 8x6
			# Dictionary with keys pairs node_type x node_type
			input_adjacency = {}
			input_adjacency[(node_type,node_type)] = [torch.tensor(np.ones(n_neighbors))]
			# Sample a trajectory
			pred_goal_, pred_traj_, _, dist_goal_, dist_traj_ = model(input_x,neighbors_st=input_neighbors,adjacency=input_adjacency,z_mode=False,cur_pos=X_global[:,-1,:2],first_history_indices=torch.tensor([0],dtype=int))
			# Transfer back to global coordinates
			ret = post_process(cfg, X_global, y_global, pred_traj_, pred_goal=pred_goal_, dist_traj=dist_traj_, dist_goal=dist_goal_)
			X_global_, y_global_, pred_goal_, pred_traj_, dist_traj_, dist_goal_ = ret
			pred_traj[:,batch_idx,:,:] = np.swapaxes(pred_traj_[0,:,:,:], 0, 1)
			obs_traj[batch_idx,:,:]    = observations_test[0,:,:2].numpy()
			gt_traj[batch_idx,:,:]     = target_test[0,:,:].numpy()
			gt_traj_rel[batch_idx,:,:] = target_test[0,:,:].numpy() - observations_test[0,-1,:2].numpy()


	# Uncertainty calibration
	logging.info("Calibration at position: {}".format(11))
	pred_traj= pred_traj[:,:,11,:]
	conf_levels,cal_pcts,unc_pcts,__,__= calibrate_and_test(pred_traj,gt_traj,None,None,11,2,gaussian=(None,None))

	# Isotonic regression: Gives a mapping from predicted alpha to corrected alpha
	iso_reg, iso_inv = regression_isotonic_fit(pred_traj,gt_traj,11,kde_size=1000,resample_size=100,sigmas_prediction=None)


	# Testing
	frame_id, batch, test_bckgd = get_testing_batch_bitrap(test_data_bitrap,DATASETS_DIR[0]+SUBDATASETS_NAMES[0][args.id_test])
	# Form batches
	batched_test_data  = torch.utils.data.DataLoader(batch,batch_size=len(batch))
	# Get the homography
	homography_to_img = np.linalg.inv(test_homography)

	for batch_idx, (observations_test, neighbors_test, target_test) in enumerate(batched_test_data):
		logging.info("Trajectories {}".format(len(observations_test)))
		# Cycle over the trajectories of this batch
		for traj_idx in range(len(observations_test)):

			# Input: batch_sizex8x6 (un-normalized)
			X_global     = torch.Tensor(observations_test[traj_idx:traj_idx+1]).to(cfg.DEVICE)
			# This is the GT: batch_sizex12x2 (un-normalized)
			y_global     = torch.Tensor(target_test[traj_idx:traj_idx+1])
			# Input: batch_sizex8x6 (**normalized**)
			node_type = env.NodeType[0]
			state     = {'position':['x','y'], 'velocity':['x','y'], 'acceleration':['x','y']}
			_, std    = env.get_standardize_params(state, node_type)
			std[0:2]  = env.attention_radius[(node_type, node_type)]
			# Reference point: the last observed position, batch_size x 6
			observations_reference       = np.zeros_like(observations_test[traj_idx:traj_idx+1,0])
			observations_reference[:,0:2]= np.array(observations_test)[traj_idx:traj_idx+1,-1, 0:2]
			observations_reference       = np.expand_dims(observations_reference,axis=1)
			# Normalize the inputs
			input_x                      = env.standardize(observations_test[traj_idx:traj_idx+1],state,node_type,mean=observations_reference,std=std)
			input_x                      = torch.tensor(input_x,dtype=torch.float).to(cfg.DEVICE)
			# Neighbors
			input_neighbors = {}
			input_neighbors[(node_type,node_type)] = [[]]
			n_neighbors = len(neighbors_test)
			for neighbor in neighbors_test:
				neighbor_normalized = env.standardize(neighbor, state, node_type, mean=observations_reference, std=std)
				input_neighbors[(node_type,node_type)][0].append(neighbor_normalized)
			# Should be a list of x tensors 8x6
			# Dictionary with keys pairs node_type x node_type
			input_adjacency = {}
			input_adjacency[(node_type,node_type)] = [torch.tensor(np.ones(n_neighbors))]
			# Sample a trajectory
			pred_goal_, pred_traj_, _, dist_goal_, dist_traj_ = model(input_x,neighbors_st=input_neighbors,adjacency=input_adjacency,z_mode=False,cur_pos=X_global[:,-1,:2],first_history_indices=torch.tensor([0],dtype=int))
			# Transfer back to global coordinates
			ret = post_process(cfg, X_global, y_global, pred_traj_, pred_goal=pred_goal_, dist_traj=dist_traj_, dist_goal=dist_goal_)
			X_global_, y_global_, pred_goal_, pred_traj_, dist_traj_, dist_goal_ = ret

			# Sampling 1000 samples from the mixture
			xs = []
			ys = []
			for i in range(pred_traj_.shape[2]):
				xs.append(pred_traj_[0,11,i,0])
				ys.append(pred_traj_[0,11,i,1])

			xmin = 0
			xmax = test_bckgd.shape[1]
			ymin = 0
			ymax = test_bckgd.shape[0]
			xx, yy = np.mgrid[xmin:xmax:100j,ymin:ymax:100j]

			# Testing/visualization uncalibrated KDE
			image_grid      = np.vstack([xx.ravel(), yy.ravel()])
			world_grid      = world_to_image_xy(np.transpose(image_grid),test_homography,flip=flipImage)
			# Prediction samples
			world_samples   = np.vstack([xs, ys])
			image_samples   = world_to_image_xy(np.transpose(world_samples),homography_to_img,flip=flipImage)
			if world_samples.shape[1]>0:
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
			figs, axs = plt.subplots(1,2,figsize=(24,12),constrained_layout = True)
			axs[0].legend_ = None
			axs[0].imshow(test_bckgd)
			observations = world_to_image_xy(observations_test[traj_idx,:,:2], homography_to_img, flip=flipImage)
			groundtruth  = world_to_image_xy(target_test[traj_idx,:,:2], homography_to_img, flip=flipImage)
			if world_samples.shape[1]>0:
				# Contour plot
				cset = axs[0].contour(xx, yy, f_unc, colors='darkgreen',levels=levels[1:],linewidths=0.75)
				if len(cset.levels)>1:
					cset.levels = np.array(alphas[1:])
					axs[0].clabel(cset, cset.levels,fontsize=8)
			axs[0].plot(observations[:,0],observations[:,1],color='blue')
			axs[0].plot([observations[-1,0],groundtruth[0,0]],[observations[-1,1],groundtruth[0,1]],color='red')
			axs[0].plot(groundtruth[:,0],groundtruth[:,1],color='red')
			axs[0].set_xlim(xmin,xmax)
			axs[0].set_ylim(ymax,ymin)
			axs[0].axes.xaxis.set_visible(False)
			axs[0].axes.yaxis.set_visible(False)
			if world_samples.shape[1]>0:
				# Plot the pdf
				axs[0].imshow(transparency,alpha=np.sqrt(transparency),cmap=plt.cm.Greens_r,extent=[xmin, xmax, ymin, ymax])

			# Testing/visualization **calibrated** KDE
			# TODO: use the calibration
			#modified_alphas = alphas_samples
			modified_alphas = iso_inv.transform(alphas_samples)

			# New values for f
			fs_samples_new  = []
			for alpha in modified_alphas:
				fs_samples_new.append(get_falpha(sorted_samples,alpha))
			fs_samples_new    = np.array(fs_samples_new)
			importance_weights= fs_samples_new/fs_samples
			kde               = st.gaussian_kde(world_samples,weights=importance_weights)
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
			cset = axs[1].contour(xx, yy, f_cal, colors='darkgreen',levels=levels[1:],linewidths=0.75)
			if len(cset.levels)>1:
				cset.levels = np.array(alphas[1:])
				axs[1].clabel(cset, cset.levels,fontsize=8)
			axs[1].plot(observations[:,0],observations[:,1],color='blue')
			axs[1].plot([observations[-1,0],groundtruth[0,0]],[observations[-1,1],groundtruth[0,1]],color='red')
			axs[1].plot(groundtruth[:,0],groundtruth[:,1],color='red')
			axs[1].imshow(test_bckgd)
			axs[1].set_xlim(xmin,xmax)
			axs[1].set_ylim(ymax,ymin)
			axs[1].axes.xaxis.set_visible(False)
			axs[1].axes.yaxis.set_visible(False)
			axs[1].imshow(norm_f_cal,alpha=np.sqrt(transparency),cmap=plt.cm.Greens_r, extent=[xmin, xmax, ymin, ymax])
			plt.show()
