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
from utils.constants import DATASETS_DIR,SUBDATASETS_NAMES,OBS_TRAJ,OBS_TRAJ_VEL,OBS_TRAJ_ACC,OBS_NEIGHBORS,PRED_TRAJ,BITRAP,FRAMES_IDS
from utils.plot_utils import plot_traj_img,plot_traj_world,plot_cov_world,world_to_image_xy

import argparse
from configs import cfg
from termcolor import colored
import numpy as np
import tqdm
import pdb
import matplotlib.pyplot as plt
import random
import scipy.stats as st
from sklearn.cluster import DBSCAN

config_files = ["cfg/bitrap_np_hotel.yml","cfg/bitrap_np_eth.yml","cfg/bitrap_np_zara1.yml","cfg/bitrap_np_zara2.yml","cfg/bitrap_np_univ.yml"]

def get_cmap(n, name='Set2'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def metric_end(p):
	print(p)
	return np.linalg.norm(p)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
	parser.add_argument('--gpu', default='0', type=str)
	parser.add_argument('--seed', default=1, type=int)
	parser.add_argument('--id-test',
						type=int, default=2, metavar='N',
						help='id of the dataset to use as test in LOO (default: 2)')
	parser.add_argument('--batch-size', '--b',
						type=int, default=256, metavar='N',
						help='input batch size for training (default: 256)')
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
	cfg.CKPT_DIR = 'training_checkpoints/bitrap/bitrap-{}-{:02d}.pth'.format(cfg.DATASET.NAME,3)
	#cfg.CKPT_DIR = 'tmp/bitrap-{}-{:02d}.pth'.format(cfg.DATASET.NAME,0)
	# Build model, optimizer and scheduler
	model = make_model(cfg)
	model = model.to(cfg.DEVICE)
	if os.path.isfile(cfg.CKPT_DIR):
	   model.load_state_dict(torch.load(cfg.CKPT_DIR))
	   logging.info(colored('Loaded checkpoint:{}'.format(cfg.CKPT_DIR), 'blue', 'on_green'))
	else:
	   logging.info(colored('The cfg.CKPT_DIR id not a file: {}'.format(cfg.CKPT_DIR), 'green', 'on_red'))
	model.K = 1000

	flipImage  = False

	##################################################################
	# With our data (to show how to use BitTrap normalization)
	# Load the dataset and perform the split
	training_data, validation_data, testing_data, test_homography = setup_loo_experiment(DATASETS_DIR[0],SUBDATASETS_NAMES[0],args.id_test,experiment_parameters,pickle_dir='pickle',use_pickled_data=False, compute_neighbors=True)
	# Torch dataset
	X_test            = np.concatenate([testing_data[OBS_TRAJ],testing_data[OBS_TRAJ_VEL],testing_data[OBS_TRAJ_ACC]],axis=2)
	test_data         = traj_dataset_bitrap(X_test,testing_data[OBS_NEIGHBORS],testing_data[PRED_TRAJ],Frame_Ids=testing_data[FRAMES_IDS])

	for k in range(100):
		frame_id, batch, test_bckgd = get_testing_batch_bitrap(test_data,DATASETS_DIR[0]+SUBDATASETS_NAMES[0][args.id_test])
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

				xmin = 0
				xmax = test_bckgd.shape[1]
				ymin = 0
				ymax = test_bckgd.shape[0]

				# Sampling 1000 samples from the mixture
				xs = []
				ys = []
				for i in range(pred_traj_.shape[2]):
					for l in range(12):
						xs.append(pred_traj_[0,l,i,0])
						ys.append(pred_traj_[0,l,i,1])

				world_samples   = np.vstack([xs, ys])
				image_samples   = world_to_image_xy(np.transpose(world_samples),homography_to_img,flip=flipImage)

				observations   = world_to_image_xy(observations_test[traj_idx,:,:2], homography_to_img, flip=flipImage)
				groundtruth    = world_to_image_xy(target_test[traj_idx,:,:2], homography_to_img, flip=flipImage)
				alpha          = 0.8
				delta_epsilon  = 0.001
				epsilon_min    = 0.01
				epsilon_max    = 100.0
				epsilon        = epsilon_min
				n_samples      = model.K
				min_samples    = max(4,n_samples//50)
				n_outs_        = n_samples
				np.swapaxes(pred_traj_,1,2)
				X              = pred_traj_[0,11,:,:]
				# Binary search version
				while (abs(n_outs_-(1.0-alpha)*n_samples)>delta_epsilon and epsilon_max-epsilon_min>delta_epsilon):
					epsilon= 0.5*(epsilon_min+epsilon_max)
					db     = DBSCAN(eps=epsilon, min_samples=min_samples).fit(X)
					labels = db.labels_

					# Number of clusters in labels, ignoring noise if present.
					n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
					n_outs_     = list(labels).count(-1)

					logging.debug("Epsilon: %f" % epsilon)
					logging.debug("Epsilon min: %f" % epsilon_min)
					logging.debug("Epsilon max: %f" % epsilon_max)
					logging.debug("Estimated number of clusters: %d" % n_clusters_)
					logging.debug("Estimated number of noise points: %d" % n_outs_)
					if (n_outs_<(1.0-alpha)*n_samples):
						epsilon_max = epsilon
					else:
						if (n_outs_>(1.0-alpha)*n_samples):
							epsilon_min = epsilon
				logging.info("Estimated number of clusters: %d" % n_clusters_)
				logging.info("Estimated number of noise points: %d" % n_outs_)
				cmap = get_cmap(n_clusters_)
				# Determine if GT is out-of-distribution
				core   = db.core_sample_indices_
				insider= False
				for j in range(len(core)):
					d = np.linalg.norm(target_test[traj_idx,11,:2]-X[j,:])
					if d<epsilon:
						insider = True
						break
				if insider:
					logging.info("Insider")
					continue
				else:
					logging.info("Outsider")
				figs, axs = plt.subplots(1,1,figsize=(12,12),constrained_layout = True)
				axs.legend_ = None
				axs.imshow(test_bckgd)
					
				for i in range(pred_traj_.shape[2]):
					if labels[i]==-1:
						axs.plot(image_samples[12*i:12*(i+1),0],image_samples[12*i:12*(i+1),1],color='black',alpha=0.1)
					else:
						color = cmap(i)
						axs.plot(image_samples[12*i:12*(i+1),0],image_samples[12*i:12*(i+1),1],color=color,alpha=0.1)
				axs.plot(observations[:,0],observations[:,1],color='blue')
				axs.plot([observations[-1,0],groundtruth[0,0]],[observations[-1,1],groundtruth[0,1]],color='red')
				axs.plot(groundtruth[:,0],groundtruth[:,1],color='red')
				axs.set_xlim(xmin,xmax)
				axs.set_ylim(ymax,ymin)
				axs.axes.xaxis.set_visible(False)
				axs.axes.yaxis.set_visible(False)
				plt.show()
