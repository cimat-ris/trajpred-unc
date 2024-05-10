'''
'''
import os
import sys
sys.path.append('../bitrap/')
sys.path.append('../bitrap/datasets')

import numpy as np
import random, logging
import torch
from torch import optim
from torch.nn import functional as F
from datasets import make_dataloader

from bitrap.modeling import make_model
from bitrap.engine import build_engine
from bitrap.utils.dataset_utils import restore
from bitrap.engine.utils import post_process
from bitrap.engine.evaluate import evaluate_multimodal
from trajpred_unc.uncertainties.calibration_utils import save_data_for_uncertainty_calibration
from trajpred_unc.utils.constants import SUBDATASETS_NAMES
import logging
from termcolor import colored
from tqdm import tqdm
import argparse
from configs import cfg
from collections import OrderedDict
import pdb
config_files  = ["cfg/bitrap_np_hotel.yml","cfg/bitrap_np_eth.yml","cfg/bitrap_np_zara1.yml","cfg/bitrap_np_zara2.yml","cfg/bitrap_np_univ.yml"]
dataset_names = ['hotel','eth','zara1','zara2','univ']

def main():
	parser = argparse.ArgumentParser(description="")
	parser.add_argument('--gpu', default='0', type=str)
	parser.add_argument('--seed', default=1, type=int)
	parser.add_argument('--log-level',type=int, default=20,help='Log level (default: 20)')
	parser.add_argument('--id-test',
						type=int, default=2, metavar='N',
						help='id of the dataset to use as test in LOO (default: 0)')
	parser.add_argument('--plot',
						action='store_true',
						help='show a few examples')
	parser.add_argument(
		"opts",
		help="Modify config options using the command-line",
		default=None,
		nargs=argparse.REMAINDER,
	)
	args = parser.parse_args()
	# Loggin format
	logging.basicConfig(format='%(levelname)s: %(message)s',level=args.log_level)
	logger = logging.getLogger('FOL')
	logger.setLevel(level=args.log_level)
	logger.info("Getting configuration")
	cfg.merge_from_file(config_files[args.id_test])
	cfg.merge_from_list(args.opts)
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

	# Load a BitTrap trained model
	cfg.CKPT_DIR = 'training_checkpoints/bitrap/bitrap-{}-{:02d}.pth'.format(cfg.DATASET.NAME,args.seed)
	# Build model, optimizer and scheduler
	model = make_model(cfg)
	model = model.to(cfg.DEVICE)
	if os.path.isfile(cfg.CKPT_DIR):
		model.load_state_dict(torch.load(cfg.CKPT_DIR))
		logging.info(colored('Loaded checkpoint:{}'.format(cfg.CKPT_DIR), 'blue', 'on_green'))
	else:
		logging.info(colored('The cfg.CKPT_DIR id not a file: {}'.format(cfg.CKPT_DIR), 'green', 'on_red'))
	logging.info(colored('Model loaded', 'blue', 'on_yellow'))
	# Number of samples for the multimodal prediction
	model.K = 100

	torch.manual_seed(args.seed)
	random.seed(args.seed)
	np.random.seed(args.seed)

	# Get test dataloader
	logger.info('Test data loader')
	test_dataloader = make_dataloader(cfg, 'test')
	# get train_val_test engines
	model.eval()
	all_img_paths  = []
	all_X_globals  = []
	all_pred_goals = []
	all_gt_goals   = []
	all_pred_trajs = []
	all_gt_trajs   = []
	all_distributions = []
	all_timesteps     = []
    
	with torch.set_grad_enabled(False):
		# Test over all the test dataset
		for batch in tqdm(test_dataloader):
			X_global = batch['input_x'].to(cfg.DEVICE)
			y_global = batch['target_y']
			img_path = batch['cur_image_file']
			input_x               = batch['input_x_st'].to(cfg.DEVICE)
			neighbors_st          = restore(batch['neighbors_x_st'])
			adjacency             = restore(batch['neighbors_adjacency'])
			first_history_indices = batch['first_history_index']
			pred_goal, pred_traj, _, dist_goal, dist_traj = model(input_x, neighbors_st=neighbors_st,
                                                                adjacency=adjacency,
                                                                z_mode=False, 
                                                                cur_pos=X_global[:,-1,:cfg.MODEL.DEC_OUTPUT_DIM],
                                                                first_history_indices=first_history_indices)
            # transfer back to global coordinates
			ret = post_process(cfg, X_global, y_global, pred_traj, pred_goal=pred_goal, dist_traj=dist_traj, dist_goal=dist_goal)
			X_global, y_global, pred_goal, pred_traj, dist_traj, dist_goal = ret
			
			all_img_paths.extend(img_path)
			all_X_globals.append(X_global)
			all_pred_goals.append(pred_goal)
			all_pred_trajs.append(pred_traj)
			all_gt_goals.append(y_global[:, -1])
			all_gt_trajs.append(y_global)
			all_timesteps.append(batch['timestep'].numpy())
			if dist_traj is not None:
				all_distributions.append(dist_traj)
			else:
				all_distributions.append(dist_goal)

			# Plot the last trajectory and prediction to be sure that everything is fine
			if args.plot:
				import matplotlib.pyplot as plt
				plt.figure()
				ind = np.random.randint(0,batch['input_x'].shape[0])
				for i in range(pred_traj.shape[2]):
					plt.plot(pred_traj[ind,:,i,0],pred_traj[ind,:,i,1],'b')
				plt.plot(X_global[ind,:,0],X_global[ind,:,1],'r')
				plt.plot(y_global[ind,:,0],y_global[ind,:,1],'g')
				plt.show()
	
        
        # Evaluate
		all_X_globals  = np.concatenate(all_X_globals, axis=0)
		all_pred_goals = np.concatenate(all_pred_goals, axis=0)
		all_pred_trajs = np.concatenate(all_pred_trajs, axis=0)
		all_gt_goals   = np.concatenate(all_gt_goals, axis=0)
		all_gt_trajs   = np.concatenate(all_gt_trajs, axis=0)
		all_timesteps  = np.concatenate(all_timesteps, axis=0)
		if hasattr(all_distributions[0], 'mus'):
			distribution = model.GMM(torch.cat([d.input_log_pis for d in all_distributions], axis=0),
                                    torch.cat([d.mus for d in all_distributions], axis=0),
                                    torch.cat([d.log_sigmas for d in all_distributions], axis=0),
                                    torch.cat([d.corrs for d in all_distributions], axis=0))
		else:
			distribution = None 
		mode = 'point'	
		eval_results = evaluate_multimodal(all_pred_trajs, all_gt_trajs, mode=mode, distribution=distribution, bbox_type=cfg.DATASET.BBOX_TYPE)
		for key, value in eval_results.items():
			info = "Testing prediction {}:{}".format(key, str(np.around(value, decimals=3)))
			if hasattr(logger, 'info'):
				logger.info(info)
			else:
				print(info)

	# Save the data for uncertainty calibration
	all_pred_trajs	=  np.swapaxes(all_pred_trajs, 1, 2)
	pred_samples    =  torch.tensor(all_pred_trajs)
	observations    =  torch.tensor(all_X_globals)[:,:,:2]
	gts             =  torch.tensor(all_gt_trajs)
	pickle_filename = cfg.METHOD+"_"+SUBDATASETS_NAMES[0][args.id_test]
	save_data_for_uncertainty_calibration(pickle_filename,pred_samples,observations,gts,None,args.id_test)


if __name__ == '__main__':
	main()
