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

from utils.datasets_utils import Experiment_Parameters, setup_loo_experiment, traj_dataset_bitrap
import logging
# Local constants
from utils.constants import OBS_TRAJ_VEL, OBS_TRAJ_ACC, OBS_NEIGHBORS, PRED_TRAJ_VEL, OBS_TRAJ, PRED_TRAJ, TEST_BITRAP_BT, TRAINING_CKPT_DIR, REFERENCE_IMG

import argparse
from configs import cfg
from termcolor import colored
import numpy as np
import tqdm
import pdb
import matplotlib.pyplot as plt
import random

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--id-test',
                        type=int, default=2, metavar='N',
                        help='id of the dataset to use as test in LOO (default: 2)')
    parser.add_argument('--batch-size', '--b',
                        type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument(
        "--config_file",
        default="../bidireaction-trajectory-prediction/configs/bitrap_np_ETH.yml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
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
    dataset_dir   = "datasets/"
    dataset_names = ['eth-hotel','eth-univ','ucy-zara01','ucy-zara02','ucy-univ']
    model_name = 'model_deterministic'

    #### Data: BitTrap way
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.BATCH_SIZE = 1
    cfg.TEST.BATCH_SIZE = 1
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # Get dataloader
    test_dataloader = make_dataloader(cfg, 'test')
    env             = test_dataloader.dataset.dataset.env

    # Load a model
    cfg.CKPT_DIR = 'training_checkpoints/bitrap-zara01-0{}.pth'.format(1)
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
    training_data, validation_data, testing_data, test_homography = setup_loo_experiment('ETH_UCY',dataset_dir,dataset_names,args.id_test,experiment_parameters,pickle_dir='pickle',use_pickled_data=False,use_neighbors=True)
    # Torch dataset
    X_test            = np.concatenate([testing_data[OBS_TRAJ],testing_data[OBS_TRAJ_VEL],testing_data[OBS_TRAJ_ACC]],axis=2)
    test_data         = traj_dataset_bitrap(X_test,testing_data[OBS_NEIGHBORS],testing_data[PRED_TRAJ])
    # Form batches
    batched_test_data = torch.utils.data.DataLoader(test_data,batch_size=1,shuffle=False)

    with torch.set_grad_enabled(False):
        pred_traj  = np.zeros((model.K, len(batched_test_data), 12, 2))
        obs_traj   = np.zeros((len(batched_test_data), 8,2))
        gt_traj    = np.zeros((len(batched_test_data),12,2))
        gt_traj_rel= np.zeros((len(batched_test_data),12,2))
        for ind, (data_test, neighbors_test, target_test) in enumerate(batched_test_data):
            # Input: batch_sizex8x6 (un-normalized)
            X_global     = torch.Tensor(data_test).to(cfg.DEVICE)
            # This is the GT: batch_sizex12x2 (un-normalized)
            y_global     = torch.Tensor(target_test)
            # Input: batch_sizex8x6 (**normalized**)
            node_type = env.NodeType[0]
            state     = {'position':['x','y'], 'velocity':['x','y'], 'acceleration':['x','y']}
            _, std    = env.get_standardize_params(state, node_type)
            std[0:2]  = env.attention_radius[(node_type, node_type)]
            # Reference point: the last observed position, batch_size x 6
            rel_state       = np.zeros_like(data_test[:,0])
            rel_state[:,0:2]= np.array(data_test)[:,-1, 0:2]
            rel_state       = np.expand_dims(rel_state,axis=1)
            # Normalize the inputs
            input_x         = env.standardize(data_test,state,node_type,mean=rel_state,std=std)
            input_x         = torch.tensor(input_x,dtype=torch.float).to(cfg.DEVICE)
            # Neighbors
            input_neighbors = {}
            input_neighbors[(node_type,node_type)] = [[]]
            n_neighbors = len(neighbors_test)
            for neighbor in neighbors_test:
                neighbor_normalized = env.standardize(neighbor, state, node_type, mean=rel_state, std=std)
                input_neighbors[(node_type,node_type)][0].append(neighbor_normalized)
            # Should be a list of x tensors 8x6
            # Dictionary with keys pairs node_type x node_type
            input_adjacency = {}
            input_adjacency[(node_type,node_type)] = [torch.tensor(np.ones(n_neighbors))]
            # Sample a trajectory
            pred_goal_, pred_traj_, _, dist_goal_, dist_traj_ = model(input_x,neighbors_st=input_neighbors,adjacency=input_adjacency,z_mode=False,cur_pos=X_global[:,-1,:2],
                             first_history_indices=torch.tensor([0],dtype=int))
            # Transfer back to global coordinates
            ret = post_process(cfg, X_global, y_global, pred_traj_, pred_goal=pred_goal_, dist_traj=dist_traj_, dist_goal=dist_goal_)
            X_global_, y_global_, pred_goal_, pred_traj_, dist_traj_, dist_goal_ = ret
            pred_traj[:,ind,:,:] = np.swapaxes(pred_traj_[0,:,:,:], 0, 1)
            obs_traj[ind,:,:]    = data_test[0,:,:2].numpy()
            gt_traj[ind,:,:]     = target_test[0,:,:].numpy()
            gt_traj_rel[ind,:,:] = target_test[0,:,:].numpy() - data_test[0,-1,:2].numpy()

        #aaaaa
        tpred_samples      =  torch.tensor(pred_traj[:,:args.batch_size,:,:])
        tpred_samples_full =  torch.tensor(pred_traj[:,args.batch_size:,:,:])
        data_test          =  torch.tensor(obs_traj[:args.batch_size,:,:])
        data_test_full     =  torch.tensor(obs_traj[args.batch_size:,:,:])
        target_test        =  torch.tensor(gt_traj[:args.batch_size,:,:])
        target_test_full   =  torch.tensor(gt_traj[args.batch_size:,:,:])
        targetrel_test     =  torch.tensor(gt_traj_rel[:args.batch_size,:,:])
        targetrel_test_full=  torch.tensor(gt_traj_rel[args.batch_size:,:,:])
        save_data_for_calibration(TEST_BITRAP_BT, tpred_samples, tpred_samples_full, data_test, data_test_full, target_test, target_test_full, targetrel_test, targetrel_test_full, None, None, args.id_test)
