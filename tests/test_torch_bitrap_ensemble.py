import pdb
import os
import sys
sys.path.append(os.path.realpath('.'))
sys.path.append('../../bidireaction-trajectory-prediction/')
sys.path.append('../../bidireaction-trajectory-prediction/datasets')
import torch
from torch import nn, optim
from torch.nn import functional as F

import pickle as pkl
from datasets import make_dataloader
from bitrap.modeling import make_model
from bitrap.engine.trainer import inference
from bitrap.utils.dataset_utils import restore
from bitrap.engine.utils import print_info, post_process

from bitrap.utils.logger import Logger
import logging

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
    parser.add_argument(
        "--config_file",
        default="",
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
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.BATCH_SIZE = 1
    cfg.TEST.BATCH_SIZE = 1
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    logger = logging.Logger("MPED_RNN")
    # get dataloaders
    test_dataloader = make_dataloader(cfg, 'test')
    nsamples = 100
    ensemble = []
    for i in range(5):
        cfg.CKPT_DIR = '../training_checkpoints/zara01-0{}.pth'.format(i+1)
        # build model, optimizer and scheduler
        model = make_model(cfg)
        model = model.to(cfg.DEVICE)
        if os.path.isfile(cfg.CKPT_DIR):
            model.load_state_dict(torch.load(cfg.CKPT_DIR))
            print(colored('Loaded checkpoint:{}'.format(cfg.CKPT_DIR), 'blue', 'on_green'))
        else:
            print(colored('The cfg.CKPT_DIR id not a file: {}'.format(cfg.CKPT_DIR), 'green', 'on_red'))
        #model.K = nsamples
        ensemble.append(model)

    # Generate all paths
    all_img_paths    = []
    all_X_globals    = []
    all_pred_trajs   = []
    all_gt_trajs     = []
    all_timesteps    = []

    with torch.set_grad_enabled(False):
        for iters, batch in enumerate(test_dataloader, start=1):
            X_global     = batch['input_x'].to(cfg.DEVICE)
            y_global     = batch['target_y']
            img_path     = batch['cur_image_file']
            resolution   = batch['pred_resolution'].numpy()
            input_x      = batch['input_x_st'].to(cfg.DEVICE)
            neighbors_st = restore(batch['neighbors_x_st'])
            neighbors_un = restore(batch['neighbors_x'])
            adjacency    = restore(batch['neighbors_adjacency'])
            first_history_indices = batch['first_history_index']
            pred_traj    = np.zeros((1,12,nsamples,2))
            for k in range(nsamples):
                id = random.randint(0,4)
                pred_goal_, pred_traj_, _, dist_goal_, dist_traj_ = ensemble[id](input_x,
                                                                neighbors_st=neighbors_st,
                                                                adjacency=adjacency,
                                                                z_mode=False,
                                                                cur_pos=X_global[:, -1, :cfg.MODEL.DEC_OUTPUT_DIM],
                                                                first_history_indices=first_history_indices)
                # transfer back to global coordinates
                ret = post_process(cfg, X_global, y_global, pred_traj_, pred_goal=pred_goal_, dist_traj=dist_traj_, dist_goal=dist_goal_)
                X_global_, y_global_, pred_goal_, pred_traj_, dist_traj_, dist_goal_ = ret
                traj_id = random.randint(0,model.K-1)
                pred_traj[0,:,k,:] = pred_traj_[0,:,traj_id,:]

            key =  list(neighbors_un.keys())[0]
            neighbors = neighbors_un[key][0]
            plt.figure()
            plt.plot(X_global_[0,:,0],X_global_[0,:,1],'red')
            for neighbor in neighbors:
                plt.plot(neighbor[:,0],neighbor[:,1],'blue')
            for sample in range(pred_traj[0].shape[1]):
                plt.plot([X_global_[0,-1,0],pred_traj[0,0,sample,0]],
                         [X_global_[0,-1,1],pred_traj[0,0,sample,1],],'green')
                plt.plot(pred_traj[0,:,sample,0],pred_traj[0,:,sample,1],'green')
            plt.axis('equal')
            plt.show()
            all_img_paths.extend(img_path)
            all_X_globals.append(X_global)
            all_pred_trajs.append(pred_traj)
            all_gt_trajs.append(y_global)
            all_timesteps.append(batch['timestep'].numpy())

        # Evaluate
        all_X_globals = np.concatenate(all_X_globals, axis=0)
        all_pred_trajs = np.concatenate(all_pred_trajs, axis=0)
        all_gt_trajs = np.concatenate(all_gt_trajs, axis=0)
        all_timesteps = np.concatenate(all_timesteps, axis=0)
