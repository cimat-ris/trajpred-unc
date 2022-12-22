'''
'''
import pdb
import os
import sys
sys.path.append(os.path.realpath('.'))
sys.path.append('../bidireaction-trajectory-prediction/')
sys.path.append('../bidireaction-trajectory-prediction/datasets')

import numpy as np
import random, logging
import torch
from torch import nn, optim
from torch.nn import functional as F
import pickle as pkl
# sys.path.remove('/home/brianyao/Documents/intention2021icra')
from datasets import make_dataloader

from bitrap.modeling import make_model

from bitrap.engine import build_engine
from bitrap.utils.scheduler import ParamScheduler, sigmoid_anneal
from bitrap.utils.logger import Logger
import logging

import argparse
from configs import cfg
from collections import OrderedDict
import pdb

dataset_names  = ['hotel','eth','zara1','zara2','univ']
def build_optimizer(cfg, model):
	all_params = model.parameters()
	optimizer = optim.Adam(all_params, lr=cfg.SOLVER.LR)
	return optimizer

def main():
	parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
	parser.add_argument('--gpu', default='0', type=str)
	parser.add_argument('--seed', default=1, type=int)
	parser.add_argument(
		"--config_file",
		default="bitrap_np_ETH.yml",
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
	#cfg.DATASET.NAME = dataset_names[args.id_test]
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	# build model, optimizer and scheduler
	model = make_model(cfg)
	model = model.to(cfg.DEVICE)
	optimizer = build_optimizer(cfg, model)
	# NOTE: add separate optimizers to train single object predictor and interaction predictor

	torch.manual_seed(args.seed)
	random.seed(args.seed)
	np.random.seed(args.seed)

	if cfg.USE_WANDB:
		logger = Logger("FOL",
						cfg,
						project = cfg.PROJECT,
						viz_backend="wandb"
						)
	else:
		logger = logging.Logger("FOL")

	dataloader_params ={
			"batch_size": cfg.SOLVER.BATCH_SIZE,
			"shuffle": True,
			"num_workers": cfg.DATALOADER.NUM_WORKERS
			}

	# get dataloaders
	train_dataloader = make_dataloader(cfg, 'train')
	val_dataloader = make_dataloader(cfg, 'val')
	test_dataloader = make_dataloader(cfg, 'test')
	logging.info('Dataloader built!')
	# get train_val_test engines
	do_train, do_val, inference = build_engine(cfg)
	logging.info('Training engine built!')

	save_checkpoint_dir = cfg.CKPT_DIR
	if not os.path.exists(save_checkpoint_dir):
		os.makedirs(save_checkpoint_dir)

	# NOTE: hyperparameter scheduler
	model.param_scheduler = ParamScheduler()
	model.param_scheduler.create_new_scheduler(
										name='kld_weight',
										annealer=sigmoid_anneal,
										annealer_kws={
											'device': cfg.DEVICE,
											'start': 0,
											'finish': 100.0,
											'center_step': 400.0,
											'steps_lo_to_hi': 100.0,
										})

	model.param_scheduler.create_new_scheduler(
										name='z_logit_clip',
										annealer=sigmoid_anneal,
										annealer_kws={
											'device': cfg.DEVICE,
											'start': 0.05,
											'finish': 5.0,
											'center_step': 300.0,
											'steps_lo_to_hi': 300.0 / 5.
										})


	if cfg.SOLVER.scheduler == 'exp':
		# exponential schedule
		lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.SOLVER.GAMMA)
	elif cfg.SOLVER.scheduler == 'plateau':
		# Plateau scheduler
		lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=5,
															min_lr=1e-07, verbose=1)
	else:
		lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 40], gamma=0.2)

	logging.info('Schedulers built!')

	for epoch in range(cfg.SOLVER.MAX_EPOCH):
		logger.info("Epoch:{}".format(epoch))
		do_train(cfg, epoch, model, optimizer, train_dataloader, cfg.DEVICE, logger=logger, lr_scheduler=lr_scheduler)
		val_loss = do_val(cfg, epoch, model, val_dataloader, cfg.DEVICE, logger=logger)
		if (epoch+1) % 1 == 0:
			inference(cfg, epoch, model, test_dataloader, cfg.DEVICE, logger=logger, eval_kde_nll=False)

		# update LR
		if cfg.SOLVER.scheduler != 'exp':
			lr_scheduler.step(val_loss)
	torch.save(model.state_dict(), os.path.join(save_checkpoint_dir, 'bitrap-{}-{:02d}.pth'.format(cfg.DATASET.NAME,args.seed)))
if __name__ == '__main__':
	main()
