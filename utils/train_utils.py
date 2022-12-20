#!/usr/bin/env python
# coding: utf-8
# Autor: Mario Xavier Canche Uc
# Centro de Investigación en Matemáticas, A.C.
# mario.canche@cimat.mx

# Cargamos las librerias
import time
import sys,os,logging,argparse
sys.path.append('.')
import math,numpy as np
import matplotlib as mpl
#mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
import pandas as pd

import torch
from torchvision import transforms
import torch.optim as optim

# Local models
from utils.datasets_utils import Experiment_Parameters, setup_loo_experiment, traj_dataset
from utils.plot_utils import plot_traj_img,plot_traj_world,plot_cov_world
from utils.directory_utils import mkdir_p
import torch.optim as optim

# Local constants
from utils.constants import IMAGES_DIR, TRAINING_CKPT_DIR

# Function to train the models
# ind is the ensemble id in the case we use an ensemble (otherwise, it is equal to zero)
def train(model,device,ensemble_id,train_data,val_data,args,model_name):
	# Optimizer
	optimizer = optim.Adam(model.parameters(),lr=args.learning_rate, betas=(.5, .999),weight_decay=0.003)
	list_loss_train = []
	list_loss_val   = []
	min_val_error   = 1000.0
	for epoch in range(args.epochs):
		# Training
		logging.info("----- ")
		logging.info("Epoch: {}".format(epoch))
		error = 0
		total = 0
		model.train()
		# Recorremos cada batch
		for batch_idx, (observations_vel, target_vel, observations_abs , target_abs) in enumerate(train_data):
			# Remember that Pytorch accumulates gradients.
			# We need to clear them out before each instance
			model.zero_grad()
			if torch.cuda.is_available():
			  observations_vel = observations_vel.to(device)
			  target_vel       = target_vel.to(device)
			  observations_abs = observations_abs.to(device)
			  target_abs       = target_abs.to(device)

			# Run our forward pass and compute the loss
			loss   = model(observations_vel, target_vel, observations_abs , target_abs, teacher_forcing=args.teacher_forcing)
			error += loss.detach().cpu().numpy()
			total += len(target_vel)

			# Step 3. Compute the gradients, and update the parameters by
			loss.backward()
			optimizer.step()
		logging.info("Trn loss: {:.4f}".format(error/total))
		list_loss_train.append(error/total)

		# Validation
		error = 0
		total = 0
		ade   = 0
		fde   = 0
		model.eval()
		with torch.no_grad():
			for batch_idx, (data_val, target_val, data_abs , target_abs) in enumerate(val_data):

				if torch.cuda.is_available():
				  data_val  = data_val.to(device)
				  target_val= target_val.to(device)
				  data_abs  = data_abs.to(device)
				  target_abs= target_abs.to(device)

				loss_val = model(data_val, target_val, data_abs , target_abs)
				error   += loss_val.cpu().numpy()
				total   += len(target_val)
				# prediction
				init_pos = np.expand_dims(data_abs.cpu().numpy()[:,-1,:],axis=1)
				pred_val = model.predict(data_val, dim_pred=12)
				if len(pred_val)==2:
					pred_val = pred_val[0]
				pred_val += init_pos
				ade    = ade + np.sum(np.average(np.sqrt(np.square(target_abs.cpu().numpy()-pred_val).sum(2)),axis=1))
				fde    = fde + np.sum(np.sqrt(np.square(target_abs.cpu().numpy()[:,-1,:]-pred_val[:,-1,:]).sum(1)))

		error = error/total
		ade   = ade/total
		fde   = fde/total
		logging.info("Val loss: {:.4f} ".format(error))
		logging.info("Val ade : {:.4f} ".format(ade))
		logging.info("Val fde : {:.4f} ".format(fde))
		list_loss_val.append(error)
		if error<min_val_error:
			min_val_error = error
			# Keep the model
			logging.info("Saving model")
			torch.save(model.state_dict(), TRAINING_CKPT_DIR+"/"+model_name+"_"+str(ensemble_id)+"_"+str(args.id_test)+".pth")

	# Error visualization
	if args.plot_losses:
		# Create new directory
		output_dir = os.path.join(IMAGES_DIR, "loss_" + model_name)
		mkdir_p(output_dir)
		plt.figure(figsize=(12,12))
		plt.plot(list_loss_train, label="loss train")
		plt.plot(list_loss_val, label="loss val")
		plt.xlabel("Epochs")
		plt.ylabel("Loss")
		plt.legend()
		plt.savefig(os.path.join(output_dir , str(ind)+"_"+str(args.id_test)+".pdf"))
		plt.show()

# Function to train the models
def train_variational(model,device,idTest,train_data,val_data,args,model_name):
	# Optimizer
	# optimizer = optim.SGD(model.parameters(), lr=initial_lr)
	optimizer = optim.Adam(model.parameters(),lr=args.learning_rate, betas=(.5, .999),weight_decay=0.8)
	list_loss_train = []
	list_loss_val   = []
	min_val_error   = 1000.0

	for epoch in range(args.epochs):
		# Training
		logging.info("----- ")
		logging.info("Epoch: {}".format(epoch))
		error = 0
		total = 0
		M     = len(train_data)
		model.train()
		for batch_idx, (data, target, data_abs, target_abs) in enumerate(train_data):
			# Step 1. Remember that Pytorch accumulates gradients.
			# We need to clear them out before each instance
			model.zero_grad()

			if torch.cuda.is_available():
				data  = data.to(device)
				target=target.to(device)
				data_abs  = data_abs.to(device)
				target_abs=target_abs.to(device)

			# Step 2. Run our forward pass and compute the losses
			pred, nl_loss, kl_loss = model(data, target, data_abs , target_abs, num_mc=args.num_mctrain)

			# TODO: Divide by the batch size
			loss   = nl_loss+ kl_loss/M
			error += loss.detach().item()
			total += len(target)

			# Step 3. Compute the gradients, and update the parameters by
			loss.backward()
			optimizer.step()
		logging.info("Training loss: {:6.3f}".format(error/total))
		list_loss_train.append(error/total)

		# Validation
		error = 0
		total = 0
		M     = len(val_data)
		model.eval()
		with torch.no_grad():
			for batch_idx, (data_val, target_val, data_abs , target_abs) in enumerate(val_data):
				if torch.cuda.is_available():
					data_val  = data_val.to(device)
					target_val=target_val.to(device)
					data_abs  = data_abs.to(device)
					target_abs= target_abs.to(device)

				pred_val, nl_loss, kl_loss = model(data_val, target_val, data_abs , target_abs)
				pi     = (2.0**(M-batch_idx))/(2.0**M-1) # From Blundell
				loss   = nl_loss+ pi*kl_loss
				error += loss.detach().item()
				total += len(target_val)

		logging.info("Validation loss: {:6.3f}".format(error/total))
		list_loss_val.append(error/total)
		if (error/total)<min_val_error:
			min_val_error = error/total
			# Keep the model
			logging.info("Saving model")
			torch.save(model.state_dict(), TRAINING_CKPT_DIR+"/"+model_name+"_"+str(args.id_test)+".pth")


	# Visualizamos los errores
	plt.figure(figsize=(12,12))
	plt.plot(list_loss_train, label="loss train")
	plt.plot(list_loss_val, label="loss val")
	plt.xlabel("Epochs")
	plt.ylabel("Loss")
	plt.legend()
