#!/usr/bin/env python
# coding: utf-8
# Autor: Mario Xavier Canche Uc
# Centro de Investigación en Matemáticas, A.C.
# mario.canche@cimat.mx

# Cargamos las librerias
import os,logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.optim as optim
from tqdm import tqdm
# Local constants
from trajpred_unc.utils.constants import TRAINING_CKPT_DIR, SUBDATASETS_NAMES
from trajpred_unc.utils.config import get_model_filename

# Function to train the models
# ind is the ensemble id in the case we use an ensemble (otherwise, it is equal to zero)
def train(model,device,ensemble_id,train_data,val_data,config):
	# Model name
	model_filename = get_model_filename(config,ensemble_id)
	# Optimizer
	optimizer = optim.Adam(model.parameters(),lr=config["train"]["initial_lr"],weight_decay=0.003)
	list_loss_train = []
	list_loss_val   = []
	min_val_error   = 1000.0
	for epoch in range(config["train"]["epochs"]):
		# Training
		error = total = 0
		model.train()
		# Cycle over batches
		for observations_vel,target_vel,observations_abs,target_abs,__,__,__ in tqdm(train_data):
			# Remember that Pytorch accumulates gradients.
			# We need to clear them out before each instance
			model.zero_grad()
			if torch.cuda.is_available():
				observations_vel = observations_vel.to(device)
				target_vel       = target_vel.to(device)
				observations_abs = observations_abs.to(device)
				target_abs       = target_abs.to(device)

			# Run our forward pass and compute the loss
			loss   = model(observations_vel, target_vel, observations_abs , target_abs, teacher_forcing=config["train"]["teacher_forcing"])
			error += loss.detach().cpu().numpy()
			total += len(target_vel)

			# Step 3. Compute the gradients, and update the parameters by
			loss.backward()
			optimizer.step()
		list_loss_train.append(error/total)

		# Validation
		error = total = ade = fde = 0
		model.eval()
		with torch.no_grad():
			for __, (observations_vel,target_vel,observations_abs,target_abs,__,__,__) in enumerate(val_data):
				if torch.cuda.is_available():
					observations_vel = observations_vel.to(device)
					target_vel       = target_vel.to(device)
					observations_abs = observations_abs.to(device)
					target_abs       = target_abs.to(device)
				loss_val = model(observations_vel,target_vel,observations_abs,target_abs)
				error   += loss_val.cpu().numpy()
				total   += len(target_vel)
				# Prediction is relative to the last observation
				init_pos = np.expand_dims(observations_abs.cpu().numpy()[:,-1,:],axis=1)
				pred_val = model.predict(observations_vel)
				if len(pred_val)==2:
					pred_val = pred_val[0]
				pred_val += init_pos
				ade    = ade + np.sum(np.average(np.sqrt(np.square(target_abs.cpu().numpy()-pred_val).sum(2)),axis=1))
				fde    = fde + np.sum(np.sqrt(np.square(target_abs.cpu().numpy()[:,-1,:]-pred_val[:,-1,:]).sum(1)))

		error = error/total
		ade   = ade/total
		fde   = fde/total
		logging.info("Epoch: {:03d} |Trn loss: {:8.6f} |Val loss: {:8.6f} |Val ade : {:6.4f} |Val fde : {:6.4f} ".format(epoch,list_loss_train[-1],error,ade,fde))
		list_loss_val.append(error)
		if error< min_val_error:
			min_val_error = error
        	# Save best checkpoints
			if not os.path.exists(config["train"]["save_dir"]):
			    # Create a new directory if it does not exist
				os.makedirs(config["save_dir"])
			save_path = config["train"]["save_dir"]+model_filename
			torch.save(model.state_dict(), save_path)

	# Error visualization
	if config["misc"]["plot_losses"]:
		output_file = os.path.join(config["misc"]["plot_dir"],"loss_" + config["train"]["model_name"]+"."+str(config["dataset"]["id_test"])+".pdf")
		if not os.path.exists(config["misc"]["plot_dir"]):
			# Create a new directory if it does not exist
			os.makedirs(config["misc"]["plot_dir"])
		plt.figure(figsize=(12,12))
		plt.plot(list_loss_train, label="loss train")
		plt.plot(list_loss_val, label="loss val")
		plt.xlabel("Epochs")
		plt.ylabel("Loss")
		plt.legend()
		plt.savefig(output_file)
		plt.show()

# Function to train the models
def train_variational(model,device,train_data,val_data,config):
	# Model name
	model_filename = get_model_filename(config,0)
	# Optimizer
	optimizer       = optim.Adam(model.parameters(),lr=config["train"]["initial_lr"],weight_decay=0.003)
	list_loss_train = []
	list_loss_val   = []
	min_val_error   = 1000.0

	for epoch in range(config["train"]["epochs"]):
		# Training
		logging.info("----- ")
		logging.info("Epoch: {}".format(epoch))
		error = 0
		total = 0
		M     = len(train_data)
		model.train()
		for observations_vel,target_vel,observations_abs,target_abs,__,__,__ in tqdm(train_data):

			# Step 1. Remember that Pytorch accumulates gradients.
			# We need to clear them out before each instance
			model.zero_grad()

			if torch.cuda.is_available():
				observations_vel  = observations_vel.to(device)
				target_vel=target_vel.to(device)
				observations_abs  = observations_abs.to(device)
				target_abs=target_abs.to(device)

			# Step 2. Run our forward pass and compute the losses
			__, nl_loss, kl_loss = model(observations_vel,target_vel,observations_abs,target_abs, num_mc=config["train"]["num_mctrain"])

			# Divide by the batch size
			loss   = nl_loss+ kl_loss/M
			error += loss.detach().item()
			total += len(target_vel)

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
			for batch_idx, (observations_vel,target_vel,observations_abs,target_abs,__,__,__) in enumerate(val_data):
				if torch.cuda.is_available():
					observations_vel  = observations_vel.to(device)
					target_vel=target_vel.to(device)
					observations_abs  = observations_abs.to(device)
					target_abs= target_abs.to(device)

				pred_val, nl_loss, kl_loss = model(observations_vel, target_vel, observations_abs , target_abs)
				pi     = (2.0**(M-batch_idx))/(2.0**M-1) # From Blundell
				loss   = nl_loss+ pi*kl_loss
				error += loss.detach().item()
				total += len(target_vel)

		logging.info("Validation loss: {:6.3f}".format(error/total))
		list_loss_val.append(error/total)
		if (error/total)<min_val_error:
			min_val_error = error/total
        	# Save best checkpoints
			if not os.path.exists(config["train"]["save_dir"]):
			    # Create a new directory if it does not exist
				os.makedirs(config["save_dir"])
			save_path = config["train"]["save_dir"]+model_filename
			torch.save(model.state_dict(), save_path)



	# Visualizamos los errores
	plt.figure(figsize=(12,12))
	plt.plot(list_loss_train, label="loss train")
	plt.plot(list_loss_val, label="loss val")
	plt.xlabel("Epochs")
	plt.ylabel("Loss")
	plt.legend()
	if config["misc"]["plot_losses"]:
		output_file = os.path.join(config["misc"]["plot_dir"],"loss_" + config["train"]["model_name"]+"."+str(config["dataset"]["id_test"])+".pdf")
		if not os.path.exists(config["misc"]["plot_dir"]):
			# Create a new directory if it does not exist
			os.makedirs(config["misc"]["plot_dir"])
		plt.savefig(output_file)
		plt.show()

	
# Perform quantitative evaluation
#def evaluation_minadefde(model,test_data,config):
def evaluation_minadefde(predictions_samples, data_test, target_test, model_name):
	logging.debug("----> Predictions: {}".format(predictions_samples.shape))
	logging.debug("----> Observations: {}".format(data_test.shape))
	logging.debug("----> Ground truth: {}".format(target_test.shape))
	# Last position
	last_pos = data_test[:,-1,:].detach().unsqueeze(1).unsqueeze(0).numpy()
	# All squared differences
	diff = target_test.detach().numpy() - (predictions_samples+last_pos)
	diff = diff**2
	# Euclidean distances
	diff = np.sqrt(np.sum(diff, axis=3))
	# minADEs for each data point 
	ade  = np.min(np.mean(diff,axis=2), axis=0)
	# minFDEs for each data point 
	fde  = np.min(diff[:,:,-1], axis=0)
	results = [["mADE", "mFDE"], [np.mean(ade), np.mean(fde)]]
    
	# Save results into a csv file
	output_csv_name = "images/calibration/" + model_name +"_min_ade_fde.csv"
	df = pd.DataFrame(results)
	df.to_csv(output_csv_name, mode='a', header=not os.path.exists(output_csv_name))
	print(df)
	return results	
