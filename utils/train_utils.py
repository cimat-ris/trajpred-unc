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

# Local constants
from utils.constants import IMAGES_DIR, TRAINING_CKPT_DIR, SUBDATASETS_NAMES

# Function to train the models
# ind is the ensemble id in the case we use an ensemble (otherwise, it is equal to zero)
def train(model,device,ensemble_id,train_data,val_data,config):
	# Model name
	model_name = config["train"]["model_name"].format(SUBDATASETS_NAMES[config["dataset"]["id_dataset"]][config["dataset"]["id_test"]],ensemble_id)
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
		for __, (observations_vel,target_vel,observations_abs,target_abs,__,__,__) in enumerate(train_data):
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
				pred_val = model.predict(observations_vel, dim_pred=12)
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
			save_path = config["train"]["save_dir"]+model_name
			torch.save(model.state_dict(), save_path)

	# Error visualization
	if config["misc"]["plot_losses"]:
		# Create new directory
		output_dir = os.path.join(IMAGES_DIR, "loss_" + model_name)
		if not os.path.exists(config["misc"]["plot_dir"]):
			# Create a new directory if it does not exist
			os.makedirs(config["misc"]["plot_dir"])
		plt.figure(figsize=(12,12))
		plt.plot(list_loss_train, label="loss train")
		plt.plot(list_loss_val, label="loss val")
		plt.xlabel("Epochs")
		plt.ylabel("Loss")
		plt.legend()
		plt.savefig(os.path.join(output_dir+str(config["dataset"]["id_test"])+".pdf"))
		plt.show()

# Function to train the models
def train_variational(model,device,idTest,train_data,val_data,args,model_name):
	# Optimizer
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

			# Divide by the batch size
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
			torch.save(model.state_dict(), TRAINING_CKPT_DIR+"/"+model_name+"_"+str(SUBDATASETS_NAMES[args.id_dataset][args.id_test])+"_0.pth")


	# Visualizamos los errores
	plt.figure(figsize=(12,12))
	plt.plot(list_loss_train, label="loss train")
	plt.plot(list_loss_val, label="loss val")
	plt.xlabel("Epochs")
	plt.ylabel("Loss")
	plt.legend()

	
# Perform quantitative evaluation
#def evaluation_minadefde(model,test_data,config):
def evaluation_minadefde( model, tpred_samples, data_test, target_test, model_name=''):
    l2dis = []
    
    print("----> tpred_samples.shape: ", tpred_samples.shape)
    print("----> data_test.shape: ", data_test.shape)
    print("----> target_test.shape: ", target_test.shape)
    for i in range(tpred_samples.shape[1]): # se mueve en las trayectorias del batch
        #normin = 1000.0
        normin = 999999999999.0
        diffmin= None
        for k in range(tpred_samples.shape[0]): # se mueve en las muestrass
            # Error for ade/fde
            diff = target_test[i,:,:].detach().numpy() - (tpred_samples[k,i,:,:]+data_test[i,-1,:].detach().numpy())
            #print(target_test[i,:,:].detach().numpy().shape)
            #print(tpred_samples[k,i,:,:].shape)
            #print(diff.shape)
            #print("-------")
            diff = diff**2
            diff = np.sqrt(np.sum(diff, axis=1))
            # To keep the min
            if np.linalg.norm(diff)<normin:
                normin  = np.linalg.norm(diff)
                diffmin = diff
        l2dis.append(diffmin)


    ade = [t for o in l2dis for t in o] # average displacement
    fde = [o[-1] for o in l2dis] # final displacement
    results = [["mADE", "mFDE"], [np.mean(ade), np.mean(fde)]]
    
    output_csv_name = "images/calibration/" + model_name +"_min_ade_fde.csv"
    df = pd.DataFrame(results)
    df.to_csv(output_csv_name, mode='a', header=not os.path.exists(output_csv_name))
    print(df)
        
    print(results)
    return results	
