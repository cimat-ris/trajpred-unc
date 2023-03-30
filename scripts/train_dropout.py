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

# Local models
from models.bayesian_models_gaussian_loss import lstm_encdec_MCDropout
from utils.datasets_utils import get_dataset
from utils.plot_utils import plot_traj_img,plot_traj_world,plot_cov_world
from utils.calibration import generate_uncertainty_evaluation_dataset
from utils.calibration_utils import save_data_for_calibration
from utils.train_utils import train, evaluation_minadefde
from utils.config import get_config

# Local constants
from utils.constants import IMAGES_DIR, DROPOUT, TRAINING_CKPT_DIR, SUBDATASETS_NAMES

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
	batched_train_data,batched_val_data,batched_test_data,homography,reference_image = get_dataset(config)

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
	datarel_test_full, targetrel_test_full, data_test_full, target_test_full, tpred_samples_full, sigmas_samples_full = generate_uncertancertainty_evaluation_dataset(batched_test_data, model, config.dropout_samples, model_name, config, device=device, type="dropout_gaussian")
	print("tpred_samples_full.shape: ", tpred_samples_full.shape)
	evaluation_minadefde( model, tpred_samples_full, data_test_full, target_test_full, "dropout_gaussian")
	#---------------------------------------------------------------------------------------------------------------

	# Testing
	cont = 0
	for batch_idx, (datarel_test, targetrel_test, data_test, target_test) in enumerate(batched_test_data):

		tpred_samples = []
		sigmas_samples = []
		# Sampling from inference dropout
		for ind in range(config.dropout_samples):

			if torch.cuda.is_available():
				datarel_test  = datarel_test.to(device)

			pred, sigmas = model.predict(datarel_test, dim_pred=12)

			tpred_samples.append(pred)
			sigmas_samples.append(sigmas)

		tpred_samples = np.array(tpred_samples)
		sigmas_samples = np.array(sigmas_samples)
		pickle_filename = model_name+"_"+str(SUBDATASETS_NAMES[config.id_dataset][config.id_test])
		save_data_for_calibration(pickle_filename, tpred_samples, tpred_samples_full, data_test, data_test_full, target_test, target_test_full, targetrel_test, targetrel_test_full, sigmas_samples, sigmas_samples_full, config.id_test)

		# Solo se ejecuta para un batch
		break

if __name__ == "__main__":
	main()
