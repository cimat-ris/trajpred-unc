#!/usr/bin/env python
# coding: utf-8
# Autor: Mario Xavier Canche Uc
# Centro de Investigación en Matemáticas, A.C.
# mario.canche@cimat.mx

# Imports
import time
import sys,os,logging, argparse
''' TF_CPP_MIN_LOG_LEVEL
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printeds
3 = INFO, WARNING, and ERROR messages are not printed
'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append('bayesian-torch')
sys.path.append('.')

from utils.calibration_utils import save_data_for_calibration

import math,numpy as np
import matplotlib as mpl
#mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
import pandas as pd

import torch
from torchvision import transforms

# Local models
from models.bayesian_models_gaussian_loss import lstm_encdec_variational
from utils.datasets_utils import get_ethucy_dataset
from utils.train_utils import train_variational
from utils.plot_utils import plot_traj_img, plot_traj_world, plot_cov_world
from utils.calibration import generate_one_batch_test
from utils.directory_utils import mkdir_p

# Local constants
from utils.constants import IMAGES_DIR, VARIATIONAL, TRAINING_CKPT_DIR

# parameters models
#initial_lr     = 0.000002

# Parser arguments
parser = argparse.ArgumentParser(description='')
parser.add_argument('--batch-size', '--b',
					type=int, default=256, metavar='N',
					help='input batch size for training (default: 256)')
parser.add_argument('--epochs', '--e',
					type=int, default=20, metavar='N',
					help='number of epochs to train (default: 200)')
parser.add_argument('--id-test',
					type=int, default=2, metavar='N',
					help='id of the dataset to use as test in LOO (default: 2)')
parser.add_argument('--num-mctrain',
					type=int, default=5, metavar='N',
					help='number of sample monte carlo for train (default: 5)')
parser.add_argument('--num-mctest',
					type=int, default=5, metavar='N',
					help='number of monte carlo for test (default: 5)')
parser.add_argument('--learning-rate', '--lr',
					type=float, default=0.0004, metavar='N',
					help='learning rate of optimizer (default: 1E-3)')
parser.add_argument('--no-retrain',
					action='store_true',
					help='do not retrain the model')
parser.add_argument('--pickle',
					action='store_true',
					help='use previously made pickle files')
parser.add_argument('--plot-losses',
					action='store_true',
					help='plot losses curves after training')
parser.add_argument('--show-plot', default=False,
					action='store_true', help='show the test plots')
parser.add_argument('--log-level',type=int, default=20,help='Log level (default: 20)')
parser.add_argument('--log-file',default='',help='Log file (default: standard output)')
args = parser.parse_args()


def main():
	# Printing parameters
	torch.set_printoptions(precision=2)
	logging.basicConfig(format='%(levelname)s: %(message)s',level=args.log_level)
	# Device
	if torch.cuda.is_available():
		logging.info(torch.cuda.get_device_name(torch.cuda.current_device()))
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	batched_train_data,batched_val_data,batched_test_data,homography,reference_image = get_ethucy_dataset(args)
	model_name    = "variational"

	# Seleccionamos una semilla
	seed = 1

	# TODO: Agregar la los argumentos
	prior_mu = 0.0
	prior_sigma = 1.0
	posterior_mu_init = 0.0
	posterior_rho_init = -4


	if args.no_retrain==False:
		# Train model for each seed
		# Seed added
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)

		# Instanciate the model
		model = lstm_encdec_variational(2,128,256,2,prior_mu,prior_sigma,posterior_mu_init,posterior_rho_init)
		model.to(device)

		# Train the model
		logging.info("Seeds: {}".format(seed))
		train_variational(model,device,args.id_test,batched_train_data,batched_val_data,args,model_name)
		if args.plot_losses:
			plt.savefig(IMAGES_DIR+"/loss_"+str(args.id_test)+".pdf")
			plt.show()


	# Instanciate the model
	model = lstm_encdec_variational(2,128,256,2,prior_mu,prior_sigma,posterior_mu_init,posterior_rho_init)
	model.to(device)


	ind_sample = np.random.randint(args.batch_size)

	# Load the previously trained model
	model.load_state_dict(torch.load(TRAINING_CKPT_DIR+"/"+model_name+"_"+str(args.id_test)+".pth"))
	model.eval()

	# Creamos la carpeta donde se guardaran las imagenes
	if not os.path.exists(IMAGES_DIR):
		os.makedirs(IMAGES_DIR)

	# Testing
	for batch_idx, (datarel_test, targetrel_test, data_test, target_test) in enumerate(batched_test_data):
		fig, ax = plt.subplots(1,1,figsize=(12,12))

		# For each element of the ensemble
		for ind in range(args.num_mctest):

			if torch.cuda.is_available():
				  datarel_test  = datarel_test.to(device)

			pred, kl, sigmas = model.predict(datarel_test, dim_pred=12)

			# Plotting
			plot_traj_world(pred[ind_sample,:,:], data_test[ind_sample,:,:], target_test[ind_sample,:,:], ax)
			plot_cov_world(pred[ind_sample,:,:],sigmas[ind_sample,:,:],data_test[ind_sample,:,:], ax)
		plt.legend()
		plt.title('Trajectory samples {}'.format(batch_idx))
		plt.savefig(IMAGES_DIR+"/pred_variational.pdf")
		if args.show_plot:
			plt.show()
		plt.close()
		# Solo aplicamos a un elemento del batch
		break


	# ## Calibramos la incertidumbre
	draw_ellipse = True

	#------------------ Obtenemos el batch unico de test para las curvas de calibracion ---------------------------
	datarel_test_full, targetrel_test_full, data_test_full, target_test_full, tpred_samples_full, sigmas_samples_full = generate_one_batch_test(batched_test_data, model, args.num_mctest, TRAINING_CKPT_DIR, model_name, id_test=args.id_test, device=device, type="variational")
	#---------------------------------------------------------------------------------------------------------------

	# Testing
	cont = 0
	for batch_idx, (datarel_test, targetrel_test, data_test, target_test) in enumerate(batched_test_data):

		tpred_samples = []
		sigmas_samples = []
		# Muestreamos con cada modelo
		for ind in range(args.num_mctest):

			if torch.cuda.is_available():
				  datarel_test  = datarel_test.to(device)

			pred, kl, sigmas = model.predict(datarel_test, dim_pred=12)

			tpred_samples.append(pred)
			sigmas_samples.append(sigmas)

		tpred_samples = np.array(tpred_samples)
		sigmas_samples = np.array(sigmas_samples)

		save_data_for_calibration(VARIATIONAL, tpred_samples, tpred_samples_full, data_test, data_test_full, target_test, target_test_full, targetrel_test, targetrel_test_full, sigmas_samples, sigmas_samples_full, args.id_test)


		# Solo se ejecuta para un batch
		break


if __name__ == "__main__":
	main()