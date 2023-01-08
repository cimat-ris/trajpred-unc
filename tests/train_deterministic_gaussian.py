#!/usr/bin/env python
# coding: utf-8
# Autor: Mario Xavier Canche Uc
# Centro de Investigación en Matemáticas, A.C.
# mario.canche@cimat.mx

# Imports
import time
import sys,os,logging, argparse

sys.path.append('.')

import math,numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
import torch.optim as optim

# Local models
from models.lstm_encdec import lstm_encdec_gaussian
from utils.datasets_utils import get_ethucy_dataset
from utils.train_utils import train
from utils.plot_utils import plot_traj_img, plot_traj_world, plot_cov_world
from utils.calibration import generate_uncertainty_evaluation_dataset
from utils.calibration_utils import save_data_for_calibration
from utils.directory_utils import mkdir_p
import torch.optim as optim
# Local constants
from utils.constants import IMAGES_DIR,TRAINING_CKPT_DIR, DETERMINISTIC_GAUSSIAN, SUBDATASETS_NAMES


# Parser arguments
parser = argparse.ArgumentParser(description='')
parser.add_argument('--batch-size', '--b',
					type=int, default=256, metavar='N',
					help='input batch size for training (default: 256)')
parser.add_argument('--epochs', '--e',
					type=int, default=200, metavar='N',
					help='number of epochs to train (default: 200)')
parser.add_argument('--examples',
					type=int, default=1, metavar='N',
					help='number of examples to exhibit (default: 1)')
parser.add_argument('--id-dataset',
					type=str, default=0, metavar='N',
					help='id of the dataset to use. 0 is ETH-UCY, 1 is SDD (default: 0)')
parser.add_argument('--id-test',
					type=int, default=2, metavar='N',
					help='id of the dataset to use as test in LOO (default: 2)')
parser.add_argument('--learning-rate', '--lr',
					type=float, default=0.0004, metavar='N',
					help='learning rate of optimizer (default: 1E-3)')
parser.add_argument('--teacher-forcing',
					action='store_true',
					help='uses teacher forcing during training')
parser.add_argument('--no-retrain',
					action='store_true',
					help='do not retrain the model')
parser.add_argument('--pickle',
					action='store_true',
					help='use previously made pickle files')
parser.add_argument('--show-plot', default=False,
					action='store_true', help='show the test plots')
parser.add_argument('--plot-losses',
					action='store_true',
					help='plot losses curves after training')
parser.add_argument('--log-level',type=int, default=20,help='Log level (default: 20)')
parser.add_argument('--log-file',default='',help='Log file (default: standard output)')
args = parser.parse_args()


def main():
	# Printing parameters
	torch.set_printoptions(precision=2)
	# Loggin format
	logging.basicConfig(format='%(levelname)s: %(message)s',level=args.log_level)

	# Device
	if torch.cuda.is_available():
		logging.info(torch.cuda.get_device_name(torch.cuda.current_device()))
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Get the ETH-UCY data
	batched_train_data,batched_val_data,batched_test_data,homography,reference_image = get_ethucy_dataset(args)
	model_name    = "deterministic_variances"
	# Seed for RNG
	seed = 1

	# Training
	if args.no_retrain==False:
		# Choose seed
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)
		# Instanciate the model
		model = lstm_encdec_gaussian(in_size=2, embedding_dim=128, hidden_dim=256, output_size=2)
		model.to(device)
		# Train the model
		train(model,device,0,batched_train_data,batched_val_data,args,model_name)

	# Model instantiation
	model = lstm_encdec_gaussian(in_size=2, embedding_dim=128, hidden_dim=256, output_size=2)
	# Load the previously trained model
	model_filename = TRAINING_CKPT_DIR+"/"+model_name+"_"+str(SUBDATASETS_NAMES[args.id_dataset][args.id_test])+"_0.pth"
	logging.info("Loading {}".format(model_filename))
	model.load_state_dict(torch.load(model_filename))
	model.eval()
	model.to(device)


	output_dir = os.path.join(IMAGES_DIR)
	mkdir_p(output_dir)

	# Testing a random trajectory index in all batches
	ind_sample = np.random.randint(args.batch_size)
	for batch_idx, (datarel_test, targetrel_test, data_test, target_test) in enumerate(batched_test_data):
		fig, ax = plt.subplots(1,1,figsize=(12,12))

		if torch.cuda.is_available():
			datarel_test  = datarel_test.to(device)

		pred, sigmas = model.predict(datarel_test, dim_pred=12)
		# Plotting
		ind = np.minimum(ind_sample,pred.shape[0]-1)
		plot_traj_world(pred[ind,:,:],data_test[ind,:,:],target_test[ind,:,:],ax)
		plot_cov_world(pred[ind,:,:],sigmas[ind,:,:],data_test[ind,:,:],ax)
		plt.legend()
		plt.savefig(os.path.join(output_dir , "pred_dropout"+".pdf"))
		if args.show_plot:
			plt.show()
		plt.close()
		# Not display more than args.examples
		if batch_idx==args.examples-1:
			break

	#------------------ Generates sub-dataset for calibration evaluation ---------------------------
	datarel_test_full, targetrel_test_full, data_test_full, target_test_full, tpred_samples_full, sigmas_samples_full = generate_uncertainty_evaluation_dataset(batched_test_data, model, 1, model_name, args, device=device)
	#---------------------------------------------------------------------------------------------------------------

	# Producing data for uncertainty calibration
	for batch_idx, (datarel_test, targetrel_test, data_test, target_test) in enumerate(batched_test_data):

		tpred_samples  = []
		sigmas_samples = []

		if torch.cuda.is_available():
			datarel_test  = datarel_test.to(device)

		pred, sigmas = model.predict(datarel_test, dim_pred=12)
		tpred_samples.append(pred)
		sigmas_samples.append(sigmas)
		# Only the first batch is used as the calibration dataset
		break

	tpred_samples = np.array(tpred_samples)
	sigmas_samples = np.array(sigmas_samples)
	# Save these testing data for uncertainty calibration
	save_data_for_calibration(DETERMINISTIC_GAUSSIAN, tpred_samples, tpred_samples_full, data_test, data_test_full, target_test, target_test_full, targetrel_test, targetrel_test_full, sigmas_samples, sigmas_samples_full, args.id_test)


if __name__ == "__main__":
	main()
