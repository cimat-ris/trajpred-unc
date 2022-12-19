#!/usr/bin/env python
# coding: utf-8

# Imports
import time
import sys,os,logging, argparse
sys.path.append('.')

import math,numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import torch
from torchvision import transforms
import torch.optim as optim

# Local models
from models.lstm_encdec import lstm_encdec
from utils.datasets_utils import get_ethucy_dataset
from utils.plot_utils import plot_traj_img
from utils.train_utils import train

# Local constants
from utils.constants import TRAINING_CKPT_DIR

# Parser arguments
parser = argparse.ArgumentParser(description='')
parser.add_argument('--batch-size', '--b',
					type=int, default=256, metavar='N',
					help='input batch size for training (default: 256)')
parser.add_argument('--epochs', '--e',
					type=int, default=80, metavar='N',
					help='number of epochs to train (default: 100)')
parser.add_argument('--examples',
					type=int, default=1, metavar='N',
					help='number of examples to exhibit (default: 1)')
parser.add_argument('--id-test',
					type=int, default=2, metavar='N',
					help='id of the dataset to use as test in LOO (default: 2)')
parser.add_argument('--learning-rate', '--lr',
					type=float, default=0.0002, metavar='N',
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

	batched_train_data,batched_val_data,batched_test_data,homography,reference_image = get_ethucy_dataset(args)
	model_name    = 'deterministic'

	# Seed for RNG
	seed = 17
	if args.no_retrain==False:
		# Choose seed
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)
		np.random.seed(seed)

		# Instanciate the model
		model = lstm_encdec(in_size=2, embedding_dim=128, hidden_dim=128, output_size=2)
		model.to(device)

		# Train the model
		train(model,device,0,batched_train_data,batched_val_data,args,model_name)

	# Model instantiation
	model = lstm_encdec(in_size=2, embedding_dim=128, hidden_dim=128, output_size=2)
	# Load the previously trained model
	model_filename = TRAINING_CKPT_DIR+"/"+model_name+"_0"+"_"+str(args.id_test)+".pth"
	logging.info("Loading {}".format(model_filename))
	model.load_state_dict(torch.load(model_filename))
	model.to(device)
	model.eval()

	ind_sample = 1

	# Testing
	for batch_idx, (datarel_test, targetrel_test, data_test, target_test) in    enumerate(batched_test_data):
		if torch.cuda.is_available():
			datarel_test  = datarel_test.to(device)

		# Prediction
		pred = model.predict(datarel_test, dim_pred=12)

		# Plotting
		plt.figure(figsize=(12,12))
		plt.imshow(reference_image)
		plot_traj_img(pred[ind_sample,:,:], data_test[ind_sample,:,:], target_test[ind_sample,:,:], homography, reference_image)
		plt.legend()
		plt.title('Trajectory samples {}'.format(batch_idx))
		if args.show_plot:
			plt.show()
		# Not display more than args.examples
		if batch_idx==args.examples-1:
			break

	# Testing: Quantitative
	ade  = 0
	fde  = 0
	total= 0
	for batch_idx, (datavel_test, targetvel_test, data_test, target_test) in enumerate(batched_test_data):
		if torch.cuda.is_available():
			datavel_test  = datavel_test.to(device)
		total += len(datavel_test)
		# prediction
		init_pos  = np.expand_dims(data_test[:,-1,:],axis=1)
		pred_test = model.predict(datavel_test, dim_pred=12) + init_pos
		ade    += np.average(np.sqrt(np.square(target_test-pred_test).sum(2)),axis=1).sum()
		fde    += (np.sqrt(np.square(target_test[:,-1,:]-pred_test[:,-1,:]).sum(1))).sum()
	logging.info("Test ade : {:.4f} ".format(ade/total))
	logging.info("Test fde : {:.4f} ".format(fde/total))
if __name__ == "__main__":
	main()
