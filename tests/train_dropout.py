#!/usr/bin/env python
# coding: utf-8
# Autor: Mario Xavier Canche Uc
# Centro de Investigación en Matemáticas, A.C.
# mario.canche@cimat.mx

# Cargamos las librerias
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

import math,numpy as np
import matplotlib as mpl
#mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
import pandas as pd

import torch
from torchvision import transforms
import torch.optim as optim

# Local models
from models.bayesian_models_gaussian_loss import lstm_encdec_MCDropout
from utils.datasets_utils import get_ethucy_dataset
from utils.plot_utils import plot_traj_img,plot_traj_world,plot_cov_world
from utils.calibration import generate_one_batch_test
from utils.calibration_utils import save_data_for_calibration
from utils.train_utils import train
import torch.optim as optim

# Local constants
from utils.constants import IMAGES_DIR, DROPOUT, TRAINING_CKPT_DIR, SUBDATASETS_NAMES

# Parser arguments
parser = argparse.ArgumentParser(description='')
parser.add_argument('--batch-size', '--b',
					type=int, default=256, metavar='N',
					help='input batch size for training (default: 256)')
parser.add_argument('--epochs', '--e',
					type=int, default=100, metavar='N',
					help='number of epochs to train (default: 200)')
parser.add_argument('--id-dataset',
					type=str, default=0, metavar='N',
					help='id of the dataset to use. 0 is ETH-UCY, 1 is SDD (default: 0)')
parser.add_argument('--id-test',
					type=int, default=2, metavar='N',
					help='id of the dataset to use as test in LOO (default: 2)')
parser.add_argument('--mc',
					type=int, default=100, metavar='N',
					help='number of elements in the ensemble (default: 100)')
parser.add_argument('--dropout-rate',
					type=int, default=0.5, metavar='N',
					help='dropout rate (default: 0.5)')
parser.add_argument('--teacher-forcing',
					action='store_true',
					help='uses teacher forcing during training')					
parser.add_argument('--learning-rate', '--lr',
					type=float, default=0.0004, metavar='N',
					help='learning rate of optimizer (default: 1E-3)')
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

model_name = 'deterministic_dropout'

# TODO: move it to train_utils?
# Function to train the models
def train2(model,device,idTest,train_data,val_data):
	# Optimizer
	optimizer = optim.Adam(model.parameters(),lr=args.learning_rate, betas=(.5, .999),weight_decay=0.8)
	list_loss_train = []
	list_loss_val   = []
	min_val_error   = 1000.0
	for epoch in range(args.epochs):
		# Training
		logging.info("Epoch: {}".format(epoch))
		error = 0
		total = 0
		model.train()
		# Recorremos cada batch
		for batch_idx, (data, target, data_abs , target_abs) in enumerate(train_data):
			# Remember that Pytorch accumulates gradients.
			# We need to clear them out bDefinefore each instance
			model.zero_grad()
			if torch.cuda.is_available():
				data  = data.to(device)
				target=target.to(device)
				data_abs  = data_abs.to(device)
				target_abs=target_abs.to(device)

			# Run our forward pass and compute the loss
			loss   = model(data, target, data_abs , target_abs)# , training=True)
			error += loss
			total += len(target)

			# Step 3. Compute the gradients, and update the parameters by
			loss.backward()
			optimizer.step()
		logging.info("Training loss: {:6.3f}".format(error.detach().cpu().numpy()/total))
		list_loss_train.append(error.detach().cpu().numpy()/total)

		# Validation
		error = 0
		total = 0
		model.eval()
		with torch.no_grad():
			for batch_idx, (data_val, target_val, data_abs , target_abs) in enumerate(val_data):
				if torch.cuda.is_available():
					data_val  = data_val.to(device)
					target_val = target_val.to(device)
					data_abs  = data_abs.to(device)
					target_abs = target_abs.to(device)
				loss_val = model(data_val, target_val, data_abs , target_abs)
				error += loss_val.cpu().numpy()
				total += len(target_val)
		error = error/total
		logging.info("Validation loss: {:6.3f}".format(error))
		list_loss_val.append(error)
		if error<min_val_error:
			min_val_error = error
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


def main():
	# Printing parameters
	torch.set_printoptions(precision=2)
	logging.basicConfig(format='%(levelname)s: %(message)s',level=args.log_level)
	# Device
	if torch.cuda.is_available():
		logging.info(torch.cuda.get_device_name(torch.cuda.current_device()))
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	batched_train_data,batched_val_data,batched_test_data,homography,reference_image = get_ethucy_dataset(args)

	# Seleccionamos de forma aleatorea las semillas
	seed = 1
	logging.info("Seeds: {}".format(seed))



	if args.no_retrain==False:
		# Seed for RNG
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)
		# Instantiate the model
		model = lstm_encdec_MCDropout(2,128,256,2, dropout_rate = args.dropout_rate)
		model.to(device)
		# Train the model
		train(model,device,0,batched_train_data,batched_val_data,args,model_name)

		if args.plot_losses:
			plt.savefig(IMAGES_DIR+"/loss_"+str(idTest)+".pdf")
			plt.show()


	# Instanciamos el modelo
	model = lstm_encdec_MCDropout(2,128,256,2, dropout_rate = args.dropout_rate)
	model.to(device)
	# Load the previously trained model
	file_name = TRAINING_CKPT_DIR+"/"+model_name+"_"+str(SUBDATASETS_NAMES[args.id_dataset][args.id_test])+"_0.pth"
	model.load_state_dict(torch.load(file_name))
	model.eval()


	ind_sample = np.random.randint(args.batch_size)

	# Testing
	for batch_idx, (datarel_test, targetrel_test, data_test, target_test) in enumerate(batched_test_data):
		fig, ax = plt.subplots(1,1,figsize=(12,12))
		if ind_sample>data_test.shape[0]:
			continue
		# For each element of the ensemble
		for ind in range(args.mc):
			if torch.cuda.is_available():
				datarel_test  = datarel_test.to(device)

			pred, sigmas = model.predict(datarel_test, dim_pred=12)
			# Plotting
			plot_traj_world(pred[ind_sample,:,:],data_test[ind_sample,:,:],target_test[ind_sample,:,:],ax)
		plt.legend()
		plt.title('Trajectory samples')
		if args.show_plot:
			plt.show()

	# ## Calibramos la incertidumbre
	draw_ellipse = True

	#------------------ Obtenemos el batch unico de test para las curvas de calibracion ---------------------------
	datarel_test_full, targetrel_test_full, data_test_full, target_test_full, tpred_samples_full, sigmas_samples_full = generate_one_batch_test(batched_test_data, model, args.mc, model_name, args, device=device, type="dropout")
	#---------------------------------------------------------------------------------------------------------------

	# Testing
	cont = 0
	for batch_idx, (datarel_test, targetrel_test, data_test, target_test) in enumerate(batched_test_data):

		tpred_samples = []
		sigmas_samples = []
		# Sampling from inference dropout
		for ind in range(args.mc):

			if torch.cuda.is_available():
				datarel_test  = datarel_test.to(device)

			pred, sigmas = model.predict(datarel_test, dim_pred=12)

			tpred_samples.append(pred)
			sigmas_samples.append(sigmas)

		tpred_samples = np.array(tpred_samples)
		sigmas_samples = np.array(sigmas_samples)

		save_data_for_calibration(DROPOUT, tpred_samples, tpred_samples_full, data_test, data_test_full, target_test, target_test_full, targetrel_test, targetrel_test_full, sigmas_samples, sigmas_samples_full, args.id_test)

		# Solo se ejecuta para un batch
		break

if __name__ == "__main__":
	main()
