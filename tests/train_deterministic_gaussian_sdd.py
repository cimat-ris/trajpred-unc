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
from utils.datasets_utils import Experiment_Parameters, setup_loo_experiment, traj_dataset
from utils.train_utils import train
from utils.plot_utils import plot_traj_img, plot_traj_world, plot_cov_world
from utils.calibration import generate_uncertainty_evaluation_dataset
from utils.calibration_utils import save_data_for_calibration
from utils.directory_utils import mkdir_p
from utils.config import get_config
import torch.optim as optim
# Local constants
from utils.constants import IMAGES_DIR, OBS_TRAJ_VEL, PRED_TRAJ_VEL, OBS_TRAJ, PRED_TRAJ, REFERENCE_IMG, TRAINING_CKPT_DIR, DETERMINISTIC_GAUSSIAN_SDD, DATASETS_DIR, SUBDATASETS_NAMES


# Parser arguments
config = get_config()

def main():
	# Printing parameters
	torch.set_printoptions(precision=2)
	# Loggin format
	logging.basicConfig(format='%(levelname)s: %(message)s',level=config.log_level)
	# Device
	if torch.cuda.is_available():
		logging.info(torch.cuda.get_device_name(torch.cuda.current_device()))
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Load the default parameters, TODO: review parameters for SDD dataset
	experiment_parameters = Experiment_Parameters(max_overlap=config.max_overlap)

	model_name    = "deterministic_variances"

	# Load the dataset and perform the split
	training_data, validation_data, test_data, _ = setup_loo_experiment(DATASETS_DIR[1],SUBDATASETS_NAMES[1],config.id_test,experiment_parameters,pickle_dir='pickle',use_pickled_data=config.pickle, validation_proportion=config.validation_proportion, compute_neighbors=False)
	# Torch dataset
	train_data = traj_dataset(training_data[OBS_TRAJ_VEL], training_data[PRED_TRAJ_VEL],training_data[OBS_TRAJ], training_data[PRED_TRAJ])
	val_data   = traj_dataset(validation_data[OBS_TRAJ_VEL], validation_data[PRED_TRAJ_VEL],validation_data[OBS_TRAJ], validation_data[PRED_TRAJ])
	test_data  = traj_dataset(test_data[OBS_TRAJ_VEL], test_data[PRED_TRAJ_VEL], test_data[OBS_TRAJ], test_data[PRED_TRAJ])

	# Form batches
	batched_train_data = torch.utils.data.DataLoader(train_data,batch_size=config.batch_size,shuffle=False)
	batched_val_data   = torch.utils.data.DataLoader(val_data,batch_size=config.batch_size,shuffle=False)
	batched_test_data  = torch.utils.data.DataLoader(test_data,batch_size=config.batch_size,shuffle=False)

	# Seed for RNG
	seed = 1

	if config.no_retrain==False:
		# Choose seed
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)

		# Instanciate the model
		model = lstm_encdec_gaussian(in_size=2, embedding_dim=128, hidden_dim=256, output_size=2)
		model.to(device)

		# Train the model
		train(model,device,0,batched_train_data,batched_val_data,config,model_name)

	# Model instantiation
	model = lstm_encdec_gaussian(in_size=2, embedding_dim=128, hidden_dim=256, output_size=2)

	# Load the previously trained model
	model_filename = TRAINING_CKPT_DIR+"/"+model_name+"_"+str(SUBDATASETS_NAMES[config.id_dataset][config.id_test])+"_0.pth"
	logging.info("Loading {}".format(model_filename))
	model.load_state_dict(torch.load(model_filename))
	model.eval()
	model.to(device)

	ind_sample = np.random.randint(config.batch_size)

	output_dir = os.path.join(IMAGES_DIR)
	mkdir_p(output_dir)

	# Testing
	for batch_idx, (datarel_test, targetrel_test, data_test, target_test) in enumerate(batched_test_data):
		fig, ax = plt.subplots(1,1,figsize=(12,12))

		if torch.cuda.is_available():
			datarel_test  = datarel_test.to(device)

		pred, sigmas = model.predict(datarel_test, dim_pred=12)
		# Plotting
		ind = np.minimum(ind_sample,pred.shape[0]-1)
		plot_traj_world(pred[ind,:,:],data_test[ind,:,:],target_test[ind,:,:],ax)

		plt.legend()
		plt.savefig(os.path.join(output_dir , "pred_dropout"+".pdf"))
		if config.show_plot:
			plt.show()
		plt.close()
		print(datarel_test.shape)
		# Not display more than config.examples
		if batch_idx==config.examples-1:
			break
	#------------------ Obtenemos el batch unico de test para las curvas de calibracion ---------------------------
	datarel_test_full, targetrel_test_full, data_test_full, target_test_full, tpred_samples_full, sigmas_samples_full = generate_uncertainty_evaluation_dataset(batched_test_data, model, 1, model_name, config, device=device)
	#---------------------------------------------------------------------------------------------------------------

	# Testing
	cont = 0
	for batch_idx, (datarel_test, targetrel_test, data_test, target_test) in enumerate(batched_test_data):

		tpred_samples = []
		sigmas_samples = []

		# Cargamos el Modelo
		model_filename = TRAINING_CKPT_DIR+"/"+model_name+"_"+str(SUBDATASETS_NAMES[config.id_dataset][config.id_test])+"_0.pth"
		logging.info("Loading {}".format(model_filename))
		model.load_state_dict(torch.load(model_filename))

		model.eval()

		if torch.cuda.is_available():
			datarel_test  = datarel_test.to(device)

		pred, sigmas = model.predict(datarel_test, dim_pred=12)

		tpred_samples.append(pred)
		sigmas_samples.append(sigmas)

		tpred_samples = np.array(tpred_samples)
		sigmas_samples = np.array(sigmas_samples)

		save_data_for_calibration(DETERMINISTIC_GAUSSIAN_SDD, tpred_samples, tpred_samples_full, data_test, data_test_full, target_test, target_test_full, targetrel_test, targetrel_test_full, sigmas_samples, sigmas_samples_full, config.id_test)
		# Solo se ejecuta para un batch y es usado como dataset de calibración
		break

if __name__ == "__main__":
	main()
