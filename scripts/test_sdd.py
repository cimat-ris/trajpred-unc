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
from utils.datasets_utils import get_dataset
from utils.train_utils import train
from utils.plot_utils import plot_traj_img, plot_traj_world, plot_cov_world
from utils.calibration_utils import save_data_for_calibration
from utils.directory_utils import mkdir_p
from utils.config import load_config,get_model_name
import torch.optim as optim

# Load configuration file (conditional model)
config = load_config("deterministic_gaussian_sdd.yaml")

def main():
    # Printing parameters
    torch.set_printoptions(precision=2)
    # Loggin format
    logging.basicConfig(format='%(levelname)s: %(message)s',level=config["misc"]["log_level"])
    # Device
    if torch.cuda.is_available():
        logging.info(torch.cuda.get_device_name(torch.cuda.current_device()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batched_train_data,batched_val_data,batched_test_data,__,__ = get_dataset(config["dataset"])
    
    # Instanciate the model
    model = lstm_encdec_gaussian(config["model"])
    model.to(device)

	# May not have to retrain the model
    if config["train"]["no_retrain"]==False:
	    # Train the model
        train(model,device,0,batched_train_data,batched_val_data,config)

	# Load the previously trained model
    model_filename = config["train"]["save_dir"]+get_model_name(config)
    logging.info("Loading {}".format(model_filename))
    model.load_state_dict(torch.load(model_filename))
    model.to(device)
    model.eval()
    ind_sample = 1
	# Testing
    for batch_idx, (observations_vel,__,observations_abs,target_abs,__,__,__) in enumerate(batched_test_data):
        if torch.cuda.is_available():
            observations_vel  = observations_vel.to(device)
        predicted_positions,sigmas_positions = model.predict(observations_vel)
		# Plotting
        ind = np.minimum(ind_sample,predicted_positions.shape[0]-1)
        __, ax = plt.subplots(1,1,figsize=(12,12))
        plot_traj_world(predicted_positions[ind,:,:],observations_abs[ind,:,:],target_abs[ind,:,:],ax)
        plot_cov_world(predicted_positions[ind,:,:],sigmas_positions[ind,:,:],observations_abs[ind,:,:],ax)        
        print(observations_vel[ind])
        # Plotting
        plt.legend('')
        plt.title('Trajectory samples {}'.format(batch_idx))
        if config["misc"]["show_test"]:
            plt.show()
		# Not display more than args.examples
        if batch_idx==config["misc"]["samples_test"]-1:
            break
if __name__ == "__main__":
    main()
