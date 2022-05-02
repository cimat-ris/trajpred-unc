#!/usr/bin/env python
# coding: utf-8

# Imports
import time
import sys,os,logging, argparse
sys.path.append('.')

import math,numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
from torchvision import transforms
import torch.optim as optim

# Local models
from models.lstm_encdec import lstm_encdec
from utils.datasets_utils import Experiment_Parameters, setup_loo_experiment, traj_dataset
from utils.plot_utils import plot_traj_img
from utils.train_utils import train

# Local constants
from utils.constants import OBS_TRAJ_REL, PRED_TRAJ_REL, OBS_TRAJ, PRED_TRAJ, TRAINING_CKPT_DIR

# Parser arguments
parser = argparse.ArgumentParser(description='')
parser.add_argument('--batch-size', '--b',
                    type=int, default=256, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--epochs', '--e',
                    type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--id-test',
                    type=int, default=2, metavar='N',
                    help='id of the dataset to use as test in LOO (default: 2)')
parser.add_argument('--learning-rate', '--lr',
                    type=float, default=0.0003, metavar='N',
                    help='learning rate of optimizer (default: 1E-3)')
parser.add_argument('--no-retrain',
                    action='store_true',
                    help='do not retrain the model')
parser.add_argument('--teacher-forcing',
                    action='store_true',
                    help='uses teacher forcing during training')
parser.add_argument('--pickle',
                    action='store_true',
                    help='use previously made pickle files')
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

    # Load the default parameters
    experiment_parameters = Experiment_Parameters(add_kp=False,obstacles=False)

    dataset_dir   = "datasets/"
    dataset_names = ['eth-hotel','eth-univ','ucy-zara01','ucy-zara02','ucy-univ']
    model_name = 'model_deterministic'

    # Load the dataset and perform the split
    training_data, validation_data, test_data, test_homography = setup_loo_experiment('ETH_UCY',dataset_dir,dataset_names,args.id_test,experiment_parameters,pickle_dir='pickle',use_pickled_data=args.pickle)
    # Torch dataset
    train_data = traj_dataset(training_data[OBS_TRAJ_REL ], training_data[PRED_TRAJ_REL],training_data[OBS_TRAJ], training_data[PRED_TRAJ])
    val_data = traj_dataset(validation_data[OBS_TRAJ_REL ], validation_data[PRED_TRAJ_REL],validation_data[OBS_TRAJ], validation_data[PRED_TRAJ])
    test_data = traj_dataset(test_data[OBS_TRAJ_REL ], test_data[PRED_TRAJ_REL], test_data[OBS_TRAJ], test_data[PRED_TRAJ])

    # Form batches
    batched_train_data = torch.utils.data.DataLoader(train_data,batch_size=args.batch_size,shuffle=False)
    batched_val_data   = torch.utils.data.DataLoader(val_data,batch_size=args.batch_size,shuffle=False)
    batched_test_data  = torch.utils.data.DataLoader(test_data,batch_size=args.batch_size,shuffle=False)

    # Seed for RNG
    seed = 1

    if args.no_retrain==False:
        # Agregamos la semilla
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # Instanciate the model
        model = lstm_encdec(2,128,256,2)
        model.to(device)

        # Entremamos el modelo
        train(model,device,0,batched_train_data,batched_val_data,args,model_name)

    # Instanciamos el modelo
    model = lstm_encdec(2,128,256,2)
    # Load the previously trained model
    model.load_state_dict(torch.load(TRAINING_CKPT_DIR+"/"+model_name+"_0"+"_"+str(args.id_test)+".pth"))
    model.to(device)
    model.eval()

    ind_sample = 1
    bck = plt.imread(os.path.join(dataset_dir,dataset_names[args.id_test],'reference.png'))

    # Testing
    for batch_idx, (datarel_test, targetrel_test, data_test, target_test) in    enumerate(batched_test_data):

        if torch.cuda.is_available():
            datarel_test  = datarel_test.to(device)

        # prediction
        pred = model.predict(datarel_test, dim_pred=12)

        # Plotting
        plt.figure(figsize=(12,12))
        plt.imshow(bck)
        plot_traj_img(pred[ind_sample,:,:], data_test[ind_sample,:,:], target_test[ind_sample,:,:], test_homography, bck)
        plt.legend()
        plt.title('Trajectory samples')
        plt.show()
        # Solo aplicamos a un elemento del batch
        break

if __name__ == "__main__":
    main()
