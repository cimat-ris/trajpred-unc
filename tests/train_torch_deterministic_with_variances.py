#!/usr/bin/env python
# coding: utf-8
# Autor: Mario Xavier Canche Uc
# Centro de Investigación en Matemáticas, A.C.
# mario.canche@cimat.mx

# Cargamos las librerias
import time
import sys,os,logging, argparse
sys.path.append('.')

import math,numpy as np
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
import pandas as pd

import torch
from torchvision import transforms
import torch.optim as optim

# Local models
from models.lstm_encdec import lstm_encdec_gaussian
from utils.datasets_utils import Experiment_Parameters, setup_loo_experiment, traj_dataset
from utils.train_utils import train
from utils.plot_utils import plot_traj_img,plot_traj_world,plot_cov_world
from utils.calibration import calibration
from utils.calibration import miscalibration_area, mean_absolute_calibration_error, root_mean_squared_calibration_error
import torch.optim as optim
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
parser.add_argument('--examples',
                    type=int, default=1, metavar='N',
                    help='number of examples to exhibit (default: 1)')
parser.add_argument('--id-test',
                    type=int, default=2, metavar='N',
                    help='id of the dataset to use as test in LOO (default: 2)')
parser.add_argument('--learning-rate', '--lr',
                    type=float, default=0.0004, metavar='N',
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
    model_name    = "deterministic_variances"

    # Load the dataset and perform the split
    training_data, validation_data, test_data, test_homography = setup_loo_experiment('ETH_UCY',dataset_dir,dataset_names,args.id_test,experiment_parameters,pickle_dir='pickle',use_pickled_data=args.pickle)
    # Torch dataset
    train_data = traj_dataset(training_data['obs_traj_rel'], training_data['pred_traj_rel'],training_data['obs_traj'], training_data['pred_traj'])
    val_data   = traj_dataset(validation_data['obs_traj_rel'], validation_data['pred_traj_rel'],validation_data['obs_traj'], validation_data['pred_traj'])
    test_data  = traj_dataset(test_data['obs_traj_rel'], test_data['pred_traj_rel'], test_data['obs_traj'], test_data['pred_traj'])

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
        model = lstm_encdec_gaussian(2,128,256,2)
        model.to(device)

        # Entremamos el modelo
        train(model,device,0,batched_train_data,batched_val_data,args,model_name)

    # Model instantiation
    model = lstm_encdec_gaussian(2,128,256,2)
    # Load the previously trained model
    model.load_state_dict(torch.load(TRAINING_CKPT_DIR+"/"+model_name+"_0"+"_"+str(args.id_test)+".pth"))
    model.eval()
    model.to(device)


    ind_sample = np.random.randint(args.batch_size)
    bck = plt.imread(os.path.join(dataset_dir,dataset_names[args.id_test],'reference.png'))

    # Testing
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
        plt.title('Trajectory samples')
        plt.show()
        # Solo aplicamos a un elemento del batch
        if batch_idx==args.examples-1:
            break

if __name__ == "__main__":
    main()
