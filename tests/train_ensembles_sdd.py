#!/usr/bin/env python
# coding: utf-8

# Imports
import time
import sys,os,logging, argparse

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
import torch.optim as optim

# Local models
from models.lstm_encdec import lstm_encdec_gaussian
from utils.datasets_utils import Experiment_Parameters, setup_loo_experiment, traj_dataset
from utils.train_utils import train
from utils.plot_utils import plot_traj_img,plot_traj_world,plot_cov_world
from utils.calibration import generate_one_batch_test
import torch.optim as optim
# Local constants
from utils.constants import OBS_TRAJ_VEL, PRED_TRAJ_VEL, OBS_TRAJ, PRED_TRAJ, REFERENCE_IMG, ENSEMBLES_SDD, TRAINING_CKPT_DIR

# Parser arguments
parser = argparse.ArgumentParser(description='')
parser.add_argument('--batch-size', '--b',
                    type=int, default=256, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--epochs', '--e',
                    type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--id-test',
                    type=int, default=2, metavar='N',
                    help='id of the dataset to use as test in LOO (default: 2)')
parser.add_argument('--max-overlap',
                    type=int, default=1, metavar='N',
                    help='Maximal overlap between trajets (default: 1)')
parser.add_argument('--num-ensembles',
                    type=int, default=5, metavar='N',
                    help='number of elements in the ensemble (default: 5)')
parser.add_argument('--learning-rate', '--lr',
                    type=float, default=0.0004, metavar='N',
                    help='learning rate of optimizer (default: 1E-3)')
parser.add_argument('--no-retrain',
                    action='store_true',
                    help='do not retrain the model')
parser.add_argument('--teacher-forcing',
                    action='store_true',
                    help='uses teacher forcing during training')
parser.add_argument('--show-plot', default=False,
                    action='store_true', help='show the test plots')
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
    logging.basicConfig(format='%(levelname)s: %(message)s',level=args.log_level)
    # Device
    if torch.cuda.is_available():
        logging.info(torch.cuda.get_device_name(torch.cuda.current_device()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the default parameters
    experiment_parameters = Experiment_Parameters(max_overlap=args.max_overlap)

    dataset_dir   = "datasets/sdd/sdd_data"
    dataset_names = ['bookstore', 'coupa', 'deathCircle', 'gates', 'hyang', 'little', 'nexus', 'quad']
    model_name    = "deterministic_variances_ens_sdd"

    # Load the dataset and perform the split
    training_data, validation_data, test_data, test_homography = setup_loo_experiment('ETH_UCY',dataset_dir,dataset_names,args.id_test,experiment_parameters,pickle_dir='pickle',use_pickled_data=args.pickle)

    # Torch dataset
    train_data = traj_dataset(training_data[OBS_TRAJ_VEL], training_data[PRED_TRAJ_VEL],training_data[OBS_TRAJ], training_data[PRED_TRAJ])
    val_data   = traj_dataset(validation_data[OBS_TRAJ_VEL], validation_data[PRED_TRAJ_VEL],validation_data[OBS_TRAJ], validation_data[PRED_TRAJ])
    test_data  = traj_dataset(test_data[OBS_TRAJ_VEL], test_data[PRED_TRAJ_VEL], test_data[OBS_TRAJ], test_data[PRED_TRAJ])

    # Form batches
    batched_train_data = torch.utils.data.DataLoader(train_data,batch_size=args.batch_size,shuffle=False)
    batched_val_data   = torch.utils.data.DataLoader(val_data,batch_size=args.batch_size,shuffle=False)
    batched_test_data  = torch.utils.data.DataLoader(test_data,batch_size=args.batch_size,shuffle=False)
    # Select random seeds
    seeds = np.random.choice(99999999, args.num_ensembles , replace=False)
    logging.info("Seeds: {}".format(seeds))

    if args.no_retrain==False:
        # Train model for each seed
        for ind, seed in enumerate(seeds):
            # Seed added
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

            # Instanciate the model
            model = lstm_encdec_gaussian(in_size=2, embedding_dim=128, hidden_dim=256, output_size=2)
            model.to(device)

            # Train the model
            print("\n*** Training for seed: ", seed, "\t\t ", ind, "/",len(seeds))
            train(model,device,ind,batched_train_data,batched_val_data,args,model_name)

    # Instanciate the model
    model = lstm_encdec_gaussian(in_size=2, embedding_dim=128, hidden_dim=256, output_size=2)
    model.to(device)


    ind_sample = np.random.randint(args.batch_size)
    bck = plt.imread(os.path.join(dataset_dir,dataset_names[args.id_test], REFERENCE_IMG))

    # Testing
    for batch_idx, (datarel_test, targetrel_test, data_test, target_test) in enumerate(batched_test_data):
        fig, ax = plt.subplots(1,1,figsize=(12,12))

        # For each element of the ensemble
        for ind in range(args.num_ensembles):
            # Load the previously trained model
            model.load_state_dict(torch.load(TRAINING_CKPT_DIR+"/"+model_name+"_"+str(ind)+"_"+str(args.id_test)+".pth"))

            model.eval()

            if torch.cuda.is_available():
                  datarel_test  = datarel_test.to(device)

            pred, sigmas = model.predict(datarel_test, dim_pred=12)
            # Plotting
            plot_traj_world(pred[ind_sample,:,:],data_test[ind_sample,:,:],target_test[ind_sample,:,:],ax)
            plot_cov_world(pred[ind_sample,:,:],sigmas[ind_sample,:,:],data_test[ind_sample,:,:],ax)
        plt.legend()
        plt.title('Trajectory samples {}'.format(batch_idx))
        if args.show_plot:
            plt.show()
        # Solo aplicamos a un elemento del batch
        break


    # ## Calibramos la incertidumbre
    draw_ellipse = True

    #------------------ Obtenemos el batch unico de test para las curvas de calibracion ---------------------------
    datarel_test_full, targetrel_test_full, data_test_full, target_test_full, tpred_samples_full, sigmas_samples_full = generate_one_batch_test(batched_test_data, model, args.num_ensembles, TRAINING_CKPT_DIR, model_name, id_test=args.id_test, device=device)
    #---------------------------------------------------------------------------------------------------------------

    # Testing
    cont = 0
    for batch_idx, (datarel_test, targetrel_test, data_test, target_test) in enumerate(batched_test_data):

        tpred_samples = []
        sigmas_samples = []
        # Muestreamos con cada modelo
        for ind in range(args.num_ensembles):

            # Cargamos el Modelo
            model.load_state_dict(torch.load(TRAINING_CKPT_DIR+"/"+model_name+"_"+str(ind)+"_"+str(args.id_test)+".pth"))
            model.eval()

            if torch.cuda.is_available():
                  datarel_test  = datarel_test.to(device)

            pred, sigmas = model.predict(datarel_test, dim_pred=12)

            tpred_samples.append(pred)
            sigmas_samples.append(sigmas)

        tpred_samples = np.array(tpred_samples)
        sigmas_samples = np.array(sigmas_samples)

        save_data_for_calibration(ENSEMBLES_SDD, tpred_samples, tpred_samples_full, data_test, data_test_full, target_test, target_test_full, targetrel_test, targetrel_test_full, sigmas_samples, sigmas_samples_full, args.id_test)

        # Solo se ejecuta para un batch y es usado como dataset de calibración
        break


if __name__ == "__main__":
    main()
