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
from models.bayesian_models_gaussian_loss import lstm_encdec
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
parser.add_argument('--id-test',
                    type=int, default=2, metavar='N',
                    help='id of the dataset to use as test in LOO (default: 2)')
parser.add_argument('--num-ensembles',
                    type=int, default=5, metavar='N',
                    help='number of elements in the ensemble (default: 5)')
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
parser.add_argument('--log-level',type=int, default=20,help='Log level (default: 20)')
parser.add_argument('--log-file',default='',help='Log file (default: standard output)')
args = parser.parse_args()


def main():
    # Printing parameters
    torch.set_printoptions(precision=2)

    # Device
    if torch.cuda.is_available():
        logging.info(torch.cuda.get_device_name(torch.cuda.current_device()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.basicConfig(format='%(levelname)s: %(message)s',level=args.log_level)
    # Load the default parameters
    experiment_parameters = Experiment_Parameters(add_kp=False,obstacles=False)

    dataset_dir   = "datasets/"
    dataset_names = ['eth-hotel','eth-univ','ucy-zara01','ucy-zara02','ucy-univ']
    model_name    = "deterministic_variances_ens"

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
    # Seleccionamos de forma aleatorea las semillas
    seeds = np.random.choice(99999999, args.num_ensembles , replace=False)
    print("Seeds: ", seeds)

    if args.no_retrain==False:
        # Entrenamos el modelo para cada semilla
        for ind, seed in enumerate(seeds):
            # Agregamos la semilla
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

            # Instanciate the model
            model = lstm_encdec(2,128,256,2)
            model.to(device)

            # Entremamos el modelo
            print("\n*** Training for seed: ", seed, "\t\t ", ind, "/",len(seeds))
            train(model,device,ind,batched_train_data,batched_val_data,args,model_name)

    # Instanciamos el modelo
    model = lstm_encdec(2,128,256,2)
    model.to(device)


    ind_sample = np.random.randint(args.batch_size)
    bck = plt.imread(os.path.join(dataset_dir,dataset_names[args.id_test],'reference.png'))

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
        plt.title('Trajectory samples')
        plt.show()
        # Solo aplicamos a un elemento del batch
        break


    # ## Calibramos la incertidumbre
    draw_ellipse = True

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

        #plt.show()

        tpred_samples = np.array(tpred_samples)
        sigmas_samples = np.array(sigmas_samples)

        # HDR y Calibracion
        auc_cal, auc_unc, exp_proportions, obs_proportions_unc, obs_proportions_cal = calibration(tpred_samples, data_test, target_test, sigmas_samples, position = 11, alpha = 0.05, idTest=args.id_test)
        plt.show()

        # Solo se ejecuta para un batch
        break


    # ## Metrics Calibration

    ma1    = miscalibration_area(exp_proportions, obs_proportions_unc)
    mace1  = mean_absolute_calibration_error(exp_proportions, obs_proportions_unc)
    rmsce1 = root_mean_squared_calibration_error(exp_proportions, obs_proportions_unc)

    print("Before Recalibration:  ", end="")
    print("MACE: {:.5f}, RMSCE: {:.5f}, MA: {:.5f}".format(mace1, rmsce1, ma1))


    # In[14]:


    ma2 = miscalibration_area(exp_proportions, obs_proportions_cal)
    mace2 = mean_absolute_calibration_error(exp_proportions, obs_proportions_cal)
    rmsce2 = root_mean_squared_calibration_error(exp_proportions, obs_proportions_cal)

    print("After Recalibration:  ", end="")
    print("MACE: {:.5f}, RMSCE: {:.5f}, MA: {:.5f}".format(mace2, rmsce2, ma2))


    df = pd.DataFrame([["","MACE","RMSCE","MA"],["Before Recalibration", mace1, rmsce1, ma1],["After Recalibration", mace2, rmsce2, ma2]])
    df.to_csv("images/metrics_calibration_"+str(args.id_test)+".csv")



if __name__ == "__main__":
    main()
