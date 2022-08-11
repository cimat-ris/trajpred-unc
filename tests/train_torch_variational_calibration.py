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
from models.bayesian_models_gaussian_loss import lstm_encdec_variational
from utils.datasets_utils import Experiment_Parameters, setup_loo_experiment, traj_dataset
from utils.plot_utils import plot_traj_img, plot_traj_world, plot_cov_world
from utils.calibration import calibration
from utils.calibration import miscalibration_area, mean_absolute_calibration_error, root_mean_squared_calibration_error

# Local constants
from utils.constants import OBS_TRAJ, OBS_TRAJ_VEL, PRED_TRAJ, PRED_TRAJ_VEL

# parameters models
#initial_lr     = 0.000002

# Parser arguments
parser = argparse.ArgumentParser(description='')
parser.add_argument('--batch-size', '--b',
                    type=int, default=64, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--epochs', '--e',
                    type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 200)')
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
parser.add_argument('--log-level',type=int, default=20,help='Log level (default: 20)')
parser.add_argument('--log-file',default='',help='Log file (default: standard output)')
args = parser.parse_args()



# Function to train the models
def train(model,device,idTest,train_data,val_data):
    # Optimizer
    # optimizer = optim.SGD(model.parameters(), lr=initial_lr)
    optimizer = optim.Adam(model.parameters(),lr=args.learning_rate, betas=(.5, .999),weight_decay=0.8)
    list_loss_train = []
    list_loss_val   = []
    min_val_error   = 1000.0

    for epoch in range(args.num_epochs):
        # Training
        print("----- ")
        print("epoch: ", epoch)
        error = 0
        total = 0
        M     = len(train_data)
        for batch_idx, (data, target, data_abs, target_abs) in enumerate(train_data):
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            if torch.cuda.is_available():
                data  = data.to(device)
                target=target.to(device)
                data_abs  = data_abs.to(device)
                target_abs=target_abs.to(device)

            # Step 2. Run our forward pass and compute the losses
            pred, nl_loss, kl_loss = model(data, target, data_abs , target_abs, num_mc=args.num_mctrain)
            
            # TODO: Divide by the batch size
            loss   = nl_loss+ kl_loss/M
            error += loss.detach().item()
            total += len(target)

            # Step 3. Compute the gradients, and update the parameters by
            loss.backward()
            optimizer.step()
        print("Trn loss: ",error/total)
        list_loss_train.append(error/total)

        # Validation
        error = 0
        total = 0
        M     = len(val_data)
        for batch_idx, (data_val, target_val, data_abs , target_abs) in enumerate(val_data):
            if torch.cuda.is_available():
                data_val  = data_val.to(device)
                target_val=target_val.to(device)
                data_abs  = data_abs.to(device)
                target_abs = target_abs.to(device)

            pred_val, nl_loss, kl_loss = model(data_val, target_val, data_abs , target_abs)
            pi     = (2.0**(M-batch_idx))/(2.0**M-1) # From Blundell
            loss   = nl_loss+ pi*kl_loss
            error += loss.detach().item()
            total += len(target_val)

        print("Val loss: ", error/total)
        list_loss_val.append(error/total)
        if (error/total)<min_val_error:
            min_val_error = error/total
            # Keep the model
            print("Saving model")
            torch.save(model.state_dict(), "training_checkpoints/model_variational_"+str(idTest)+".pth")

    
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

    # Device
    if torch.cuda.is_available():
        logging.info(torch.cuda.get_device_name(torch.cuda.current_device()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.basicConfig(format='%(levelname)s: %(message)s',level=20)
    # Load the default parameters
    experiment_parameters = Experiment_Parameters()

    dataset_dir   = "datasets/"
    dataset_names = ['eth-hotel','eth-univ','ucy-zara01','ucy-zara02','ucy-univ']
    idTest        = 2
    pickle        = False

    # Load the dataset and perform the split
    training_data, validation_data, test_data, test_homography = setup_loo_experiment('ETH_UCY',dataset_dir,dataset_names,idTest,experiment_parameters,pickle_dir='pickle',use_pickled_data=pickle)

    # Torch dataset
    train_data = traj_dataset(training_data[OBS_TRAJ_VEL], training_data[PRED_TRAJ_VEL],training_data[OBS_TRAJ], training_data[PRED_TRAJ])
    val_data = traj_dataset(validation_data[OBS_TRAJ_VEL], validation_data[PRED_TRAJ_VEL],validation_data[OBS_TRAJ], validation_data[PRED_TRAJ])
    test_data = traj_dataset(test_data[OBS_TRAJ_VEL], test_data[PRED_TRAJ_VEL], test_data[OBS_TRAJ], test_data[PRED_TRAJ])

    # Form batches
    batched_train_data = torch.utils.data.DataLoader( train_data, batch_size = batch_size, shuffle=False)
    batched_val_data =  torch.utils.data.DataLoader( val_data, batch_size = batch_size, shuffle=False)
    batched_test_data =  torch.utils.data.DataLoader( test_data, batch_size = batch_size, shuffle=False)
    # Seleccionamos una semilla
    seed = 1
    print("Seeds: ", seed)

    prior_mu = 0.0
    prior_sigma = 1.0
    posterior_mu_init = 0.0
    posterior_rho_init = -4


    if args.no_retrain==False:
        # Agregamos la semilla
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
        # Instanciate the model
        model = lstm_encdec_variational(2,128,256,2,prior_mu,prior_sigma,posterior_mu_init,posterior_rho_init)
        model.to(device)
        # Entremamos el modelo
        train(model,device,idTest,batched_train_data,batched_val_data)
        if args.plot_losses:
            plt.savefig("images/loss_"+str(idTest)+".pdf")
            plt.show()


    # Instanciamos el modelo
    model = lstm_encdec_variational(2,128,256,2,prior_mu,prior_sigma,posterior_mu_init,posterior_rho_init)
    model.to(device)

    # Cargamos el modelo
    model.load_state_dict(torch.load("training_checkpoints/model_variational_"+str(idTest)+".pth"))
    model.eval()


    ind_sample = np.random.randint(args.batch_size)
    bck = plt.imread(os.path.join(dataset_dir,dataset_names[idTest],'reference.png'))

    # Testing
    for batch_idx, (datarel_test, targetrel_test, data_test, target_test) in enumerate(batched_test_data):
        fig, ax = plt.subplots(1,1,figsize=(12,12))

        # For each element of the ensemble
        for ind in range(args.num_mctest):
            
            if torch.cuda.is_available():
                  datarel_test  = datarel_test.to(device)

            pred, kl, sigmas = model.predict(datarel_test, dim_pred=12)

            # ploting
            plot_traj_world(pred[ind_sample,:,:], data_test[ind_sample,:,:], target_test[ind_sample,:,:], ax)
            plot_cov_world(pred[ind_sample,:,:],sigmas[ind_sample,:,:],data_test[ind_sample,:,:], ax)
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
        for ind in range(args.num_mctest):

            if torch.cuda.is_available():
                  datarel_test  = datarel_test.to(device)

            pred, kl, sigmas = model.predict(datarel_test, dim_pred=12)

            tpred_samples.append(pred)
            sigmas_samples.append(sigmas)

        #plt.show()
        
        tpred_samples = np.array(tpred_samples)
        sigmas_samples = np.array(sigmas_samples)

        # HDR y Calibracion
        auc_cal, auc_unc, exp_proportions, obs_proportions_unc, obs_proportions_cal = calibration(tpred_samples, data_test, target_test, sigmas_samples, position = 11, alpha = 0.05, idTest=idTest)
        plt.show()

        # Solo se ejecuta para un batch
        break


    # ## Metrics Calibration

    ma1 = miscalibration_area(exp_proportions, obs_proportions_unc)
    mace1 = mean_absolute_calibration_error(exp_proportions, obs_proportions_unc)
    rmsce1 = root_mean_squared_calibration_error(exp_proportions, obs_proportions_unc)

    print("Before Recalibration:  ", end="")
    print("MACE: {:.5f}, RMSCE: {:.5f}, MA: {:.5f}".format(mace1, rmsce1, ma1))


    # In[ ]:


    ma2 = miscalibration_area(exp_proportions, obs_proportions_cal)
    mace2 = mean_absolute_calibration_error(exp_proportions, obs_proportions_cal)
    rmsce2 = root_mean_squared_calibration_error(exp_proportions, obs_proportions_cal)

    print("After Recalibration:  ", end="")
    print("MACE: {:.5f}, RMSCE: {:.5f}, MA: {:.5f}".format(mace2, rmsce2, ma2))


    df = pd.DataFrame([["","MACE","RMSCE","MA"],["Before Recalibration", mace1, rmsce1, ma1],["After Recalibration", mace2, rmsce2, ma2]])
    df.to_csv("images/metrics_calibration_"+str(idTest)+".csv")



if __name__ == "__main__":
    main()
