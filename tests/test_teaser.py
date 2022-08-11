#!/usr/bin/env python
# coding: utf-8
# Autor: Mario Xavier Canche Uc
# Centro de Investigación en Matemáticas, A.C.
# mario.canche@cimat.mx

# Imports
import time
import sys,os,logging, argparse
sys.path.append('bayesian-torch')
sys.path.append('.')

import math,numpy as np
import matplotlib as mpl
import matplotlib.patches as patches
#mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import cv2
import torch
from torchvision import transforms
import torch.optim as optim
import scipy.stats as st

# Local models
from models.lstm_encdec import lstm_encdec_gaussian
from utils.datasets_utils import Experiment_Parameters, setup_loo_experiment, traj_dataset
from utils.train_utils import train
from utils.plot_utils import plot_traj_img,plot_traj_world,plot_cov_world,world_to_image_xy
from utils.calibration import calibrate_IsotonicReg, generate_one_batch_test
from utils.calibration import generate_metrics_calibration_conformal, generate_newKDE
from utils.hdr import get_alpha,get_alpha_bs,get_falpha,sort_sample
import torch.optim as optim
# Local constants
from utils.constants import OBS_TRAJ_VEL, PRED_TRAJ_VEL, OBS_TRAJ, PRED_TRAJ, FRAMES_IDS, REFERENCE_IMG, TRAINING_CKPT_DIR

# Gets a testing batch of trajectories starting at the same frame (for visualization)
def get_testing_batch(testing_data,testing_data_path):
    # A trajectory id
    randomtrajId     = np.random.randint(len(testing_data),size=1)[0]
    # Last observed frame id for a random trajectory in the testing dataset
    print(testing_data.Frame_Ids[randomtrajId])
    frame_id         = testing_data.Frame_Ids[randomtrajId][7]
    idx              = np.where((testing_data.Frame_Ids[:,7]==frame_id))[0]
    # Get the video corresponding to the testing
    cap   = cv2.VideoCapture(testing_data_path+'/video.avi')
    frame = 0
    while(cap.isOpened()):
        ret, test_bckgd = cap.read()
        if frame == frame_id:
            break
        frame = frame + 1
    # Form the batch
    return frame_id, traj_dataset(*(testing_data[idx])), test_bckgd

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
parser.add_argument('--num-ensembles',
                    type=int, default=10, metavar='N',
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
    experiment_parameters = Experiment_Parameters()

    dataset_dir   = "datasets/"
    dataset_names = ['eth-hotel','eth-univ','ucy-zara01','ucy-zara02','ucy-univ']
    model_name    = "deterministic_variances_ens"

    # Load the dataset and perform the split
    training_data, validation_data, testing_data, test_homography = setup_loo_experiment('ETH_UCY',dataset_dir,dataset_names,args.id_test,experiment_parameters,pickle_dir='pickle',use_pickled_data=args.pickle)

    # Torch dataset
    train_data = traj_dataset(training_data[OBS_TRAJ_VEL], training_data[PRED_TRAJ_VEL],training_data[OBS_TRAJ], training_data[PRED_TRAJ])
    val_data   = traj_dataset(validation_data[OBS_TRAJ_VEL], validation_data[PRED_TRAJ_VEL],validation_data[OBS_TRAJ], validation_data[PRED_TRAJ])
    test_data  = traj_dataset(testing_data[OBS_TRAJ_VEL], testing_data[PRED_TRAJ_VEL], testing_data[OBS_TRAJ], testing_data[PRED_TRAJ], testing_data[FRAMES_IDS])

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
            # Testing: Quantitative
            ade  = 0
            fde  = 0
            total= 0
            for batch_idx, (datavel_test, targetvel_test, data_test, target_test) in    enumerate(batched_test_data):
                if torch.cuda.is_available():
                    datavel_test  = datavel_test.to(device)
                total += len(datavel_test)
                # prediction
                init_pos  = np.expand_dims(data_test[:,-1,:],axis=1)
                pred_test = model.predict(datavel_test, dim_pred=12)[0] + init_pos
                ade    += np.average(np.sqrt(np.square(target_test-pred_test).sum(2)),axis=1).sum()
                fde    += (np.sqrt(np.square(target_test[:,-1,:]-pred_test[:,-1,:]).sum(1))).sum()
            logging.info("Test ade : {:.4f} ".format(ade/total))
            logging.info("Test fde : {:.4f} ".format(fde/total))

    # Instanciate the models
    models= []
    # For each element of the ensemble
    for ind in range(args.num_ensembles):
        model = lstm_encdec_gaussian(in_size=2, embedding_dim=128, hidden_dim=256, output_size=2)
        model.to(device)
        # Load the previously trained model
        model.load_state_dict(torch.load(TRAINING_CKPT_DIR+"/"+model_name+"_"+str(ind)+"_"+str(args.id_test)+".pth"))
        models.append(model)



    bck = plt.imread(os.path.join(dataset_dir,dataset_names[args.id_test], REFERENCE_IMG))


    #------------------ Obtenemos el batch unico de test para las curvas de calibracion ---------------------------
    datarel_test_full, targetrel_test_full, data_test_full, target_test_full, tpred_samples_full, sigmas_samples_full = generate_one_batch_test(batched_test_data, model, args.num_ensembles, TRAINING_CKPT_DIR, model_name, id_test=args.id_test, device=device)
    #---------------------------------------------------------------------------------------------------------------

    # Testing
    cont = 0
    for batch_idx, (datarel_test, targetrel_test, data_test, target_test) in enumerate(batched_test_data):

        pred_samples_cal   = []
        sigmas_samples_cal = []
        # For each model of the ensemble
        for ind in range(args.num_ensembles):
            # Load the model
            model.load_state_dict(torch.load(TRAINING_CKPT_DIR+"/"+model_name+"_"+str(ind)+"_"+str(args.id_test)+".pth"))
            model.eval()
            if torch.cuda.is_available():
                  datarel_test  = datarel_test.to(device)
            pred, sigmas = model.predict(datarel_test, dim_pred=12)
            # Keep moments
            pred_samples_cal.append(pred)
            sigmas_samples_cal.append(sigmas)
        # Stack the means and covariances
        pred_samples_cal   = np.array(pred_samples_cal)
        sigmas_samples_cal = np.array(sigmas_samples_cal)


        # ---------------------------------- Calibration HDR cap libro -------------------------------------------------
        print("**********************************************")
        print("***** Calibracion con Isotonic Regresion *****")
        print("**********************************************")

        # HDR y Calibracion
        if False:
            isotonic = calibrate_IsotonicReg(pred_samples_cal, data_test, target_test, sigmas_samples_cal, position=11, idTest=args.id_test, gaussian=True)
            pickle_out = open('isotonic.pickle',"wb")
            pickle.dump(isotonic, pickle_out)
        else:
            pickle_in = open('isotonic.pickle',"rb")
            isotonic = pickle.load(pickle_in)

        # Solo se ejecuta para un batch y es usado como dataset de calibración
        break

    frame_id, batch, test_bckgd = get_testing_batch(test_data,dataset_dir+dataset_names[args.id_test])
    # Form batches
    batched_test_data  = torch.utils.data.DataLoader(batch,batch_size=len(batch))
    n_trajs            = len(batch)
    # Get the homography
    homography_to_img = np.linalg.inv(test_homography)

    for batch_idx, (datarel_test, targetrel_test, data_test, target_test) in enumerate(batched_test_data):
        for traj_idx in range(len(datarel_test)):
            # Output for each element of the ensemble
            preds =[]
            sigmas=[]
            for idx in range(args.num_ensembles):
                if torch.cuda.is_available():
                      datarel_test  = datarel_test.to(device)
                pred, sigma = models[idx].predict(datarel_test, dim_pred=12)
                preds.append(pred[traj_idx]),sigmas.append(sigma[traj_idx])
            # Sampling from the mixture
            xs = []
            ys = []
            for i in range(1000):
                k      = np.random.randint(args.num_ensembles)
                mean   = preds[k][11]
                cov    = np.array([[sigmas[k][11,0],sigmas[k][11,2]],[sigmas[k][11,2],sigmas[k][11,1]]])
                sample = np.random.multivariate_normal(mean, cov, 1)[0]+ np.array([data_test[traj_idx,-1].numpy()])
                xs.append(sample[0,0])
                ys.append(sample[0,1])

            # xmin = min(min(xs),data_test[traj_idx,:,0].min().numpy())
            # xmax = max(max(xs),data_test[traj_idx,:,0].max().numpy())
            # ymin = min(min(ys),data_test[traj_idx,:,1].min().numpy())
            # ymax = max(max(ys),data_test[traj_idx,:,1].max().numpy())
            xmin = 0
            xmax = test_bckgd.shape[1]
            ymin = 0
            ymax = test_bckgd.shape[0]
            xx, yy = np.mgrid[xmin:xmax:100j,ymin:ymax:100j]

            # Testing/visualization uncalibrated KDE
            image_grid      = np.vstack([xx.ravel(), yy.ravel()])
            world_grid      = world_to_image_xy(np.transpose(image_grid),test_homography,flip=False)
            world_samples   = np.vstack([xs, ys])
            image_samples   = world_to_image_xy(np.transpose(world_samples),homography_to_img,flip=False)
            kernel          = st.gaussian_kde(world_samples)
            fs_samples      = kernel.evaluate(world_samples)
            sorted_samples  = sort_sample(fs_samples)
            observed_alphas = np.array([get_alpha(sorted_samples,fk) for fk in fs_samples ])
            # TODO: at some point define it in terms of alpha, not 1-alpha
            #fig, ax = plt.subplots(1,1,figsize=(12,12))
            #plt.plot(observed_alphas,modified_alphas,'+')
            #plt.show()

            # Visualization of the uncalibrated KDE
            alphas = np.linspace(1.0,0.0,num=5,endpoint=False)
            levels = []
            for alpha in alphas:
                level = get_falpha(sorted_samples,alpha)
                levels.append(level)
            f_unc        = np.reshape(kernel(np.transpose(world_grid)).T, xx.shape)
            norm_f_unc   = np.rot90(f_unc-np.min(f_unc))/(np.max(f_unc)-np.min(f_unc))
            transparency = np.sqrt(norm_f_unc)

            ## Or kernel density estimate plot instead of the contourf plot
            figs, axs = plt.subplots(1,2,figsize=(24,12),constrained_layout = True)
            axs[0].legend_ = None
            axs[0].imshow(test_bckgd)
            observations = world_to_image_xy(data_test[traj_idx,:,:], homography_to_img, flip=False)
            groundtruth  = world_to_image_xy(target_test[traj_idx,:,:], homography_to_img, flip=False)
            # Contour plot
            cset = axs[0].contour(xx, yy, f_unc, colors='k',levels=levels[1:],linewidths=0.5)
            cset.levels = np.array(alphas[1:])
            axs[0].clabel(cset, cset.levels,fontsize=8)
            axs[0].plot(observations[:,0],observations[:,1],color='blue')
            axs[0].plot([observations[-1,0],groundtruth[0,0]],[observations[-1,1],groundtruth[0,1]],color='red')
            axs[0].plot(groundtruth[:,0],groundtruth[:,1],color='red')
            #axs[0].plot(image_samples[:,0],image_samples[:,1],'+',color='green')
            # axs[0].plot(target_test[traj_idx,:,0],target_test[traj_idx,:,1],color='red')
            axs[0].set_xlim(xmin,xmax)
            axs[0].set_ylim(ymax,ymin)
            axs[0].axes.xaxis.set_visible(False)
            axs[0].axes.yaxis.set_visible(False)
            axs[0].imshow(transparency,alpha=transparency,cmap='viridis',extent=[xmin, xmax, ymin, ymax])
            # Testing/visualization **calibrated** KDE
            modified_alphas = isotonic.transform(observed_alphas)
            fs_samples_new  = []
            for alpha in modified_alphas:
                fs_samples_new.append(get_falpha(sorted_samples,alpha))
            fs_samples_new    = np.array(fs_samples_new)
            sorted_samples_new= sort_sample(fs_samples_new)
            importance_weights= fs_samples_new/fs_samples
            kernel            = st.gaussian_kde(world_samples,weights=importance_weights)
            f_cal             = np.reshape(kernel(np.transpose(world_grid)).T, xx.shape)
            norm_f_cal        = np.rot90(f_cal-np.min(f_cal))/(np.max(f_cal)-np.min(f_cal))
            transparency      = np.sqrt(norm_f_cal)
            # Visualization of the uncalibrated KDE
            alphas = np.linspace(1.0,0.0,num=5,endpoint=False)
            levels = []
            for alpha in alphas:
                level = get_falpha(sorted_samples_new,alpha)
                levels.append(level)
            cset = axs[1].contour(xx, yy, f_cal, colors='k',levels=levels[1:],linewidths=0.5)
            cset.levels = np.array(alphas[1:])
            axs[1].clabel(cset, cset.levels,fontsize=8)
            axs[1].plot(observations[:,0],observations[:,1],color='blue')
            axs[1].plot([observations[-1,0],groundtruth[0,0]],[observations[-1,1],groundtruth[0,1]],color='red')
            axs[1].plot(groundtruth[:,0],groundtruth[:,1],color='red')
            axs[1].imshow(test_bckgd)
            axs[1].set_xlim(xmin,xmax)
            axs[1].set_ylim(ymax,ymin)
            axs[1].axes.xaxis.set_visible(False)
            axs[1].axes.yaxis.set_visible(False)
            axs[1].imshow(norm_f_cal,alpha=transparency,cmap='viridis', extent=[xmin, xmax, ymin, ymax])
            plt.show()


if __name__ == "__main__":
    main()
