import os
import pickle
import numpy as np
import matplotlib.image as mpimg
import numpy as np
import cv2
import tensorflow as tf
import torch
from torch.utils.data import Dataset
from utils.constants import (
    FRAMES_IDS, KEY_IDX, OBS_NEIGHBORS, OBS_TRAJ, OBS_TRAJ_VEL, OBS_TRAJ_ACC, OBS_TRAJ_THETA, PRED_TRAJ, PRED_TRAJ_VEL, PRED_TRAJ_ACC,
    TRAIN_DATA_STR, TEST_DATA_STR, VAL_DATA_STR, MUN_POS_CSV
)
import logging

# Parameters
# The only datasets that can use add_kp are PETS2009-S2L1, TOWN-CENTRE
class Experiment_Parameters:
    def __init__(self):
        # Maximum number of persons in a frame
        self.person_max =70
        # Observation length (trajlet size)
        self.obs_len    = 8
        # Prediction length
        self.pred_len   = 12
        # Delimiter
        self.delim        = ','

# Creamos la clase para el dataset
class traj_dataset(Dataset):

    def __init__(self, Xrel_Train, Yrel_Train, X_Train, Y_Train, transform=None):
        self.Xrel_Train = Xrel_Train
        self.Yrel_Train = Yrel_Train
        self.X_Train = X_Train
        self.Y_Train = Y_Train
        self.transform = transform

    def __len__(self):
        return len(self.X_Train)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        xrel = self.Xrel_Train[idx]
        yrel = self.Yrel_Train[idx]
        x = self.X_Train[idx]
        y = self.Y_Train[idx]

        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
            xrel = self.transform(xrel)
            yrel = self.transform(yrel)

        return xrel, yrel, x, y

def get_testing_batch_synthec(testing_data,testing_data_path):
    # A trajectory id
    testing_data_arr = list(testing_data.as_numpy_iterator())
    randomtrajId     = np.random.randint(len(testing_data_arr),size=1)[0]
    frame_id         = testing_data_arr[randomtrajId][FRAMES_IDS][0]
    # Form the batch
    filtered_data  = testing_data.filter(lambda x: x[FRAMES_IDS][0]==frame_id)
    filtered_data  = filtered_data.batch(20)
    for element in filtered_data.as_numpy_iterator():
        return element

def prepare_data(datasets_path, datasets_names, parameters):
    datasets = range(len(datasets_names))
    datasets = list(datasets)

    # Paths for the datasets used to form the training set
    used_data_dirs = [datasets_names[x] for x in datasets]
    # Sequence lengths
    obs_len  = parameters.obs_len
    pred_len = parameters.pred_len
    seq_len  = obs_len + pred_len
    logging.info("Sequence length (observation+prediction): {}".format(seq_len))

    # Lists that will hold the data
    seq_pos_all                  = []
    seq_theta_all                = []
    seq_vel_all                  = []
    seq_acc_all                  = []
    seq_neighbors_all            = []
    seq_frames_all               = []  # [N, seq_len]

    # Scan all the datasets
    for idx,dataset_name in enumerate(datasets_names):
        seq_neighbors_dataset= []
        seq_pos_dataset      = []
        traj_data_path       = os.path.join(datasets_path+dataset_name, MUN_POS_CSV)
        logging.info("Reading "+traj_data_path)

        # Raw trajectory coordinates
        raw_traj_data = np.genfromtxt(traj_data_path, delimiter= parameters.delim)

        # We suppose that the frame ids are in ascending order
        frame_ids = np.unique(raw_traj_data[:, 0]).tolist()
        ped_ids   = np.unique(raw_traj_data[:, 1]).tolist()
        logging.info("Total number of frames: {}".format(len(frame_ids)))
        logging.info("Total number of agents: {}".format(len(np.unique(raw_traj_data[:, 1]).tolist())))

        raw_traj_data_per_frame = [] # people in frame
        raw_traj_data_per_ped   = {} # dictionnary of full trajectories, indexed by person id
        # Group the spatial pedestrian data frame by frame
        # List indexed by frame ids.
        # Data: id_frame, id_person, x, y
        for frame in frame_ids:
            raw_traj_data_per_frame.append(raw_traj_data[raw_traj_data[:, 0]==frame, :])
        for ped in ped_ids:
            t  = raw_traj_data[raw_traj_data[:, 1]==ped,0:1]
            px = raw_traj_data[raw_traj_data[:, 1]==ped,2]
            py = raw_traj_data[raw_traj_data[:, 1]==ped,3]
            if len(px)>1:
                # TODO: see where to read the dt info (0.4)
                vx = np.gradient(px,0.4)
                ax = np.gradient(vx,0.4)
                vx = np.expand_dims(vx,axis=1)
                ax = np.expand_dims(ax,axis=1)
                vy = np.gradient(py,0.4)
                ay = np.gradient(vy,0.4)
                vy = np.expand_dims(vy,axis=1)
                ay = np.expand_dims(ay,axis=1)
                pv = np.concatenate([t,np.expand_dims(px,axis=1),np.expand_dims(py,axis=1),vx,vy,ax,ay],axis=1)
                raw_traj_data_per_ped[ped]=pv
        counter = 0
        # Iterate over the frames
        for idx, frame in enumerate(frame_ids):
            if idx+seq_len>=len(frame_ids):
                break
            frame_max      = frame_ids[idx+seq_len]
            # Consider frame sequences of size seq_len = obs+pred
            # id_frame, id_person, x, y por every person present in the frame
            raw_seq_data   = raw_traj_data_per_frame[idx:idx+seq_len]
            raw_seq_data   = np.concatenate(raw_seq_data,axis=0)
            # Unique indices for the persons in the sequence "raw_seq_data"
            peds_in_seq    = list(np.unique(raw_seq_data[:,1]))
            # Number of pedestrians to consider
            num_peds_in_seq= len(peds_in_seq)

            # The following arrays have the same shape
            # "pos_seq_data" contains all the absolute positions of all the pedestrians in the sequence
            # and he information is encoded in an absolute frame (no transformation)
            pos_seq_data   = np.zeros((num_peds_in_seq, seq_len, 2), dtype="float32")
            # Same, with only the velocities and accelerations
            vel_seq_data   = np.zeros((num_peds_in_seq, seq_len, 2), dtype="float32")
            acc_seq_data   = np.zeros((num_peds_in_seq, seq_len, 2), dtype="float32")
            # Same with orientations
            theta_seq_data = np.zeros((num_peds_in_seq, seq_len, 1), dtype="float32")
            # Is the array that have the sequence of Id_person of all people that there are in frame sequence
            frame_ids_seq_data = np.zeros((num_peds_in_seq, seq_len), dtype="int32")

            # Tensor for holding nighbor data
            neighbors_data = np.zeros((num_peds_in_seq, seq_len, parameters.person_max, 3),dtype="float32")
            neighbors_data[:] = np.NaN

            ped_count = 0
            # For all the persons appearing in this sequence
            # We will make one entry in the sequences list
            for ped_id in peds_in_seq:
                # Get the information about ped_id, in the whole sequence
                ped_seq_data      = raw_seq_data[raw_seq_data[:,1]==ped_id,:]
                if (not (ped_id in raw_traj_data_per_ped.keys())):
                    continue
                ped_seq_data_full = raw_traj_data_per_ped[ped_id]
                ped_seq_data_mod  = ped_seq_data_full[(ped_seq_data_full[:,0]>=frame) & (ped_seq_data_full[:,0]<frame_max)]
                # We want pedestrians whose number of observations inside this sequence is exactly seq_len
                if len(ped_seq_data_mod) != seq_len:
                    # We do not have enough observations for this person
                    continue

                # To keep neighbors data for the person ped_id
                neighbors_ped_seq = np.zeros((seq_len, parameters.person_max, 3),dtype="float32")
                # Scan all the frames of the sequence
                for frame_idx,frame_id in enumerate(np.unique(raw_seq_data[:,0]).tolist()):
                    # Information of frame "frame_id"
                    frame_data = raw_seq_data[raw_seq_data[:,0]==frame_id,:]
                    # Id, x, y of the pedestrians of frame "frame_id"
                    frame_data = frame_data[:,1:4]
                    # For all the persons in the sequence
                    for neighbor_ped_idx,neighbor_ped_id in enumerate(peds_in_seq):
                        # Get the data of this specific person
                        neighbor_data = frame_data[frame_data[:, 0]==neighbor_ped_id,:]
                        # If we have information for this pedestrian, add it to the neighbors struture
                        if neighbor_data.size != 0:
                            neighbors_ped_seq[frame_idx,neighbor_ped_idx,:] = neighbor_data
                # Contains the neighbor data for ped_count
                neighbors_data[ped_count,:,:,:] = neighbors_ped_seq

                # Absolute x,y and velocities for all person_id
                pos_seq_data[ped_count, :, :] = ped_seq_data_mod[:,1:3]
                vel_seq_data[ped_count, :, :] = ped_seq_data_mod[:,3:5]
                acc_seq_data[ped_count, :, :] = ped_seq_data_mod[:,5:7]
                # Orientations
                # TODO
                # theta_seq_data[ped_count,1:, 0] = np.arctan2(ped_seq_pos[1:, 1] - ped_seq_pos[:-1, 1],ped_seq_pos[1:, 0] - ped_seq_pos[:-1, 0])
                # theta_seq_data[ped_count,0,  0] = theta_seq_data[ped_count,1,  0]

                # For each tracked person
                # we keep the list of all the frames in which it is present
                frame_ids_seq_data[ped_count, :] = ped_seq_data_mod[:,0]
                # Increment ped_count (persons )
                ped_count += 1

            # Only count_ped data are preserved in the following three arrays
            # Append all the trajectories (pos_seq_data) starting at this frame
            seq_pos_all.append(pos_seq_data[:ped_count])
            # Append all the displacement trajectories (pos_seq_data) starting at this frame
            seq_vel_all.append(vel_seq_data[:ped_count])
            seq_acc_all.append(acc_seq_data[:ped_count])
            seq_theta_all.append(theta_seq_data[:ped_count])
            # Append all the frame ranges (frame_ids_seq_data) starting at this frame
            seq_frames_all.append(frame_ids_seq_data[:ped_count])
            # Information used locally for this dataset
            seq_pos_dataset.append(pos_seq_data[:ped_count])
            # Neighbours
            seq_neighbors_all.append(neighbors_data[:ped_count])
            # Append all the neighbor data (neighbors_data) starting at this frame
            seq_neighbors_dataset.append(neighbors_data[:ped_count])

        # Neighbors information
        seq_neighbors_dataset = np.concatenate(seq_neighbors_dataset, axis = 0)
        obs_neighbors         = seq_neighbors_dataset[:,:obs_len,:,:]
        seq_pos_dataset       = np.concatenate(seq_pos_dataset,axis=0)
        logging.info("Total number of trajlets in this dataset: {}".format(seq_pos_dataset.shape[0]))

    # Upper level (all datasets)
    # Concatenate all the content of the lists (pos/relative pos/frame ranges)
    seq_pos_all   = np.concatenate(seq_pos_all, axis=0)
    seq_vel_all   = np.concatenate(seq_vel_all, axis=0)
    seq_acc_all   = np.concatenate(seq_acc_all, axis=0)
    seq_theta_all = np.concatenate(seq_theta_all, axis=0)
    seq_frames_all= np.concatenate(seq_frames_all, axis=0)
    seq_neighbors_all = np.concatenate(seq_neighbors_all, axis=0)
    logging.info("Total number of sample sequences: {}".format(len(seq_pos_all)))

    # We split into the obs traj and pred_traj
    # [total, obs_len, 2] and  [total, pred_len, 2]
    obs_traj      = seq_pos_all[:, :obs_len, :]
    obs_traj_theta= seq_theta_all[:, :obs_len, :]
    pred_traj     = seq_pos_all[:, obs_len:, :]
    frame_obs     = seq_frames_all[:, :obs_len]
    obs_traj_vel  = seq_vel_all[:, :obs_len, :]
    pred_traj_vel = seq_vel_all[:, obs_len:, :]
    obs_traj_acc  = seq_acc_all[:, :obs_len, :]
    pred_traj_acc = seq_acc_all[:, obs_len:, :]
    neighbors_obs= seq_neighbors_all[:, :obs_len, :]
    # Save all these data as a dictionary
    data = {
        OBS_TRAJ:       obs_traj,
        OBS_TRAJ_VEL:   obs_traj_vel,
        OBS_TRAJ_ACC:   obs_traj_acc,
        OBS_TRAJ_THETA: obs_traj_theta,
        PRED_TRAJ:      pred_traj,
        PRED_TRAJ_VEL:  pred_traj_vel,
        FRAMES_IDS:     frame_obs,
        OBS_NEIGHBORS:  neighbors_obs
    }
    return data

def setup_loo_experiment(experiment_name,ds_path,ds_names,leave_id,experiment_parameters,use_pickled_data=False,pickle_dir='pickle/',validation_proportion=0.1):
    # Dataset to be tested
    testing_datasets_names  = [ds_names[leave_id]]
    training_datasets_names = ds_names[:leave_id]+ds_names[leave_id+1:]
    logging.info('Testing/validation dataset: {}'.format(testing_datasets_names))
    logging.info('Training datasets: {}'.format(training_datasets_names))
    if not use_pickled_data:
        # Process data specified by the path to get the trajectories with
        logging.info('Extracting data from the datasets')
        test_data  = prepare_data(ds_path, testing_datasets_names, experiment_parameters)
        train_data = prepare_data(ds_path, training_datasets_names, experiment_parameters)

        # Count how many data we have (sub-sequences of length 8, in pred_traj)
        n_test_data  = len(test_data[list(test_data.keys())[2]])
        n_train_data = len(train_data[list(train_data.keys())[2]])
        idx          = np.random.permutation(n_train_data)
        # TODO: validation should be done from a similar distribution as test set!
        validation_pc= validation_proportion
        validation   = int(n_train_data*validation_pc)
        training     = int(n_train_data-validation)

        # Indices for training
        idx_train = idx[0:training]
        #  Indices for validation
        idx_val   = idx[training:]
        # Training set
        training_data = {
            OBS_TRAJ:       train_data[OBS_TRAJ][idx_train],
            OBS_TRAJ_REL:   train_data[OBS_TRAJ_REL][idx_train],
            OBS_TRAJ_THETA: train_data[OBS_TRAJ_THETA][idx_train],
            PRED_TRAJ:      train_data[PRED_TRAJ][idx_train],
            PRED_TRAJ_REL:  train_data[PRED_TRAJ_REL][idx_train],
            FRAMES_IDS:     train_data[FRAMES_IDS][idx_train],
        }
        # Test set
        testing_data = {
            OBS_TRAJ:       test_data[OBS_TRAJ][:],
            OBS_TRAJ_REL:   test_data[OBS_TRAJ_REL][:],
            OBS_TRAJ_THETA: test_data[OBS_TRAJ_THETA][:],
            PRED_TRAJ:      test_data[PRED_TRAJ][:],
            PRED_TRAJ_REL:  test_data[PRED_TRAJ_REL][:],
            FRAMES_IDS:     test_data[FRAMES_IDS][:],
        }
        # Validation set
        validation_data ={
            OBS_TRAJ:       train_data[OBS_TRAJ][idx_val],
            OBS_TRAJ_REL:   train_data[OBS_TRAJ_REL][idx_val],
            OBS_TRAJ_THETA: train_data[OBS_TRAJ_THETA][idx_val],
            PRED_TRAJ:      train_data[PRED_TRAJ][idx_val],
            PRED_TRAJ_REL:  train_data[PRED_TRAJ_REL][idx_val],
            FRAMES_IDS:     train_data[FRAMES_IDS][idx_val],
        }
        # Training dataset
        pickle_out = open(pickle_dir+TRAIN_DATA_STR+experiment_name+'.pickle',"wb")
        pickle.dump(training_data, pickle_out, protocol=2)
        pickle_out.close()

        # Test dataset
        pickle_out = open(pickle_dir+TEST_DATA_STR+experiment_name+'.pickle',"wb")
        pickle.dump(test_data, pickle_out, protocol=2)
        pickle_out.close()

        # Validation dataset
        pickle_out = open(pickle_dir+VAL_DATA_STR+experiment_name+'.pickle',"wb")
        pickle.dump(validation_data, pickle_out, protocol=2)
        pickle_out.close()
    else:
        # Unpickle the ready-to-use datasets
        logging.info("Unpickling...")
        pickle_in = open(pickle_dir+TRAIN_DATA_STR+experiment_name+'.pickle',"rb")
        training_data = pickle.load(pickle_in)
        pickle_in = open(pickle_dir+TEST_DATA_STR+experiment_name+'.pickle',"rb")
        test_data = pickle.load(pickle_in)
        pickle_in = open(pickle_dir+VAL_DATA_STR+experiment_name+'.pickle',"rb")
        validation_data = pickle.load(pickle_in)

    logging.info("Training data: "+ str(len(training_data[list(training_data.keys())[0]])))
    logging.info("Test data: "+ str(len(test_data[list(test_data.keys())[0]])))
    logging.info("Validation data: "+ str(len(validation_data[list(validation_data.keys())[0]])))

    # Load the homography corresponding to this dataset
    homography_file = os.path.join(ds_path+testing_datasets_names[0]+'/H.txt')
    test_homography = np.genfromtxt(homography_file)
    return training_data,validation_data,test_data,test_homography
