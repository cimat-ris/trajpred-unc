import os
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from trajpred_unc.utils.constants import (
	FRAMES_IDS, KEY_IDX, OBS_NEIGHBORS, OBS_TRAJ, OBS_TRAJ_VEL, OBS_TRAJ_ACC, OBS_TRAJ_THETA, PRED_TRAJ, PRED_TRAJ_VEL, PRED_TRAJ_ACC,REFERENCE_IMG,PED_IDS,
	TRAIN_DATA_STR, TEST_DATA_STR, VAL_DATA_STR, MUN_POS_CSV, DATASETS_DIR, SUBDATASETS_NAMES
)
import logging

"""
Trajectory dataset
"""
class traj_dataset(Dataset):

	def __init__(self, Xrel_Train, Yrel_Train, X_Train, Y_Train, Neighbors=None,Frame_Ids=None,Ped_Ids=None):
		self.Xrel_Train= Xrel_Train
		self.Yrel_Train= Yrel_Train
		self.X_Train   = X_Train
		self.Y_Train   = Y_Train
		self.Frame_Ids = Frame_Ids
		self.neighbors = Neighbors

	def __len__(self):
		return len(self.X_Train)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		# Access to elements
		xrel = self.Xrel_Train[idx]
		yrel = self.Yrel_Train[idx]
		x    = self.X_Train[idx]
		y    = self.Y_Train[idx]
		n    = None
		if self.neighbors is not None:
			n    = np.asarray(self.neighbors[idx])
		return xrel, yrel, x, y, n

#  Trajectory dataset
class traj_dataset_bitrap(Dataset):

	def __init__(self, X_global, Neighbors, Y_global, Frame_Ids=None):
		self.X_global  = X_global
		self.Y_global  = Y_global
		self.neighbors = Neighbors
		self.Frame_Ids = Frame_Ids

	def __len__(self):
		return len(self.X_global)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		# Access to elements
		x = self.X_global[idx]
		y = self.Y_global[idx]
		if isinstance(idx, int):
			n = self.neighbors[idx]
		else:
			n = [self.neighbors[i] for i in idx]
		return x, n, y

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

def collate_fn_padd(batch):
	'''
	Padds batch of variable length
	'''
	xrels, yrels, xs, ys, ns = zip(*batch)
	mask    = None
	lengths = None
    ## Get sequence lengths
	if ns[0] is not None:
		lengths = torch.tensor([n.shape[0] for n in ns ])
		## Padding
		ns      = [ torch.Tensor(n) for n in ns]
		ns      = torch.nn.utils.rnn.pad_sequence(ns,batch_first=True)
		# Mask: true when there is a neighbor
		mask    = (ns != 0)
		mask    = mask[:,:,:,0]
	else:
		ns      = None
	return torch.Tensor(np.asarray(xrels)),torch.Tensor(np.asarray(yrels)),torch.Tensor(np.asarray(xs)),torch.Tensor(np.asarray(ys)),ns,lengths,mask

def get_raw_data(datasets_path, dataset_name, delim):
	"""
	Open dataset file and returns the raw array
	Args:
		- datasets_path: the path to the datasets directory
		- dataset_name: the name of the sub-dataset to read
		- delim: delimiter
	Returns:
		- the numpy array with all raw data
	"""
	if 'sdd' in datasets_path:
		traj_data_path = os.path.join(datasets_path, dataset_name)
		logging.info("Reading "+traj_data_path+'.pickle')
		 # Unpickle raw datasets
		logging.info("Unpickling raw data...")
		pickle_in = open(traj_data_path+'.pickle',"rb")
		raw_traj_data = pickle.load(pickle_in)
		raw_traj_data = raw_traj_data.to_numpy()
	else:
		traj_data_path       = os.path.join(datasets_path+dataset_name, MUN_POS_CSV)
		logging.info("Reading "+traj_data_path)
		# Raw trajectory coordinates
		raw_traj_data = np.genfromtxt(traj_data_path, delimiter= delim)
	return raw_traj_data

def prepare_data(datasets_path, datasets_names, config):
	"""
	Open dataset file and returns the raw array
	Args:
		- datasets_path: the path to the datasets directory
		- dataset_names: the names of the sub-datasets to read
		- config: configuration dictionary
	Returns:
		- dictionary with useful data for HTP
	"""
	datasets = range(len(datasets_names))
	datasets = list(datasets)

	# Sequence lengths
	obs_len  = config["obs_len"]
	pred_len = config["pred_len"]
	seq_len  = obs_len + pred_len
	logging.info("Sequence length (observation+prediction): {}".format(seq_len))

	# Lists that will hold the data
	seq_pos_all                  = []
	seq_theta_all                = []
	seq_vel_all                  = []
	seq_acc_all                  = []
	seq_neighbors_all            = []
	seq_frames_all               = []  # [N, seq_len]
	seq_ped_ids_all              = []

	# Scan all the datasets
	for idx,dataset_name in enumerate(datasets_names):
		# Raw trajectory coordinates
		raw_traj_data = get_raw_data(datasets_path, dataset_name, config["delim"])

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
		for frame in tqdm(frame_ids,desc="Gathering data per frame"):
			raw_traj_data_per_frame.append(raw_traj_data[raw_traj_data[:, 0]==frame])
		for ped in tqdm(ped_ids,desc="Gathering data per pedestrian"):
			t  = raw_traj_data[raw_traj_data[:, 1]==ped,0:1]
			id = raw_traj_data[raw_traj_data[:, 1]==ped,1:2]
			px = raw_traj_data[raw_traj_data[:, 1]==ped,2]
			py = raw_traj_data[raw_traj_data[:, 1]==ped,3]
			if len(px)>1:
				# FIXME: same bug here as in Trajectron++, because of np.gradient. Needs to be corrected!
				# TODO: see where to read the dt info (0.4)
				vx = np.ediff1d(px, to_begin=(px[1] - px[0])) / config["dt"]
				ax = np.ediff1d(vx, to_begin=(vx[1] - vx[0])) / config["dt"]
				vx = np.expand_dims(vx,axis=1)
				ax = np.expand_dims(ax,axis=1)
				vy = np.ediff1d(px, to_begin=(py[1] - py[0])) / config["dt"]
				ay = np.ediff1d(vy, to_begin=(vy[1] - vy[0])) / config["dt"]
				vy = np.expand_dims(vy,axis=1)
				ay = np.expand_dims(ay,axis=1)
				pv = np.concatenate([t,np.expand_dims(px,axis=1),np.expand_dims(py,axis=1),vx,vy,ax,ay,id],axis=1)
				raw_traj_data_per_ped[ped]=pv
		counter    = 0
		last_frame = -config["max_overlap"]
		# Iterate over the frames to define sequences
		for idx, frame in enumerate(tqdm(frame_ids, desc='Defining sequences for scene '+dataset_name)):
			# We will not have enough frames
			if idx+seq_len>=len(frame_ids):
				break
			if frame<last_frame+config["max_overlap"]:
				continue
			last_frame     = frame
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
			# and the information is encoded in an absolute frame (no transformation)
			pos_seq_data   = np.zeros((num_peds_in_seq, seq_len, 2), dtype="float32")
			# Same, with only the velocities and accelerations
			vel_seq_data   = np.zeros((num_peds_in_seq, seq_len, 2), dtype="float32")
			acc_seq_data   = np.zeros((num_peds_in_seq, seq_len, 2), dtype="float32")
			# Same with orientations
			theta_seq_data = np.zeros((num_peds_in_seq, seq_len, 1), dtype="float32")
			# Is the array that have the sequence of Id_person of all people that there are in frame sequence
			frame_ids_seq_data = np.zeros((num_peds_in_seq, seq_len), dtype="int32")
			ped_ids_seq_data   = np.zeros((num_peds_in_seq, 1), dtype="int32")
			ped_count = 0
			# Tensor for holding neighbor data, for each sequence. List of 8x6 tensors
			neighbors_data = []
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

				if config["use_neighbors"]:
					# To keep neighbors data for the person ped_id
					neighbors_ped_seq = []
					# For all the persons in the sequence
					for neighbor_ped_id in peds_in_seq:
						# Get
						if (not (neighbor_ped_id in raw_traj_data_per_ped.keys())):
							continue
						if (neighbor_ped_id==ped_id):
							continue
						neighbor_seq_data_full = raw_traj_data_per_ped[neighbor_ped_id]
						neighbor_seq_data_mod  = neighbor_seq_data_full[(neighbor_seq_data_full[:,0]>=frame) & (neighbor_seq_data_full[:,0]<frame_max)]
						neighbor_data          = np.zeros((obs_len, 6), dtype="float32")
						for (n_idx,frame_id) in enumerate(np.unique(neighbor_seq_data_mod[:,0]).tolist()):
							temp = np.where(ped_seq_data_mod[:,0]==frame_id)
							if temp[0].size != 0:
								idx             = temp[0][0]
								if idx<obs_len:
									neighbor_data[idx,:] = neighbor_seq_data_mod[n_idx,1:7]
									# Position is taken as relative to the considered agent
									neighbor_data[idx,0] -= ped_seq_data_mod[idx,1]
									neighbor_data[idx,1] -= ped_seq_data_mod[idx,2]
						neighbors_ped_seq.append(neighbor_data)
					# Contains the neighbor data per sequence
					neighbors_data.append(neighbors_ped_seq)

				# Absolute x,y and velocities for all person_id
				pos_seq_data[ped_count, :, :] = ped_seq_data_mod[:,1:3]
				vel_seq_data[ped_count, :, :] = ped_seq_data_mod[:,3:5]
				acc_seq_data[ped_count, :, :] = ped_seq_data_mod[:,5:7]

				# For each tracked person
				# we keep the list of all the frames in which it is present
				frame_ids_seq_data[ped_count, :] = ped_seq_data_mod[:,0]
				ped_ids_seq_data[ped_count, 0]   = ped_seq_data_mod[0,7:8]
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
			seq_ped_ids_all.append(ped_ids_seq_data[:ped_count])
			# Sets of neighbours for each sequence
			seq_neighbors_all.extend(neighbors_data)
	# Upper level (all datasets)
	# Concatenate all the content of the lists (pos/relative pos/frame ranges)
	seq_pos_all      = np.concatenate(seq_pos_all, axis=0)
	seq_vel_all      = np.concatenate(seq_vel_all, axis=0)
	seq_acc_all      = np.concatenate(seq_acc_all, axis=0)
	seq_theta_all    = np.concatenate(seq_theta_all, axis=0)
	seq_frames_all   = np.concatenate(seq_frames_all, axis=0)
	seq_ped_ids_all  = np.concatenate(seq_ped_ids_all, axis=0)
	#seq_neighbors_all= np.concatenate(seq_neighbors_all, axis=0)
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
	neighbors_obs = seq_neighbors_all
	# Save all these data as a dictionary
	data = {
		OBS_TRAJ:       obs_traj,
		OBS_TRAJ_VEL:   obs_traj_vel,
		OBS_TRAJ_ACC:   obs_traj_acc,
		OBS_TRAJ_THETA: obs_traj_theta,
		PRED_TRAJ:      pred_traj,
		PRED_TRAJ_VEL:  pred_traj_vel,
		FRAMES_IDS:     frame_obs,
		OBS_NEIGHBORS:  neighbors_obs,
		PED_IDS:        seq_ped_ids_all
	}
	return data

def setup_loo_experiment(config):
	ds_path  = DATASETS_DIR[config["id_dataset"]]
	ds_names = SUBDATASETS_NAMES[config["id_dataset"]]
	# Experiment name is set to the name of the test dataset
	leave_id  			    = config["id_test"]	
	experiment_name         = ds_names[leave_id]
	# Dataset to be tested
	testing_datasets_names  = [ds_names[leave_id]]
	training_datasets_names = ds_names[:leave_id]+ds_names[leave_id+1:]
	logging.info('Testing/validation dataset: {}'.format(testing_datasets_names))
	logging.info('Training datasets: {}'.format(training_datasets_names))
	if not config["pickle"]:
		# Process data specified by the path to get the trajectories with
		logging.info('Extracting data from the datasets')
		test_data  = prepare_data(ds_path, testing_datasets_names, config)
		train_data = prepare_data(ds_path, training_datasets_names, config)

		# Count how many data we have (sub-sequences of length 8, in pred_traj)
		n_test_data  = len(test_data[list(test_data.keys())[2]])
		n_train_data = len(train_data[list(train_data.keys())[2]])
		idx          = np.random.permutation(n_train_data)
		# TODO: validation should be done from a similar distribution as test set!
		validation_pc= config["validation_proportion"]
		validation   = int(n_train_data*validation_pc)
		training     = int(n_train_data-validation)
		# Indices for training
		idx_train = idx[0:training]
		#  Indices for validation
		idx_val   = idx[training:]
		# Training set
		training_data = {
			OBS_TRAJ:       train_data[OBS_TRAJ][idx_train],
			OBS_TRAJ_VEL:   train_data[OBS_TRAJ_VEL][idx_train],
			OBS_TRAJ_ACC:   train_data[OBS_TRAJ_ACC][idx_train],
			OBS_TRAJ_THETA: train_data[OBS_TRAJ_THETA][idx_train],
			PRED_TRAJ:      train_data[PRED_TRAJ][idx_train],
			PRED_TRAJ_VEL:  train_data[PRED_TRAJ_VEL][idx_train],
			FRAMES_IDS:     train_data[FRAMES_IDS][idx_train],
			PED_IDS:        train_data[PED_IDS][idx_train],
		}
		# Test set
		testing_data = {
			OBS_TRAJ:       test_data[OBS_TRAJ][:],
			OBS_TRAJ_VEL:   test_data[OBS_TRAJ_VEL][:],
			OBS_TRAJ_ACC:   test_data[OBS_TRAJ_ACC][:],
			OBS_TRAJ_THETA: test_data[OBS_TRAJ_THETA][:],
			PRED_TRAJ:      test_data[PRED_TRAJ][:],
			PRED_TRAJ_VEL:  test_data[PRED_TRAJ_VEL][:],
			FRAMES_IDS:     test_data[FRAMES_IDS][:],
			PED_IDS:        test_data[PED_IDS][:],
		}
		# Validation set
		validation_data ={
			OBS_TRAJ:       train_data[OBS_TRAJ][idx_val],
			OBS_TRAJ_VEL:   train_data[OBS_TRAJ_VEL][idx_val],
			OBS_TRAJ_ACC:   train_data[OBS_TRAJ_ACC][:],
			OBS_TRAJ_THETA: train_data[OBS_TRAJ_THETA][idx_val],
			PRED_TRAJ:      train_data[PRED_TRAJ][idx_val],
			PRED_TRAJ_VEL:  train_data[PRED_TRAJ_VEL][idx_val],
			FRAMES_IDS:     train_data[FRAMES_IDS][idx_val],
			PED_IDS:        train_data[PED_IDS][idx_val],
		}
		if config["use_neighbors"]:
			training_data[OBS_NEIGHBORS]   = [train_data[OBS_NEIGHBORS][i] for i in idx_train]
			testing_data[OBS_NEIGHBORS]    = test_data[OBS_NEIGHBORS][:]
			validation_data[OBS_NEIGHBORS] = [train_data[OBS_NEIGHBORS][i] for i in idx_val]

		if not os.path.exists(config["pickle_dir"]):
			# Create a new directory if it does not exist
			os.makedirs(config["pickle_dir"])

		# Training dataset
		pickle_out = open(config["pickle_dir"]+TRAIN_DATA_STR+experiment_name+'.pickle',"wb")
		pickle.dump(training_data, pickle_out, protocol=2)
		pickle_out.close()

		# Test dataset
		pickle_out = open(config["pickle_dir"]+TEST_DATA_STR+experiment_name+'.pickle',"wb")
		pickle.dump(test_data, pickle_out, protocol=2)
		pickle_out.close()

		# Validation dataset
		pickle_out = open(config["pickle_dir"]+VAL_DATA_STR+experiment_name+'.pickle',"wb")
		pickle.dump(validation_data, pickle_out, protocol=2)
		pickle_out.close()
	else:
		# Unpickle the ready-to-use datasets
		logging.info("Unpickling...")
		pickle_in = open(config["pickle_dir"]+TRAIN_DATA_STR+experiment_name+'.pickle',"rb")
		training_data = pickle.load(pickle_in)
		pickle_in = open(config["pickle_dir"]+TEST_DATA_STR+experiment_name+'.pickle',"rb")
		test_data = pickle.load(pickle_in)
		pickle_in = open(config["pickle_dir"]+VAL_DATA_STR+experiment_name+'.pickle',"rb")
		validation_data = pickle.load(pickle_in)

	logging.info("Training data: "+ str(len(training_data[list(training_data.keys())[0]])))
	logging.info("Test data: "+ str(len(test_data[list(test_data.keys())[0]])))
	logging.info("Validation data: "+ str(len(validation_data[list(validation_data.keys())[0]])))
	if 'sdd' in ds_path:
		test_homography = {}
	else:
		# Load the homography corresponding to this dataset
		homography_file = os.path.join(ds_path+testing_datasets_names[0]+'/H.txt')
		test_homography = np.genfromtxt(homography_file)
	return training_data,validation_data,test_data,test_homography

def get_dataset(config):
	# Load the dataset and perform the split
	training_data, validation_data, test_data, homography = setup_loo_experiment(config)
	# Load the reference image
	if not 'sdd' in DATASETS_DIR[config["id_dataset"]]:
		reference_image = plt.imread(os.path.join(DATASETS_DIR[config["id_dataset"]],SUBDATASETS_NAMES[config["id_dataset"]][config["id_test"]],REFERENCE_IMG))
	else:
		reference_image = None
	# Torch dataset
	if config["use_neighbors"]==True:
		train_data= traj_dataset(training_data[OBS_TRAJ_VEL ], training_data[PRED_TRAJ_VEL],training_data[OBS_TRAJ], training_data[PRED_TRAJ], Frame_Ids=training_data[FRAMES_IDS], Ped_Ids=training_data[PED_IDS],Neighbors=training_data[OBS_NEIGHBORS])
		val_data  = traj_dataset(validation_data[OBS_TRAJ_VEL ],validation_data[PRED_TRAJ_VEL],validation_data[OBS_TRAJ], validation_data[PRED_TRAJ], Frame_Ids=validation_data[FRAMES_IDS], Ped_Ids=validation_data[PED_IDS], Neighbors=validation_data[OBS_NEIGHBORS])
		test_data = traj_dataset(test_data[OBS_TRAJ_VEL ], test_data[PRED_TRAJ_VEL], test_data[OBS_TRAJ], test_data[PRED_TRAJ],Frame_Ids=test_data[FRAMES_IDS], Ped_Ids=test_data[PED_IDS],Neighbors=test_data[OBS_NEIGHBORS])
	else:
		train_data= traj_dataset(training_data[OBS_TRAJ_VEL ], training_data[PRED_TRAJ_VEL],training_data[OBS_TRAJ], training_data[PRED_TRAJ], Frame_Ids=training_data[FRAMES_IDS], Ped_Ids=training_data[PED_IDS])
		val_data  = traj_dataset(validation_data[OBS_TRAJ_VEL ],validation_data[PRED_TRAJ_VEL],validation_data[OBS_TRAJ], validation_data[PRED_TRAJ], Frame_Ids=validation_data[FRAMES_IDS], Ped_Ids=validation_data[PED_IDS])
		test_data = traj_dataset(test_data[OBS_TRAJ_VEL ], test_data[PRED_TRAJ_VEL], test_data[OBS_TRAJ], test_data[PRED_TRAJ],Frame_Ids=test_data[FRAMES_IDS], Ped_Ids=test_data[PED_IDS])

	# Form batches
	batched_train_data = torch.utils.data.DataLoader(train_data,batch_size=config["batch_size"],shuffle=True,collate_fn=collate_fn_padd)
	batched_val_data   = torch.utils.data.DataLoader(val_data,batch_size=config["batch_size"],shuffle=True,collate_fn=collate_fn_padd)
	batched_test_data  = torch.utils.data.DataLoader(test_data,batch_size=config["batch_size"],shuffle=True,collate_fn=collate_fn_padd)
	return batched_train_data,batched_val_data,batched_test_data,homography,reference_image

# Gets a testing batch of trajectories starting at the same frame (for visualization)
def get_testing_batch(test_data,config):
	# A trajectory id
    randomtrajId     = np.random.randint(len(test_data),size=1)[0]
    # Last observed frame id for a random trajectory in the testing dataset
    frame_id         = test_data.Frame_Ids[randomtrajId][7]
    idx              = np.where((test_data.Frame_Ids[:,7]==frame_id))[0]
    ds_path          = DATASETS_DIR[config["id_dataset"]]
    ds_names         = SUBDATASETS_NAMES[config["id_dataset"]][config["id_test"]]     
	# Get the video corresponding to the testing
    cap              = cv2.VideoCapture(ds_path+ds_names+'/video.avi')
    frame = 0
    while(cap.isOpened()):
        __, test_bckgd = cap.read()
        if frame == frame_id:
            break
        frame = frame + 1
    # Form the batch
    return frame_id, traj_dataset(*(test_data[idx])), test_bckgd

# Gets a testing batch of trajectories starting at the same frame (for visualization)
def get_testing_batch_bitrap(testing_data,testing_data_path):
	# A trajectory id
	randomtrajId     = np.random.randint(len(testing_data),size=1)[0]
	# Last observed frame id for a random trajectory in the testing dataset
	frame_id         = testing_data.Frame_Ids[randomtrajId][7]
	logging.info("Frame ID: {}".format(frame_id))
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
	return frame_id, traj_dataset_bitrap(*(testing_data[idx])), test_bckgd
