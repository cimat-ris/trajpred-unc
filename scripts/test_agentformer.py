import numpy as np
import argparse
import os, logging
import sys
import subprocess
import shutil
import matplotlib.pyplot as plt
sys.path.append('../AgentFormer')
from data.dataloader import data_generator
from lib.torch import *
from lib.config import Config
from model.model_lib import model_dict
from lib.utils import prepare_seed, print_log, mkdir_if_missing
sys.path.append('.')
from utils.datasets_utils import get_dataset
from utils.calibration_utils import save_data_for_calibration
from utils.config import get_config
from utils.constants import SUBDATASETS_NAMES,AGENTFORMER

configurations = ['hotel_agentformer_pre','eth_agentformer_pre','zara1_agentformer_pre','zara2_agentformer_pre','univ_agentformer_pre']
def get_model_prediction(data, sample_k):
	model.set_data(data)
	# Past data: data['pre_motion_3D'] is a list of n 8x2 trajectories
	# Ground truth: data['fut_motion_3D'] is a list of n 12x2 trajectories
	recon_motion_3D, _     = model.inference(mode='recon', sample_num=sample_k)
	sample_motion_3D, data = model.inference(mode='infer', sample_num=sample_k, need_weights=False)
	# Output future samples: sample_motion_3D: sample_kxnx12x2
	sample_motion_3D = sample_motion_3D.transpose(0, 1).contiguous()
	return recon_motion_3D, sample_motion_3D

def get_trajectories(predictions, raw_data):
	pred_num = 0
	fut_data, seq_name, frame, valid_id, pred_mask = raw_data['fut_data'], raw_data['seq'], raw_data['frame'], raw_data['valid_id'], raw_data['pred_mask']
	pred_samples    = []
	for s in range(predictions.shape[0]): # Cycle over the samples
		pred_arr = []
		for i in range(len(valid_id)):    # Cycle over the agents
			identity = valid_id[i]
			if pred_mask is not None and pred_mask[i] != 1.0:
				continue

			"""Cycle over future frames"""
			trajectory = []
			for j in range(cfg.future_frames):
				cur_data = fut_data[j]
				if len(cur_data) > 0 and identity in cur_data[:, 1]:
					frame_data = cur_data[cur_data[:, 1] == identity].squeeze()
				else:
					frame_data = most_recent_data.copy()
					frame_data[0] = frame + j + 1
				frame_data[[13, 15]] = predictions[s, i, j].cpu().numpy()   # [13, 15] corresponds to 2D pos
				most_recent_data = frame_data.copy()
				trajectory.append(frame_data)
			trajectory = np.array(trajectory)
			pred_arr.append(trajectory)
		if len(pred_arr) > 0:
			pred_arr = np.array(pred_arr)
			indices = [13, 15]            # x, z (remove y which is the height)
			pred_arr = pred_arr[:,:,indices]
			pred_samples.append(pred_arr)
	return np.array(pred_samples)

def apply_model(generator, cfg):
	total_num_pred = 0
	all_predictions= []
	all_gt         = []
	all_obs        = []
	while not generator.is_epoch_end():
		data = generator()
		if data is None:
			continue
		seq_name, frame = data['seq'], data['frame']
		frame = int(frame)
		sys.stdout.write('testing seq: %s, frame: %06d                \r' % (seq_name, frame))
		sys.stdout.flush()
		all_obs.append(torch.stack(data['pre_motion_3D'], dim=0) * cfg.traj_scale)
		all_gt.append(torch.stack(data['fut_motion_3D'], dim=0) * cfg.traj_scale)
		with torch.no_grad():
			recon_motion_3D, sample_motion_3D = get_model_prediction(data, cfg.sample_k)
		recon_motion_3D, sample_motion_3D = recon_motion_3D * cfg.traj_scale, sample_motion_3D * cfg.traj_scale

		# Transform the output into our format for trajectories
		all_predictions.append(get_trajectories(sample_motion_3D,data))

	all_predictions = np.concatenate(all_predictions,axis=1)
	all_gt          = torch.concatenate(all_gt,axis=0)
	all_obs         = torch.concatenate(all_obs,axis=0)
	total_num_pred  = all_predictions.shape[1]
	logging.info(f'\n\n Number of predicted trajectories: {total_num_pred}')
	return all_predictions,all_gt,all_obs

if __name__ == '__main__':
	# Loggin format
	config = get_config(agentformer=True)
	logging.basicConfig(format='%(levelname)s: %(message)s',level=20)
	# Device
	if torch.cuda.is_available():
		logging.info(torch.cuda.get_device_name(torch.cuda.current_device()))
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model_name = AGENTFORMER
	""" setup """
	cfg   = Config(configurations[config.id_test])
	cfg.seed = config.seed
	epoch = cfg.get_last_epoch()
	torch.set_default_dtype(torch.float32)
	torch.set_grad_enabled(False)
	log = open(os.path.join(cfg.log_dir, 'log_test.txt'), 'w')

	# Set seed
	logging.info("Seed: {}".format(config.seed))
	prepare_seed(cfg.seed)
	torch.manual_seed(config.seed)
	torch.cuda.manual_seed(config.seed)
	random.seed(config.seed)
	np.random.seed(config.seed)
	""" Load model """
	model_id = cfg.get('model_id', 'agentformer')
	model = model_dict[model_id](cfg)
	if epoch is None:
		logging.error("Could not identify model")
		sys.exit()
	model.set_device(device)
	model.eval()
	if epoch > 0:
		cp_path = cfg.model_path % epoch
		logging.info(f'Loading model from checkpoint: {cp_path}')
		model_cp = torch.load(cp_path, map_location='cpu')
		model.load_state_dict(model_cp['model_dict'], strict=False)

	""" Save results and compute metrics """
	generator = data_generator(cfg, log, split='test', phase='testing')
	tpred_samples,gt_trajectories,obs_trajectories = apply_model(generator, cfg)
	select = np.random.permutation(gt_trajectories.shape[0])
	tpred_samples   = tpred_samples[:,select,:,:]
	gt_trajectories = gt_trajectories[select,:,:]
	obs_trajectories= obs_trajectories[select,:,:]
	tpred_samples_cal =tpred_samples[:,:256,:,:]
	tpred_samples_test=tpred_samples[:,256:,:,:]
	data_cal          = obs_trajectories[:256,:,:]
	data_test         = obs_trajectories[256:,:,:]
	target_cal        = gt_trajectories[:256,:,:]
	target_test       = gt_trajectories[256:,:,:]
	pickle_filename = model_name+"_"+str(SUBDATASETS_NAMES[0][config.id_test])
	save_data_for_calibration(pickle_filename, tpred_samples_cal, tpred_samples_test, data_cal, data_test, target_cal, target_test, None, None, config.id_test)
	for k in range(config.examples):
		idx = np.random.choice(data_cal.shape[0],1)[0]
		plt.figure()
		plt.plot(data_cal[idx,:,0],data_cal[idx,:,1],'g-')
		for j in range(tpred_samples_cal.shape[0]):
			plt.plot(tpred_samples_cal[j,idx,:,0],tpred_samples_cal[j,idx,:,1],'b-')
		plt.plot(target_cal[idx,:,0],target_cal[idx,:,1],'r-')
		plt.axis('equal')
		plt.show()
	# TODO: how to use the same batch for calibration?
