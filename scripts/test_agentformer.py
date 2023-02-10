import numpy as np
import argparse
import os, logging
import sys
import subprocess
import shutil
sys.path.append('../AgentFormer')
from data.dataloader import data_generator
from lib.torch import *
from lib.config import Config
from model.model_lib import model_dict
from lib.utils import prepare_seed, print_log, mkdir_if_missing
sys.path.append('.')
from utils.datasets_utils import get_dataset
from utils.config import get_config


def get_model_prediction(data, sample_k):
	model.set_data(data)
	# Past data: data['pre_motion_3D'] is a list of n 8x2 trajectories
	# Ground truth: data['fut_motion_3D'] is a list of n 12x2 trajectories
	print(data.keys())
	print(data['frame'])
	print(data['valid_id'])
	recon_motion_3D, _     = model.inference(mode='recon', sample_num=sample_k)
	sample_motion_3D, data = model.inference(mode='infer', sample_num=sample_k, need_weights=False)
	# Output future samples: sample_motion_3D: sample_kxnx12x2
	sample_motion_3D = sample_motion_3D.transpose(0, 1).contiguous()
	return recon_motion_3D, sample_motion_3D

def get_trajectories(predictions, raw_data):
	pred_num = 0
	fut_data, seq_name, frame, valid_id, pred_mask = raw_data['fut_data'], raw_data['seq'], raw_data['frame'], raw_data['valid_id'], raw_data['pred_mask']
	pred_samples = []
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
				#pred_arr.append(all_data)
			trajectory = np.array(trajectory)
			pred_arr.append(trajectory)
		if len(pred_arr) > 0:
			pred_arr = np.array(pred_arr)
			indices = [0, 1, 13, 15]            # frame, ID, x, z (remove y which is the height)
			pred_arr = pred_arr[:,:,indices]
			pred_samples.append(pred_arr)
	return np.array(pred_samples)

def test_model(generator, cfg):
	total_num_pred = 0
	all_predictions= []
	while not generator.is_epoch_end():
		data = generator()
		if data is None:
			continue
		seq_name, frame = data['seq'], data['frame']
		frame = int(frame)
		sys.stdout.write('testing seq: %s, frame: %06d                \r' % (seq_name, frame))
		sys.stdout.flush()

		gt_motion_3D = torch.stack(data['fut_motion_3D'], dim=0).to(device) * cfg.traj_scale
		with torch.no_grad():
			recon_motion_3D, sample_motion_3D = get_model_prediction(data, cfg.sample_k)
		recon_motion_3D, sample_motion_3D = recon_motion_3D * cfg.traj_scale, sample_motion_3D * cfg.traj_scale

		# Transform the output into our format for trajectories
		all_predictions.append(get_trajectories(sample_motion_3D,data))

	all_predictions = np.concatenate(all_predictions,axis=1)
	total_num_pred  = all_predictions.shape[1]
	logging.info(f'\n\n Number of predicted trajectories: {total_num_pred}')
	return all_predictions
	# save_data_for_calibration(pickle_filename, tpred_samples, tpred_samples_full, data_test, data_test_full, target_test, target_test_full, targetrel_test, targetrel_test_full, sigmas_samples, sigmas_samples_full, config.id_test)

if __name__ == '__main__':
	# Loggin format
	config = get_config(agentformer=True)
	logging.basicConfig(format='%(levelname)s: %(message)s',level=20)
	# Device
	if torch.cuda.is_available():
		logging.info(torch.cuda.get_device_name(torch.cuda.current_device()))
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	""" setup """
	cfg   = Config(config.cfg)
	epoch = cfg.get_last_epoch()
	torch.set_default_dtype(torch.float32)
	torch.set_grad_enabled(False)
	log = open(os.path.join(cfg.log_dir, 'log_test.txt'), 'w')

	prepare_seed(cfg.seed)
	""" Load model """
	model_id = cfg.get('model_id', 'agentformer')
	model = model_dict[model_id](cfg)
	model.set_device(device)
	model.eval()
	if epoch > 0:
		cp_path = cfg.model_path % epoch
		logging.info(f'Loading model from checkpoint: {cp_path}')
		model_cp = torch.load(cp_path, map_location='cpu')
		model.load_state_dict(model_cp['model_dict'], strict=False)

	""" Save results and compute metrics """
	generator = data_generator(cfg, log, split='test', phase='testing')
	test_model(generator, cfg)

	# TODO: how to use the same batch for calibration?
