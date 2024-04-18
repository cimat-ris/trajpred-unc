import numpy as np
import pandas as pd
import os, logging
from tqdm import tqdm
import torch
from sklearn.isotonic import IsotonicRegression
from trajpred_unc.utils.config import get_model_filename
# Local utils helpers
from trajpred_unc.utils.directory_utils import Output_directories
# HDR utils
from trajpred_unc.uncertainties.hdr_kde import get_alpha
# Calibration metrics
from trajpred_unc.uncertainties.conformal_recalibration import get_within_proportions,calibrate_conformal,calibrate_density,calibrate_relative_density,calibrate_alpha_density
from trajpred_unc.uncertainties.kde import evaluate_kde
from trajpred_unc.uncertainties.calibration_metrics import generate_metrics_curves

def gt_evaluation(target_test, target_test2, trajectory_id, time_position, fk, s_xk_yk, gaussian=False, fk_max=1.0):
	"""
	GT evaluation
	"""
	# TODO: avoid these two cases?
	if gaussian:
		gt = target_test2[trajectory_id,time_position,:].detach().numpy()
		fk_yi = np.array([fk.pdf(gt)])
		s_xk_yk.append(np.array([fk_yi/fk_max]))
	else:
		gt = target_test[trajectory_id,time_position,:].detach().numpy()
		fk_yi = fk.pdf(gt)
		s_xk_yk.append(fk_yi/fk_max)

# Given a value of alpha and sorted values of te density, deduce the alpha-th value of the density
def get_falpha(orden, alpha):
	# We find f_gamma(HDR) from the pdf samples
	orden_idx, orden_val = zip(*orden)
	ind = np.where(np.cumsum(orden_val) >= alpha)[0]
	if ind.shape[0] == 0:
		fa = orden[-1][0]
	else:
		# TODO: why is the index used in one case and the values in the other?
		fa = orden_idx[ind[0]]
	return fa

def get_samples_pdfWeight(pdf,num_sample):
	# Muestreamos de la nueva función de densidad pesada
	return pdf.resample(num_sample)

def generate_uncertainty_evaluation_dataset(batched_test_data,model,config,device=None,type="gaussian"):
	#----------- Dataset TEST -------------
	observations_vels= []
	target_vels      = []
	observations_abss= []
	target_abss      = []
	total_trajectories  = 0
	for batch_idx, (observations_vel,target_vel,observations_abs,target_abs,__,__,__) in enumerate(batched_test_data):
		total_trajectories+=observations_vel.shape[0]
		# The first batch is used for uncertainty calibration, so we skip it
		if batch_idx==0:
			continue
		 # Batches saved into array respectively
		observations_vels.append(observations_vel)
		target_vels.append(target_vel)
		observations_abss.append(observations_abs)
		target_abss.append(target_abs)

	# Batches concatenated to have only one
	observations_vels = torch.cat(observations_vels, dim=0)
	target_vels       = torch.cat(target_vels, dim=0)
	observations_abss = torch.cat(observations_abss, dim=0)
	target_abss       = torch.cat(target_abss, dim=0)
	logging.info('Using test data for uncertainty evaluation: {} trajectories'.format(total_trajectories))

	# Unique batch predictions obtained
	predictions_samples = []
	sigmas_samples      = []

	# Each model sampled
	for ind in range(config["misc"]["model_samples"]):
		if type == "ensemble":
			model_filename = config["train"]["save_dir"]+get_model_filename(config,ensemble_id=ind)
			logging.info("Loading {}".format(model_filename))
			model.load_state_dict(torch.load(model_filename))
			model.eval()
		if torch.cuda.is_available():
			observations_vels  = observations_vels.to(device)
		# Model prediction obtained
		if type == "variational":
			prediction,__,sigmas= model.predict(observations_vels)
		else:
			prediction,sigmas   = model.predict(observations_vels)

		# Sample saved
		predictions_samples.append(prediction)
		sigmas_samples.append(sigmas)

	predictions_samples= np.array(predictions_samples)
	sigmas_samples     = np.array(sigmas_samples)
	return observations_vels,target_vels,observations_abss,target_abss,predictions_samples,sigmas_samples

def generate_uncertainty_calibration_dataset(batched_test_data,model,config,device=None,type="ensemble"):
	#----------- Dataset TEST -------------
	observations_vels= []
	target_vels      = []
	observations_abss= []
	target_abss      = []
	total_trajectories  = 0
	for batch_idx,(observations_vel,target_vel,observations_abs,target_abs,__,__,__) in enumerate(batched_test_data):
		total_trajectories+=observations_vel.shape[0]
		 # Batches saved into array respectively
		observations_vels.append(observations_vel)
		target_vels.append(target_vel)
		observations_abss.append(observations_abs)
		target_abss.append(target_abs)

		# Batches concatenated to have only one
		observations_vels = torch.cat(observations_vels, dim=0)
		target_vels       = torch.cat(target_vels, dim=0)
		observations_abss = torch.cat(observations_abss, dim=0)
		target_abss       = torch.cat(target_abss, dim=0)
		logging.info('Using test data for uncertainty calibration: {} trajectories'.format(total_trajectories))
		break

	# Unique batch predictions obtained
	predictions_samples_cal = []
	sigmas_samples_cal      = []

	# Each model sampled
	for ind in range(config["misc"]["model_samples"]):
		if type == "ensemble":
			model_filename = config["train"]["save_dir"]+get_model_filename(config,ensemble_id=ind)
			logging.info("Loading {}".format(model_filename))
			model.load_state_dict(torch.load(model_filename))
			model.eval()
		if torch.cuda.is_available():
			observations_vels  = observations_vels.to(device)
		# Model prediction obtained
		if type == "variational":
			prediction,__,sigmas= model.predict(observations_vels)
		else:
			prediction,sigmas   = model.predict(observations_vels)

		# Sample saved
		predictions_samples_cal.append(prediction)
		sigmas_samples_cal.append(sigmas)

	predictions_samples_cal= np.array(predictions_samples_cal)
	sigmas_samples_cal     = np.array(sigmas_samples_cal)
	return observations_vels,target_vels,observations_abss,target_abss,predictions_samples_cal,sigmas_samples_cal

#-----------------------------------------------------------------------------------

def save_metrics(prediction_method_name, metrics_cal, metrics_test, method_id, output_dirs):
	# Guardamos con un data frame
	df = pd.DataFrame(metrics_cal)
	output_csv_name = os.path.join(output_dirs.metrics, "calibration_metrics_cal_"+prediction_method_name+"_" + str(method_id) + ".csv")
	df.to_csv(output_csv_name, mode='a', header=not os.path.exists(output_csv_name))
	logging.info("Metrics on the calibration set:")
	print(df)

	# Guardamos con un data frame
	df = pd.DataFrame(metrics_test)
	output_csv_name = os.path.join(output_dirs.metrics, "calibration_metrics_test_"+prediction_method_name+"_" + str(method_id) + ".csv")
	df.to_csv(output_csv_name, mode='a', header=not os.path.exists(output_csv_name))
	logging.info("Metrics on the test set:")
	print(df)

def get_quantile(scores, alpha):
	"""
	Get a certain quantile value from a set of values
	Args:
	  - scores: set of (conformal) scores
	  - alpha: confidence level
	Returns:
	  - quantile value
	"""
	# Sort samples
	sorted_scores = sorted(scores, reverse=True)
	# Index of alpha-th sample
	ind = int(np.rint(len(sorted_scores)*alpha))
	if alpha==0.0:
		return sorted_scores[0]*1000 # Force to a large value
	elif alpha==1.0:
		return 0.0
	return sorted_scores[ind]

def regression_isotonic_fit(predictions_calibration,gt_calibration, kde_size=1000, resample_size=100, sigmas_prediction=None):
	predicted_hdr = []
	# Cycle over the calibration data
	for k in range(predictions_calibration.shape[1]):
		if sigmas_prediction is not None:
			# Evaluate the GT over the KDE
			__,f_gt0,f_samples,__ = evaluate_kde(predictions_calibration[:,k,:],sigmas_prediction[:,k,:],gt_calibration[k, :],kde_size,resample_size)
		else:
			# Evaluate the GT over the KDE
			__,f_gt0,f_samples,__ = evaluate_kde(predictions_calibration[:,k,:],None,gt_calibration[k,:],kde_size,resample_size)
		# Deduce predicted alpha
		predicted_hdr.append(get_alpha(f_samples,f_gt0))

	# Sort GT values in increasing order
	sorted_alphas_density_values = sorted(predicted_hdr)
	new_hdr = np.zeros(len(predicted_hdr))
	for i, alpha in enumerate(predicted_hdr):
		# Index of alpha-th smallest sample
		ind        = int(np.rint((len(sorted_alphas_density_values)-1)*alpha))
		new_hdr[i] = sorted_alphas_density_values[ind]
	# Fit empirical_hdr to predicted_hdr with isotonic regression
	isotonic = IsotonicRegression(out_of_bounds='clip')
	isotonic.fit(predicted_hdr,new_hdr)
	isotonic_inverse = IsotonicRegression(out_of_bounds='clip')
	isotonic_inverse.fit(new_hdr,predicted_hdr)
	return isotonic, isotonic_inverse



def calibrate_and_test(prediction,groundtruth,prediction_test,groundtruth_test,method,kde_size=1500,resample_size=200, gaussian=[None,None]):
	"""
	Args:
		- prediction: output of the prediction algorithm for time_position
		- groundtruth: corresponding ground truth position
		- time_position: apply it at the considered time step
		- method: choice of the calibration method
		- kde_size: number of samples to use in KDE
		- resample_size:
		- gaussian: if not None, see prediction as a mean and this as the covariance matrix
	Returns:
		- confidence levels at which calibration has been done
		- uncalibrated percentages for conformal calibration
		- calibrated percentages for conformal calibration
	"""
	# Perform calibration for alpha values (confidences) in the range [0,1]
	conf_step   = 0.05
	conf_levels = np.arange(start=0.0, stop=1.0+conf_step, step=conf_step)

	# Percentages on calibration dataset
	cal_pcts = []
	unc_pcts = []
	# Percentages on test dataset
	cal_pcts_test = []
	unc_pcts_test = []

	# Cycle over the confidence levels
	for alpha in tqdm(conf_levels):
		if alpha==0.0:
			unc_pcts.append(0.0)
			cal_pcts.append(0.0)
			unc_pcts_test.append(0.0)
			cal_pcts_test.append(0.0)
			continue
		if alpha==1.0:
			unc_pcts.append(1.0)
			cal_pcts.append(1.0)
			unc_pcts_test.append(1.0)
			cal_pcts_test.append(1.0)
			continue
		# ------------------------------------------------------------
		all_f_samples      = []
		all_f_samples_test = []
		all_f_gt           = []
		all_f_gt_test      = []

		# Cycle over the trajectories (batch)
		for k in range(prediction.shape[1]):
			if gaussian[0] is not None:
				# Estimate a KDE, produce samples and evaluate the groundtruth on it
				__,f_gt,f_samples,__ = evaluate_kde(prediction[:,k,:],gaussian[0][:,k,:],groundtruth[k,:],kde_size,resample_size)
			else:
				# Estimate a KDE, produce samples and evaluate the groundtruth on it
				__,f_gt,f_samples,__ = evaluate_kde(prediction[:,k,:],None,groundtruth[k,:],kde_size,resample_size)
			all_f_samples.append(f_samples)
			all_f_gt.append(f_gt)

		# In case we have also test data	
		if prediction_test is not None:
			for k in range(prediction_test.shape[1]):
				if gaussian[1] is not None:
					# Estimate a KDE, produce samples and evaluate the groundtruth on it
					__, f_gt_test, f_samples_test,__ = evaluate_kde(prediction_test[:,k,:],gaussian[1][:,k,:],groundtruth_test[k, :],kde_size,resample_size)
				else:
					# Estimate a KDE, produce samples and evaluate the groundtruth on it
					__, f_gt_test, f_samples_test,__ = evaluate_kde(prediction_test[:,k,:],None,groundtruth_test[k,:],kde_size,resample_size)
				all_f_samples_test.append(f_samples_test)
				all_f_gt_test.append(f_gt_test)

		# ------------------------------------------------------------
		all_f_gt           = np.array(all_f_gt)
		all_f_gt_test      = np.array(all_f_gt_test)
		all_f_samples      = np.array(all_f_samples)
		all_f_samples_test = np.array(all_f_samples_test)

		# Calibration
		threshold = calibrate_conformal(all_f_gt, all_f_samples, alpha, method)

		# Evaluation before/after calibration: Calibration dataset
		proportion_uncalibrated,proportion_calibrated = get_within_proportions(all_f_gt,all_f_samples,method,threshold,alpha)
		# Evaluation before/after calibration: Test dataset
		proportion_uncalibrated_test,proportion_calibrated_test = get_within_proportions(all_f_gt_test,all_f_samples_test,method,threshold,alpha)
		# ------------------------------------------------------------
		unc_pcts.append(proportion_uncalibrated)
		cal_pcts.append(proportion_calibrated)
		unc_pcts_test.append(proportion_uncalibrated_test)
		cal_pcts_test.append(proportion_calibrated_test)

	return conf_levels,cal_pcts,unc_pcts,cal_pcts_test,unc_pcts_test

def generate_metrics_calibration(prediction_method_name, predictions_calibration, observations_calibration, data_gt, data_pred_test, data_obs_test, data_gt_test, methods=[0,1,2], kde_size=1500, resample_size=100, gaussian=[None,None], relative_coords_flag=True, time_positions = [3,7,11]):
	# Cycle over requested methods
	for method_id in methods:
		logging.info("Evaluating uncertainty calibration method: {}".format(method_id))
		#--------------------- Calculamos las metricas de calibracion ---------------------------------
		metrics_cal  = [["","MACE","RMSCE","MA"]]
		metrics_test = [["","MACE","RMSCE","MA"]]
		output_dirs   = Output_directories()
		# Recorremos cada posicion para calibrar
		for position in time_positions:
			if relative_coords_flag:
				# Convert it to absolute (starting from the last observed position)
				this_pred_out_abs      = predictions_calibration[:,:,position,:]+observations_calibration[:,-1,:]
				this_pred_out_abs_test = data_pred_test[:, :, position, :] + data_obs_test[:, -1, :]
			else:
				this_pred_out_abs      = predictions_calibration[:, :, position, :]
				this_pred_out_abs_test = data_pred_test[:, :, position, :]

			# Uncertainty calibration
			logging.info("Calibration metrics at position: {}".format(position))
			conf_levels, cal_pcts, unc_pcts, cal_pcts_test, unc_pcts_test = calibrate_and_test(this_pred_out_abs, data_gt, this_pred_out_abs_test, data_gt_test, position, method_id, kde_size, resample_size, gaussian=gaussian)

			# Metrics Calibration for data calibration
			logging.info("Calibration metrics (Calibration dataset)")
			generate_metrics_curves(conf_levels, unc_pcts, cal_pcts, metrics_cal, position, method_id, output_dirs, prediction_method_name, )
			# Metrics Calibration for data test
			logging.info("Calibration evaluation (Test dataset)")
			generate_metrics_curves(conf_levels, unc_pcts_test, cal_pcts_test, metrics_test, position, method_id, output_dirs, prediction_method_name, suffix='test')

		#--------------------- Guardamos las metricas de calibracion ---------------------------------
		save_metrics(prediction_method_name, metrics_cal, metrics_test, method_id, output_dirs)

 #---------------------------------------------------------------------------------------

def calibrate_and_test_all(prediction,groundtruth,prediction_test,groundtruth_test,kde_size=1500,resample_size=200, gaussian=[None,None]):
	# Perform calibration for alpha values in the range [0,1]
	step        = 0.05
	conf_levels = np.arange(start=0.0, stop=1.0+step, step=step)

	cal_pcts0 = []
	unc_pcts0 = []
	cal_pcts_test0 = []
	unc_pcts_test0 = []

	cal_pcts1 = []
	unc_pcts1 = []
	cal_pcts_test1 = []
	unc_pcts_test1 = []

	cal_pcts2 = []
	unc_pcts2 = []
	cal_pcts_test2 = []
	unc_pcts_test2 = []

	# Cycle over the confidence levels
	for i,alpha in enumerate(tqdm(conf_levels)):
		if alpha==0.0:
			unc_pcts0.append(0.0)
			cal_pcts0.append(0.0)
			unc_pcts_test0.append(0.0)
			cal_pcts_test0.append(0.0)

			unc_pcts1.append(0.0)
			cal_pcts1.append(0.0)
			unc_pcts_test1.append(0.0)
			cal_pcts_test1.append(0.0)

			unc_pcts2.append(0.0)
			cal_pcts2.append(0.0)
			unc_pcts_test2.append(0.0)
			cal_pcts_test2.append(0.0)
			continue
		if alpha==1.0:
			unc_pcts0.append(1.0)
			cal_pcts0.append(1.0)
			unc_pcts_test0.append(1.0)
			cal_pcts_test0.append(1.0)

			unc_pcts1.append(1.0)
			cal_pcts1.append(1.0)
			unc_pcts_test1.append(1.0)
			cal_pcts_test1.append(1.0)

			unc_pcts2.append(1.0)
			cal_pcts2.append(1.0)
			unc_pcts_test2.append(1.0)
			cal_pcts_test2.append(1.0)
			continue

		# ------------------------------------------------------------
		f_density_max = []
		f_density_max_test = []
		all_f_samples      = []
		all_f_samples_test = []
		all_f_gt           = []
		all_f_gt_test      = []

		# Cycle over the trajectories (batch)
		for k in range(prediction.shape[1]):
			if gaussian[0] is not None:
				# Estimate a KDE, produce samples and evaluate the groundtruth on it
				f_kde, f_gt, f_samples,samples = evaluate_kde(prediction[:,k,:],gaussian[0][:,k,:],groundtruth[k,:],kde_size,resample_size)
			else:
				# Estimate a KDE, produce samples and evaluate the groundtruth on it
				f_kde, f_gt, f_samples, samples = evaluate_kde(prediction[:,k,:],None,groundtruth[k,:],kde_size,resample_size)
			all_f_samples.append(f_samples)
			all_f_gt.append(f_gt)

		for k in range(prediction_test.shape[1]):
			if gaussian[1] is not None:
				# Estimate a KDE, produce samples and evaluate the groundtruth on it
				__, f_gt_test, f_samples_test,__ = evaluate_kde(prediction_test[:,k,:],gaussian[1][:,k,:],groundtruth_test[k, :],kde_size,resample_size)
			else:
				# Estimate a KDE, produce samples and evaluate the groundtruth on it
				__, f_gt_test, f_samples_test,__ = evaluate_kde(prediction_test[:,k,:],None,groundtruth_test[k,:],kde_size,resample_size)
			all_f_samples_test.append(f_samples_test)
			all_f_gt_test.append(f_gt_test)

		# ------------------------------------------------------------
		all_f_gt           = np.array(all_f_gt)
		all_f_gt_test      = np.array(all_f_gt_test)
		all_f_samples      = np.array(all_f_samples)
		all_f_samples_test = np.array(all_f_samples_test)

		fa0 = calibrate_density(all_f_gt, alpha)
		fa1 = calibrate_relative_density(all_f_gt, all_f_samples, alpha)
		fa2 = calibrate_alpha_density(all_f_gt, all_f_samples, alpha)

		# Evaluation before/after calibration: Calibration dataset
		proportion_uncalibrated0, proportion_calibrated0 = get_within_proportions(all_f_gt, all_f_samples, 0, fa0, alpha)
		proportion_uncalibrated1, proportion_calibrated1 = get_within_proportions(all_f_gt, all_f_samples, 1, fa1, alpha)
		proportion_uncalibrated2, proportion_calibrated2 = get_within_proportions(all_f_gt, all_f_samples, 2, fa2, alpha)
		# Evaluation before/after calibration: Test dataset
		proportion_uncalibrated_test0, proportion_calibrated_test0 = get_within_proportions(all_f_gt_test, all_f_samples_test, 0, fa0, alpha)
		proportion_uncalibrated_test1, proportion_calibrated_test1 = get_within_proportions(all_f_gt_test, all_f_samples_test, 1, fa1, alpha)
		proportion_uncalibrated_test2, proportion_calibrated_test2 = get_within_proportions(all_f_gt_test, all_f_samples_test, 2, fa2, alpha)
		# ------------------------------------------------------------
		unc_pcts0.append(proportion_uncalibrated0)
		cal_pcts0.append(proportion_calibrated0)
		unc_pcts_test0.append(proportion_uncalibrated_test0)
		cal_pcts_test0.append(proportion_calibrated_test0)

		unc_pcts1.append(proportion_uncalibrated1)
		cal_pcts1.append(proportion_calibrated1)
		unc_pcts_test1.append(proportion_uncalibrated_test1)
		cal_pcts_test1.append(proportion_calibrated_test1)

		unc_pcts2.append(proportion_uncalibrated2)
		cal_pcts2.append(proportion_calibrated2)
		unc_pcts_test2.append(proportion_uncalibrated_test2)
		cal_pcts_test2.append(proportion_calibrated_test2)

	return [conf_levels, cal_pcts0, unc_pcts0, cal_pcts_test0, unc_pcts_test0], [conf_levels, cal_pcts1, unc_pcts1, cal_pcts_test1, unc_pcts_test1], [conf_levels, cal_pcts2, unc_pcts2, cal_pcts_test2, unc_pcts_test2]

 #---------------------------------------------------------------------------------------

def generate_metrics_calibration_all(prediction_method_name, predictions_calibration, observations_calibration, data_gt, data_pred_test, data_obs_test, data_gt_test, kde_size=1500, resample_size=100, gaussian=[None,None], relative_coords_flag=True, time_positions = [3,7,11]):
	logging.info("Evaluating uncertainty calibration method: 0, 1, 2")
	#--------------------- Calculamos las metricas de calibracion ---------------------------------
	metrics_cal0  = [["","MACE","RMSCE","MA"]]
	metrics_cal1  = [["","MACE","RMSCE","MA"]]
	metrics_cal2  = [["","MACE","RMSCE","MA"]]
	metrics_test0 = [["","MACE","RMSCE","MA"]]
	metrics_test1 = [["","MACE","RMSCE","MA"]]
	metrics_test2 = [["","MACE","RMSCE","MA"]]
	output_dirs   = Output_directories()
	# Recorremos cada posicion para calibrar
	for position in time_positions:
		logging.info("Calibration metrics at position: {}".format(position))
		#FIXME: is it still useful to have this flag?
		if relative_coords_flag:
			# Convert it to absolute (starting from the last observed position)
			this_pred_out_abs      = predictions_calibration[:,:,position,:]+observations_calibration[:,-1,:]
			this_pred_out_abs_test = data_pred_test[:, :, position, :] + data_obs_test[:, -1, :]
		else:
			this_pred_out_abs      = predictions_calibration[:, :, position, :]
			this_pred_out_abs_test = data_pred_test[:, :, position, :]
		if gaussian[0] is not None:
			gaussian_t = [gaussian[0][:, :, position, :], gaussian[1][:, :, position, :]]
		else:
			gaussian_t = gaussian	
		# Uncertainty calibration
		conf_cal = calibrate_and_test_all(this_pred_out_abs,data_gt[:,position,:],this_pred_out_abs_test,data_gt_test[:,position,:], kde_size, resample_size, gaussian=gaussian_t)
		conf_levels0, cal_pcts0, unc_pcts0, cal_pcts_test0, unc_pcts_test0 = conf_cal[0]
		conf_levels1, cal_pcts1, unc_pcts1, cal_pcts_test1, unc_pcts_test1 = conf_cal[1]
		conf_levels2, cal_pcts2, unc_pcts2, cal_pcts_test2, unc_pcts_test2 = conf_cal[2]

		# Metrics Calibration for data calibration
		logging.info("Calibration metrics (Calibration dataset)")
		generate_metrics_curves(conf_levels0, unc_pcts0, cal_pcts0, metrics_cal0, position, 0, output_dirs, prediction_method_name)
		generate_metrics_curves(conf_levels1, unc_pcts1, cal_pcts1, metrics_cal1, position, 1, output_dirs, prediction_method_name)
		generate_metrics_curves(conf_levels2, unc_pcts2, cal_pcts2, metrics_cal2, position, 2, output_dirs, prediction_method_name)
		# Metrics Calibration for data test
		logging.info("Calibration evaluation (Test dataset)")
		generate_metrics_curves(conf_levels0, unc_pcts_test0, cal_pcts_test0, metrics_test0, position, 0, output_dirs, prediction_method_name, suffix='test')
		generate_metrics_curves(conf_levels1, unc_pcts_test1, cal_pcts_test1, metrics_test1, position, 1, output_dirs, prediction_method_name, suffix='test')
		generate_metrics_curves(conf_levels2, unc_pcts_test2, cal_pcts_test2, metrics_test2, position, 2, output_dirs, prediction_method_name, suffix='test')

	#--------------------- Guardamos las metricas de calibracion ---------------------------------
	save_metrics(prediction_method_name, metrics_cal0, metrics_test0, 0, output_dirs)
	save_metrics(prediction_method_name, metrics_cal1, metrics_test1, 1, output_dirs)
	save_metrics(prediction_method_name, metrics_cal2, metrics_test2, 2, output_dirs)

 #---------------------------------------------------------------------------------------
