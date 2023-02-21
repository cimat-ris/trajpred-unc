import numpy as np
import pandas as pd
import os, logging
from tqdm import tqdm
import random
import timeit
import torch
from scipy.stats import gaussian_kde
from scipy.stats import multivariate_normal,multinomial
from sklearn.metrics import auc
from sklearn.isotonic import IsotonicRegression
from utils.plot_utils import plot_calibration_curves, plot_HDR_curves, plot_calibration_pdf, plot_calibration_pdf_traj, plot_traj_world, plot_traj_img_kde
# Local utils helpers
from utils.directory_utils import Output_directories
# Local constants
from utils.constants import IMAGES_DIR, TRAINING_CKPT_DIR, SUBDATASETS_NAMES, CALIBRATION_CONFORMAL_FVAL, CALIBRATION_CONFORMAL_FREL, CALIBRATION_CONFORMAL_ALPHA
# HDR utils
from utils.hdr import sort_sample, get_alpha
# Calibration metrics
from utils.calibration_metrics import miscalibration_area,mean_absolute_calibration_error,root_mean_squared_calibration_error

from utils.plot_utils import plot_calibration_curves2

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
# TODO: to be correct, the sorted values should also be normalized
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

def compute_calibration_metrics(exp_proportions, obs_proportions, metrics_data, position, key):
	"""
	Compute MA, MACE and RMSCE calibration metrics and save those into metrics_data dictionary
	Args:
	Returns:
	"""
	ma    = miscalibration_area(exp_proportions, obs_proportions)
	mace  = mean_absolute_calibration_error(exp_proportions, obs_proportions)
	rmsce = root_mean_squared_calibration_error(exp_proportions, obs_proportions)
	metrics_data.append([key + " pos " + str(position),mace,rmsce,ma])
	logging.info("{}:  MACE: {:.5f}, RMSCE: {:.5f}, MA: {:.5f}".format(key,mace,rmsce,ma))

def generate_uncertainty_evaluation_dataset(batched_test_data, model, num_samples, model_name, args, device=None, dim_pred=12, type="ensemble"):
	#----------- Dataset TEST -------------
	datarel_test_full   = []
	targetrel_test_full = []
	data_test_full      = []
	target_test_full    = []
	total_trajectories  = 0
	for batch_idx, (datarel_test, targetrel_test, data_test, target_test) in enumerate(batched_test_data):
		total_trajectories+=datarel_test.shape[0]
		# The first batch is used for uncertainty calibration, so we skip it
		if batch_idx==0:
			continue

		 # Batches saved into array respectively
		datarel_test_full.append(datarel_test)
		targetrel_test_full.append(targetrel_test)
		data_test_full.append(data_test)
		target_test_full.append(target_test)

	# Batches concatenated to have only one
	datarel_test_full = torch.cat(datarel_test_full, dim=0)
	targetrel_test_full = torch.cat( targetrel_test_full, dim=0)
	data_test_full = torch.cat( data_test_full, dim=0)
	target_test_full = torch.cat( target_test_full, dim=0)
	logging.info('Using test data for uncertainty calibration and evaluation: {} trajectories'.format(total_trajectories))

	# Unique batch predictions obtained
	tpred_samples_full = []
	sigmas_samples_full = []

	# Each model sampled
	for ind in range(num_samples):
		if type == "ensemble":
			model_filename = TRAINING_CKPT_DIR+"/"+model_name+"_"+str(SUBDATASETS_NAMES[args.id_dataset][args.id_test])+"_"+str(ind)+".pth"
			logging.info("Loading {}".format(model_filename))
			model.load_state_dict(torch.load(model_filename))
			model.eval()

		if torch.cuda.is_available():
			datarel_test_full  = datarel_test_full.to(device)

		# Model prediction obtained
		if type == "variational":
			pred, kl, sigmas = model.predict(datarel_test_full, dim_pred=12)
		else:
			pred, sigmas = model.predict(datarel_test_full, dim_pred=12)

		# Sample saved
		tpred_samples_full.append(pred)
		sigmas_samples_full.append(sigmas)

	tpred_samples_full = np.array(tpred_samples_full)
	sigmas_samples_full = np.array(sigmas_samples_full)

	return datarel_test_full, targetrel_test_full, data_test_full, target_test_full, tpred_samples_full, sigmas_samples_full

#-----------------------------------------------------------------------------------
def generate_metrics_curves(conf_levels, unc_pcts, cal_pcts, metrics, position, method, output_dirs, suffix="cal"):
	# Evaluate metrics before/after calibration
	compute_calibration_metrics(conf_levels, unc_pcts, metrics, position, "Before Recalibration")
	compute_calibration_metrics(conf_levels, cal_pcts, metrics, position, "After  Recalibration")
	# Save plot_calibration_curves
	output_image_name = os.path.join(output_dirs.confidence, "confidence_level_"+suffix+"_method_"+str(method)+"_pos_"+str(position)+".pdf")
	plot_calibration_curves2(conf_levels, unc_pcts, cal_pcts, output_image_name)

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


def gaussian_kde_from_gaussianmixture(prediction, sigmas_prediction, kde_size=1000, resample_size=100):
	"""
	Builds a KDE representation from a Gaussian mixture (output of one of the prediction algorithms)
	Args:
	  - prediction: set of position predictions
	  - sigmas_prediction: covariances of the predictions
	  - resample_size: number of samples to produce from the KDE
	Returns:
	  - kde: PDF estimate through KDE
	  - sample_kde: Sampled points (x,y) from the PDF
	"""
	# This array will hold the parameters of each element of the mixture
	gaussian_mixture = []
	# Form the Gaussian mixture
	for idx_ensemble in range(sigmas_prediction.shape[0]):
		# Get means and standard deviations
		sigmas_samples_ensemble = sigmas_prediction[idx_ensemble,:]
		sx, sy, cor = sigmas_samples_ensemble[0], sigmas_samples_ensemble[1], sigmas_samples_ensemble[2]
		# Predictions arrive here in **absolute coordinates**
		mean                = prediction[idx_ensemble, :]
		# TODO: use the correlations too?
		covariance          = np.array([[sx, 0],[0, sy]])
		gaussian_mixture.append(multivariate_normal(mean,covariance))
	# Performs sampling on the Gaussian mixture
	pi                 = np.ones((len(gaussian_mixture),))/len(gaussian_mixture)
	partition          = multinomial(n=kde_size,p=pi).rvs(size=1)
	sample_pdf         = np.zeros((kde_size,2))
	sum                = 0
	for gaussian_id,gaussian in enumerate(gaussian_mixture):
		#sample_pdf.append(gaussian.rvs(size=partition[0][gaussian_id]))
		sample_pdf[sum:sum+partition[0][gaussian_id]]=gaussian.rvs(size=partition[0][gaussian_id])
		sum = sum +partition[0][gaussian_id]
	# Use the samples to generate a KDE
	f_density = gaussian_kde(sample_pdf.T)
	rows_id = random.sample(range(0,sample_pdf.shape[0]),resample_size)
	return f_density,sample_pdf[rows_id, :]

def evaluate_kde(prediction, sigmas_prediction, ground_truth, kde_size=1000, resample_size=100):
	"""
	Builds a KDE representation for the prediction and evaluate the ground truth on it
	Args:
	  - prediction: set of predicted positions
	  - sigmas_prediction: set of covariances on the predicted position (may be None)
	  - ground_truth: set of ground truth positions
	  - resample_size: number of samples to produce from the KDE
	Returns:
	  - f_ground_truth: PDF values at the ground truth points
	  - f_samples: PDF values at the samples
	"""
	if sigmas_prediction is not None:
		# In this case, we use a Gaussian output and create a KDE representation from it
		f_density, samples = gaussian_kde_from_gaussianmixture(prediction,sigmas_prediction,kde_size=kde_size,resample_size=resample_size)
		f_samples      = f_density.pdf(samples.T)
	else:
		# In this case, we just have samples and create the KDE from them
		f_density = gaussian_kde(prediction.T)
		# Then we sample from the obtained representation
		samples   = f_density.resample(resample_size,0)
		f_samples = f_density.pdf(samples)
	# Evaluate the GT and the samples on the obtained KDE
	f_ground_truth = f_density.pdf(ground_truth.T)
	return f_density, f_ground_truth, f_samples, samples

def regresion_isotonic_fit(this_pred_out_abs, data_gt, position, kde_size=1000, resample_size=100, sigmas_prediction=None):
	predicted_hdr = []
	# Recorremos todo el conjunto de calibracion (batch)
	for k in range(this_pred_out_abs.shape[1]):

		if sigmas_prediction is not None:
			# Creamos la funcion de densidad, evaluamos el gt y muestreamos
			__,f_gt0,f_samples,__ = evaluate_kde(this_pred_out_abs[:,k,:], sigmas_prediction[:, k, position, :], data_gt[k, position, :], kde_size, resample_size)
		else:
		  # Creamos la funcion de densidad, evaluamos el gt y muestreamos
			__,f_gt0,f_samples,__ = evaluate_kde(this_pred_out_abs[:,k,:],[None,None],data_gt[k, position, :], kde_size, resample_size)

		predicted_hdr.append(get_alpha(f_samples,f_gt0))

	# Empirical HDR
	empirical_hdr = np.zeros(len(predicted_hdr))

	for i, p in enumerate(predicted_hdr):
		# TODO: check whether < or <=
		empirical_hdr[i] = np.sum(np.array(predicted_hdr) <= p)/len(predicted_hdr)

	# Fit empirical_hdr to predicted_hdr with isotonic regression
	isotonic = IsotonicRegression(out_of_bounds='clip')
	isotonic.fit(empirical_hdr, predicted_hdr)

	return isotonic

def calibrate_density(gt_density_values, alpha):
	"""
	Performs uncertainty calibration by using the density values as conformal scores
	Args:
		- gt_density_values: values of the density function at the GT points
		- alpha: confidence value to consider
	Returns:
		- Threshold on the density value to be used for marking confidence at least alpha
	"""
	# Sort GT values by decreasing order
	gt_density_values     = gt_density_values.reshape(-1)
	sorted_density_values = sorted(gt_density_values,reverse=True)
	# Index of alpha-th sample
	ind = int(np.rint(gt_density_values.shape[0]*alpha))
	if ind==gt_density_values.shape[0]:
		return 0.0
	# The alpha-th largest element gives the threshold
	return sorted_density_values[ind]

def calibrate_relative_density(gt_density_values, samples_density_values, alpha):
	"""
	Performs uncertainty calibration by using the relative density values as conformal scores
	Args:
		- gt_density_values: array of values of the density function at the GT points (for a set of trajectories)
		- samples_density_values: array of array of of values of the density function at some samples (for a set of trajectories)
		- alpha: confidence value to consider
	Returns:
		- Threshold on the relative density value to be used for marking confidence at least alpha
	"""
	gt_relative_density_values = np.divide(gt_density_values.reshape(-1),1.5*samples_density_values.max(axis=1))
	# Sort GT values by decreasing order
	sorted_relative_density_values = sorted(gt_relative_density_values, reverse=True)
	# Index of alpha-th sample
	ind = int(np.rint(len(sorted_relative_density_values)*alpha))
	if ind==len(sorted_relative_density_values):
		return 0.0
	# The alpha-th largest element gives the threshold
	return sorted_relative_density_values[ind]

def calibrate_alpha_density(gt_density_values, samples_density_values, alpha):
	"""
	Performs uncertainty calibration by using the alpha-density values as conformal scores
	Args:
		- gt_density_values: array of values of the density function at the GT points (for a set of trajectories)
		- samples_density_values: array of array of of values of the density function at some samples (for a set of trajectories)
		- alpha: confidence value to consider
	Returns:
		- Threshold on the density value to be used for marking confidence at least alpha
	"""
	alphas_k = []
	gt_density_values          = gt_density_values.reshape(-1)
	# Cycle over the calibration dataset trajectories
	for trajectory_id in range(samples_density_values.shape[0]):
		alphas_k.append(get_alpha(samples_density_values[trajectory_id], gt_density_values[trajectory_id]) )

	# Sort GT values by increasing order
	sorted_alphas_density_values = sorted(alphas_k)
	# Index of alpha-th smallest sample
	ind = int(np.rint(len(sorted_alphas_density_values)*alpha))
	if ind==len(sorted_alphas_density_values):
		return 0.0
	# The alpha-th smallest element gives the threshold
	return sorted_alphas_density_values[ind]

def check_quantile(gt_density_value, samples_density_values, alpha):
	"""
	Args:
		- gt_density_value: evaluations of ground truth on the KDE
		- samples_density_values: evaluations of samples on the KDE
		- alpha: confidence level
	Returns:
		- True/false whether the GT is within an interval of confidence of alpha
	"""
	return (np.mean((samples_density_values>=gt_density_value))<=alpha)

def get_within_proportions(gt_density_values, samples_density_values, method, fa, alpha):
	"""
	Args:
		- gt_density_values: evaluations of ground truth on the KDE
		- samples_density_values: evaluations of samples on the KDE
		- method: choice of the calibration method
		- fa: calibration threshold
		- alpha: confidence level
	Returns:
		- uncalibrated percentages for conformal calibration
		- calibrated percentages for conformal calibration
	"""
	within_cal                 = []
	within_unc                 = []
	gt_density_values          = gt_density_values.reshape(-1)
	# For each individual trajectory
	for trajectory_id in range(gt_density_values.shape[0]):
		# Get quantile value (uncalibrated case)
		within_unc_ = check_quantile(gt_density_values[trajectory_id],samples_density_values[trajectory_id],alpha)
		within_unc.append(within_unc_)
		if method==CALIBRATION_CONFORMAL_FVAL:
			within_cal.append((gt_density_values[trajectory_id]>=fa))
		elif method==CALIBRATION_CONFORMAL_FREL:
			within_cal.append((gt_density_values[trajectory_id]>=samples_density_values[trajectory_id].max()*1.5*fa))
		elif method==CALIBRATION_CONFORMAL_ALPHA:
			# Sort samples p.d.f. values by decreasing order
			sorted_density_values = np.array(sorted(samples_density_values[trajectory_id], reverse=True))
			accum_density_values  = (sorted_density_values/sorted_density_values.sum()).cumsum()
			# First index where accumulated density is superior to fa
			accum_density_value[-1] = 1.0
			ind                   = np.where(accum_density_values>=fa)[0][0]
			#if (gt_density_values[trajectory_id]>sorted_density_values.max()):
			#	print(accum_density_values,alpha,fa,ind,gt_density_values[trajectory_id],sorted_density_values.max())
			#	print(sorted_density_values.shape[0])
			# Index of alpha-th sample
			if ind==len(sorted_density_values):
				fa_new = 0.0
			# The alpha-th largest element gives the threshold
			fa_new = sorted_density_values[ind]
			within_cal.append((gt_density_values[trajectory_id]>=fa_new))

	return np.mean(np.array(within_unc)), np.mean(np.array(within_cal))


def calibrate_uncertainty(prediction,groundtruth,time_position,method,kde_size=1000,resample_size=100, gaussian=None):
	"""
	Args:
		- prediction: output of the prediction algorithm for time_position
		- groundtruth: corresponding ground truth position
		- time_position: considered time step
		- method: choice of the calibration method
		- resample_size: number of samples to use in KDE
		- gaussian: if not None, see prediction as a mean and this as the covariance matrix
	Returns:
		- confidence levels at which calibration has been done
		- uncalibrated percentages for conformal calibration
		- calibrated percentages for conformal calibration
	"""
	# Perform calibration for alpha values in the range [0,1]
	step        = 0.05
	conf_levels = np.arange(start=step, stop=1.0, step=step)
	logging.info("Performing uncertainty calibration")
	cal_pcts = []
	unc_pcts = []

	# Cycle over the confidence levels
	for i,alpha in enumerate(tqdm(conf_levels)):
		# ------------------------------------------------------------
		f_density_max = []
		all_f_samples = []
		all_f_gt      = []
		# Cycle over the trajectories (batch)
		for k in range(prediction.shape[1]):
			if gaussian is not None:
				# Estimate a KDE, produce samples and evaluate the groundtruth on it
				f_kde, f_gt, f_samples,samples = evaluate_kde(prediction[:,k,:],gaussian[:,k,time_position,:],groundtruth[k,time_position,:],kde_size,resample_size)
			else:
				# Estimate a KDE, produce samples and evaluate the groundtruth on it
				f_kde, f_gt, f_samples, samples = evaluate_kde(prediction[:,k,:],None,groundtruth[k,time_position,:],kde_size,resample_size)
			all_f_samples.append(f_samples)
			all_f_gt.append(f_gt)

		# ------------------------------------------------------------
		all_f_gt           = np.array(all_f_gt)
		all_f_samples      = np.array(all_f_samples)

		if method == CALIBRATION_CONFORMAL_FVAL:
			# Calibration based on the density values
			fa = calibrate_density(all_f_gt, alpha)
		elif method == CALIBRATION_CONFORMAL_FREL:
			# Calibration based on the relative density values
			fa = calibrate_relative_density(all_f_gt, all_f_samples, alpha)
		elif method == CALIBRATION_CONFORMAL_ALPHA:
			# Calibration using alpha values on the density values
			fa = calibrate_alpha_density(all_f_gt, all_f_samples, alpha)
		else:
			logging.error("Calibration method not implemented")
			#raise()
			pass
		# Evaluation before/after calibration: Calibration dataset
		proportion_uncalibrated,proportion_calibrated = get_within_proportions(all_f_gt, all_f_samples, method, fa, alpha)
		# ------------------------------------------------------------
		unc_pcts.append(proportion_uncalibrated)
		cal_pcts.append(proportion_calibrated)

	return conf_levels, cal_pcts, unc_pcts

def calibration_test(prediction,groundtruth,prediction_test,groundtruth_test,time_position,method,kde_size=1500,resample_size=200, gaussian=[None,None]):
	# Perform calibration for alpha values in the range [0,1]
	step        = 0.05
	conf_levels = np.arange(start=step, stop=1.0, step=step)

	cal_pcts = []
	unc_pcts = []
	cal_pcts_test = []
	unc_pcts_test = []

	#if method == 3: # Isotonic
	#	# Regresion isotonic training
	#	isotonic = regresion_isotonic_fit(prediction,groundtruth,time_position,resample_size,sigmas_prediction=gaussian[0])

	# Cycle over the confidence levels
	for i,alpha in enumerate(tqdm(conf_levels)):
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
				f_kde, f_gt, f_samples,samples = evaluate_kde(prediction[:,k,:],gaussian[0][:,k,time_position,:],groundtruth[k,time_position,:],kde_size,resample_size)
			else:
				# Estimate a KDE, produce samples and evaluate the groundtruth on it
				f_kde, f_gt, f_samples, samples = evaluate_kde(prediction[:,k,:],None,groundtruth[k,time_position,:],kde_size,resample_size)
			all_f_samples.append(f_samples)
			all_f_gt.append(f_gt)

		for k in range(prediction_test.shape[1]):
			if gaussian[1] is not None:
				# Estimate a KDE, produce samples and evaluate the groundtruth on it
				__, f_gt_test, f_samples_test,__ = evaluate_kde(prediction_test[:,k,:],gaussian[1][:,k,time_position,:],groundtruth_test[k,time_position, :],kde_size,resample_size)
			else:
				# Estimate a KDE, produce samples and evaluate the groundtruth on it
				__, f_gt_test, f_samples_test,__ = evaluate_kde(prediction_test[:,k,:],None,groundtruth_test[k,time_position,:],kde_size,resample_size)
			all_f_samples_test.append(f_samples_test)
			all_f_gt_test.append(f_gt_test)

		# ------------------------------------------------------------
		all_f_gt           = np.array(all_f_gt)
		all_f_gt_test      = np.array(all_f_gt_test)
		all_f_samples      = np.array(all_f_samples)
		all_f_samples_test = np.array(all_f_samples_test)

		if method == CALIBRATION_CONFORMAL_FVAL:
			# Calibration based on the density values
			fa = calibrate_density(all_f_gt, alpha)
		elif method == CALIBRATION_CONFORMAL_FREL:
			# Calibration based on the relative density values
			fa = calibrate_relative_density(all_f_gt, all_f_samples, alpha)
		elif method == CALIBRATION_CONFORMAL_ALPHA:
			# Calibration using alpha values on the density values
			fa = calibrate_alpha_density(all_f_gt, all_f_samples, alpha)
		else:
			logging.error("Calibration method not implemented")
			#raise()
			pass

		# Evaluation before/after calibration: Calibration dataset
		proportion_uncalibrated,proportion_calibrated = get_within_proportions(all_f_gt, all_f_samples, method, fa, alpha)
		# Evaluation before/after calibration: Test dataset
		# proportion_uncalibrated_test,proportion_calibrated_test = get_within_proportions(all_f_gt_test, all_f_samples_test, method, fa, alpha)
		# ------------------------------------------------------------
		unc_pcts.append(proportion_uncalibrated)
		cal_pcts.append(proportion_calibrated)
		unc_pcts_test.append(proportion_uncalibrated_test)
		cal_pcts_test.append(proportion_calibrated_test)

	return conf_levels, cal_pcts, unc_pcts, cal_pcts_test, unc_pcts_test

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
			conf_levels, cal_pcts, unc_pcts, cal_pcts_test, unc_pcts_test = calibration_test(this_pred_out_abs, data_gt, this_pred_out_abs_test, data_gt_test, position, method_id, kde_size, resample_size, gaussian=gaussian)

			# Metrics Calibration for data calibration
			logging.info("Calibration metrics (Calibration dataset)")
			generate_metrics_curves(conf_levels, unc_pcts, cal_pcts, metrics_cal, position, method_id, output_dirs)
			# Metrics Calibration for data test
			logging.info("Calibration evaluation (Test dataset)")
			generate_metrics_curves(conf_levels, unc_pcts_test, cal_pcts_test, metrics_test, position, method_id, output_dirs, suffix='test')

		#--------------------- Guardamos las metricas de calibracion ---------------------------------
		save_metrics(prediction_method_name, metrics_cal, metrics_test, method_id, output_dirs)

 #---------------------------------------------------------------------------------------

def calibration_test_all(prediction,groundtruth,prediction_test,groundtruth_test,time_position,kde_size=1500,resample_size=200, gaussian=[None,None]):
	# Perform calibration for alpha values in the range [0,1]
	step        = 0.05
	conf_levels = np.arange(start=step, stop=1.0, step=step)

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
				f_kde, f_gt, f_samples,samples = evaluate_kde(prediction[:,k,:],gaussian[0][:,k,time_position,:],groundtruth[k,time_position,:],kde_size,resample_size)
			else:
				# Estimate a KDE, produce samples and evaluate the groundtruth on it
				f_kde, f_gt, f_samples, samples = evaluate_kde(prediction[:,k,:],None,groundtruth[k,time_position,:],kde_size,resample_size)
			all_f_samples.append(f_samples)
			all_f_gt.append(f_gt)

		for k in range(prediction_test.shape[1]):
			if gaussian[1] is not None:
				# Estimate a KDE, produce samples and evaluate the groundtruth on it
				__, f_gt_test, f_samples_test,__ = evaluate_kde(prediction_test[:,k,:],gaussian[1][:,k,time_position,:],groundtruth_test[k,time_position, :],kde_size,resample_size)
			else:
				# Estimate a KDE, produce samples and evaluate the groundtruth on it
				__, f_gt_test, f_samples_test,__ = evaluate_kde(prediction_test[:,k,:],None,groundtruth_test[k,time_position,:],kde_size,resample_size)
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
		#proportion_uncalibrated,proportion_calibrated = get_within_proportions(all_f_gt, all_f_samples, method, fa, alpha)
		proportion_uncalibrated0, proportion_calibrated0 = get_within_proportions(all_f_gt, all_f_samples, 0, fa0, alpha)
		proportion_uncalibrated1, proportion_calibrated1 = get_within_proportions(all_f_gt, all_f_samples, 1, fa1, alpha)
		proportion_uncalibrated2, proportion_calibrated2 = get_within_proportions(all_f_gt, all_f_samples, 2, fa2, alpha)
		# Evaluation before/after calibration: Test dataset
		#proportion_uncalibrated_test,proportion_calibrated_test = get_within_proportions(all_f_gt_test, all_f_samples_test, method, fa, alpha)
		proportion_uncalibrated_test0, proportion_calibrated_test0 = get_within_proportions(all_f_gt_test, all_f_samples_test, 0, fa0, alpha)
		proportion_uncalibrated_test1, proportion_calibrated_test1 = get_within_proportions(all_f_gt_test, all_f_samples_test, 1, fa1, alpha)
		proportion_uncalibrated_test2, proportion_calibrated_test2 = get_within_proportions(all_f_gt_test, all_f_samples_test, 2, fa2, alpha)
		# ------------------------------------------------------------
		#unc_pcts.append(proportion_uncalibrated)
		#cal_pcts.append(proportion_calibrated)
		#unc_pcts_test.append(proportion_uncalibrated_test)
		#cal_pcts_test.append(proportion_calibrated_test)

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
		if relative_coords_flag:
			# Convert it to absolute (starting from the last observed position)
			this_pred_out_abs      = predictions_calibration[:,:,position,:]+observations_calibration[:,-1,:]
			this_pred_out_abs_test = data_pred_test[:, :, position, :] + data_obs_test[:, -1, :]
		else:
			this_pred_out_abs      = predictions_calibration[:, :, position, :]
			this_pred_out_abs_test = data_pred_test[:, :, position, :]

		# Uncertainty calibration
		logging.info("Calibration metrics at position: {}".format(position))
		conf_cal = calibration_test_all(this_pred_out_abs, data_gt, this_pred_out_abs_test, data_gt_test, position, kde_size, resample_size, gaussian=gaussian)
		conf_levels0, cal_pcts0, unc_pcts0, cal_pcts_test0, unc_pcts_test0 = conf_cal[0]
		conf_levels1, cal_pcts1, unc_pcts1, cal_pcts_test1, unc_pcts_test1 = conf_cal[1]
		conf_levels2, cal_pcts2, unc_pcts2, cal_pcts_test2, unc_pcts_test2 = conf_cal[2]

		# Metrics Calibration for data calibration
		logging.info("Calibration metrics (Calibration dataset)")
		generate_metrics_curves(conf_levels0, unc_pcts0, cal_pcts0, metrics_cal0, position, 0, output_dirs)
		generate_metrics_curves(conf_levels1, unc_pcts1, cal_pcts1, metrics_cal1, position, 1, output_dirs)
		generate_metrics_curves(conf_levels2, unc_pcts2, cal_pcts2, metrics_cal2, position, 2, output_dirs)
		# Metrics Calibration for data test
		logging.info("Calibration evaluation (Test dataset)")
		generate_metrics_curves(conf_levels0, unc_pcts_test0, cal_pcts_test0, metrics_test0, position, 0, output_dirs, suffix='test')
		generate_metrics_curves(conf_levels1, unc_pcts_test1, cal_pcts_test1, metrics_test1, position, 1, output_dirs, suffix='test')
		generate_metrics_curves(conf_levels2, unc_pcts_test2, cal_pcts_test2, metrics_test2, position, 2, output_dirs, suffix='test')

	#--------------------- Guardamos las metricas de calibracion ---------------------------------
	save_metrics(prediction_method_name, metrics_cal0, metrics_test0, 0, output_dirs)
	save_metrics(prediction_method_name, metrics_cal1, metrics_test1, 1, output_dirs)
	save_metrics(prediction_method_name, metrics_cal2, metrics_test2, 2, output_dirs)

 #---------------------------------------------------------------------------------------
