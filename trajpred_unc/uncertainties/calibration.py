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
from trajpred_unc.uncertainties.conformal_recalibration import evaluate_within_proportions,recalibrate_conformal
from trajpred_unc.uncertainties.kde import evaluate_kde
from trajpred_unc.uncertainties.calibration_metrics import generate_metrics_curves


# Given a value of alpha and sorted values of the density, deduce the alpha-th value of the density
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

def generate_uncertainty_evaluation_dataset(batched_test_data,model,config,device=None,type="gaussian"):
	#----------- Dataset TEST -------------
	observations= []
	target      = []
	total_trajectories  = 0
	# Keep observations and targets
	for (observations_,targets_,__,__,__) in batched_test_data:
		total_trajectories+=observations_.shape[0]
		# Batches saved into array respectively
		observations.append(observations_)
		target.append(targets_)
	# Batches concatenated
	observations = torch.cat(observations, dim=0)
	target       = torch.cat(target, dim=0)
	logging.info('Using test data for uncertainty evaluation: {} trajectories'.format(total_trajectories))

	# Unique batch predictions obtained
	predictions_samples = []
	sigmas_samples      = []

	# Each sampled model
	nsamples = 1
	if config["misc"]["ensemble"]:
		nsamples = config["misc"]["model_samples"]
	for ind in range(nsamples):
		if type == "ensemble":
			model_filename = config["train"]["save_dir"]+get_model_filename(config,ensemble_id=ind)
			logging.info("Loading {}".format(model_filename))
			model.load_state_dict(torch.load(model_filename))
			model.eval()
		if torch.cuda.is_available():
			observations  = observations.to(device)
		# Model prediction obtained
		if type == "variational":
			prediction,__,sigmas= model.predict(observations[:,:,2:4 ],observations[:,:,0:2])
		else:
			prediction,sigmas   = model.predict(observations[:,:,2:4],observations[:,:,0:2])
		# Sample saved
		predictions_samples.append(prediction)
		sigmas_samples.append(sigmas)

	predictions_samples= np.array(predictions_samples)
	predictions_samples= np.swapaxes(predictions_samples,0,1)
	sigmas_samples     = np.array(sigmas_samples)
	sigmas_samples     = np.swapaxes(sigmas_samples,0,1)
	return observations,target,predictions_samples,sigmas_samples

#-----------------------------------------------------------------------------------
def save_metrics(prediction_method_name, metrics_cal, metrics_test, method_id, output_dirs):
	# Guardamos con un data frame
	df              = pd.DataFrame(metrics_cal)
	output_csv_name = os.path.join(output_dirs.metrics, "calibration_metrics_cal_"+prediction_method_name+"_" + str(method_id) + ".csv")
	df.to_csv(output_csv_name, mode='a', header=not os.path.exists(output_csv_name))
	logging.info("Metrics on the calibration set:")
	print(df)

	# Guardamos con un data frame
	df              = pd.DataFrame(metrics_test)
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
	for k in range(predictions_calibration.shape[0]):
		if sigmas_prediction is not None:
			# Evaluate the GT over the KDE
			__,f_gt0,f_samples,__ = evaluate_kde(predictions_calibration[k,:,:],sigmas_prediction[k,:,:],gt_calibration[k, :],kde_size,resample_size)
		else:
			# Evaluate the GT over the KDE
			__,f_gt0,f_samples,__ = evaluate_kde(predictions_calibration[k,:,:],None,gt_calibration[k,:],kde_size,resample_size)
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

	"""
	Recalibrate and test with a list of specified methods
	Args:
	  - prediction: set of predicted positions ntrajectories x nsamples x nsteps x 2
	  - groundtruth: set of ground truth positions ntrajectories x nsteps x 2
	  - prediction_test: set of predicted positions ntrajectories x nsamples x nsteps x 2
	  - groundtruth_test: set of ground truth positions ntrajectories x nsteps x 2
	  - methods: list of methods to evaluate
	  - kde_size: size of the samples to build the KDE
	  - resample_size: size of the resampling
	  - gaussian: list of variances associated to the prediction (if available)
	Returns:
	  - results: dictionary with the results
	"""
def recalibrate_and_test(prediction,groundtruth,prediction_test,groundtruth_test,methods=[0,1,2],kde_size=1500,resample_size=200,gaussian=[None,None]):
	# Perform calibration for alpha values in the range [0,1]
	step        = 0.05
	conf_levels = np.arange(start=0.0, stop=1.0+step, step=step)
	# Keep the results in a dictionary
	results     = {}
	for method in methods:
		results[method] = {}
		results[method]["confidence_levels"]=conf_levels
		results[method]["recalibrated"] = {}
		results[method]["raw"] = {}
		results[method]["recalibrated"]["onCalibrationData"]= []
		results[method]["raw"]["onCalibrationData"]         = []
		results[method]["recalibrated"]["onTestData"]       = []
		results[method]["raw"]["onTestData"]                = []

	# Cycle over the confidence levels
	for alpha in tqdm(conf_levels):
		if alpha==0.0:
			for method in methods:
				results[method]["recalibrated"]["onCalibrationData"].append(0.0)
				results[method]["raw"]["onCalibrationData"].append(0.0)
				results[method]["recalibrated"]["onTestData"].append(0.0)
				results[method]["raw"]["onTestData"].append(0.0)
			continue
		if alpha==1.0:
			for method in methods:
				results[method]["recalibrated"]["onCalibrationData"].append(1.0)
				results[method]["raw"]["onCalibrationData"].append(1.0)
				results[method]["recalibrated"]["onTestData"].append(1.0)
				results[method]["raw"]["onTestData"].append(1.0)
			continue

		# ------------------------------------------------------------
		all_f_samples_calib= []
		all_f_samples_test = []
		all_f_gt_calib     = []
		all_f_gt_test      = []
		# Cycle over the trajectories (calibration batch)
		for k in range(prediction.shape[0]):
			if gaussian[0] is not None:
				# Estimate a KDE, produce samples and evaluate the groundtruth on it
				__,f_gt,f_samples,__ = evaluate_kde(prediction[k,:,:],gaussian[0][k,:,:],groundtruth[k,:],kde_size,resample_size)
			else:
				# Estimate a KDE, produce samples and evaluate the groundtruth on it
				__,f_gt,f_samples,__ = evaluate_kde(prediction[k,:,:],None,groundtruth[k,:],kde_size,resample_size)
			all_f_samples_calib.append(f_samples)
			all_f_gt_calib.append(f_gt)
		if prediction_test is not None:
			for k in range(prediction_test.shape[0]):
				if gaussian[1] is not None:
					# Estimate a KDE, produce samples and evaluate the groundtruth on it
					__, f_gt_test, f_samples_test,__ = evaluate_kde(prediction_test[k,:,:],gaussian[1][k,:,:],groundtruth_test[k,:],kde_size,resample_size)
				else:
					# Estimate a KDE, produce samples and evaluate the groundtruth on it
					__, f_gt_test, f_samples_test,__ = evaluate_kde(prediction_test[k,:,:],None,groundtruth_test[k,:],kde_size,resample_size)
				all_f_samples_test.append(f_samples_test)
				all_f_gt_test.append(f_gt_test)

		# ------------------------------------------------------------
		all_f_gt_calib     = np.array(all_f_gt_calib)
		all_f_gt_test      = np.array(all_f_gt_test)
		all_f_samples_calib= np.array(all_f_samples_calib)
		all_f_samples_test = np.array(all_f_samples_test)
		for method in methods:
			recalibration_threshold                                    = recalibrate_conformal(all_f_gt_calib, all_f_samples_calib, method, alpha)
			# Evaluation before/after calibration: Calibration dataset
			proportion_uncalibrated0, proportion_calibrated0           = evaluate_within_proportions(all_f_gt_calib, all_f_samples_calib, method, recalibration_threshold, alpha)
			# Evaluation before/after calibration: Test dataset
			proportion_uncalibrated_test0, proportion_calibrated_test0 = evaluate_within_proportions(all_f_gt_test, all_f_samples_test, method, recalibration_threshold, alpha)
			results[method]["recalibrated"]["onCalibrationData"].append(proportion_calibrated0)
			results[method]["raw"]["onCalibrationData"].append(proportion_uncalibrated0)
			results[method]["recalibrated"]["onTestData"].append(proportion_calibrated_test0)
			results[method]["raw"]["onTestData"].append(proportion_uncalibrated_test0)
			# ------------------------------------------------------------

	return results

 #---------------------------------------------------------------------------------------

def generate_calibration_metrics(prediction_method_name, predictions_calibration, observations_calibration, data_gt, data_pred_test, data_obs_test, data_gt_test, methods=[0,1,2], kde_size=1500, resample_size=100, gaussian=[None,None], time_positions = [3,7,11]):
	logging.info("Evaluating uncertainty calibration methods: {}".format(methods))	
	metrics_onCalibration = []
	metrics_onTest        = []
	for method in methods:
		metrics_onCalibration.append([["","MACE","RMSCE","MA"]])
		metrics_onTest.append([["","MACE","RMSCE","MA"]])
	output_dirs           = Output_directories()
	# Evaluate calibration metrics for all time positions
	for position in time_positions:

		logging.info("Calibration metrics at time position: {}".format(position))
		this_pred_out_abs      = predictions_calibration[:, :, position, :]
		this_pred_out_abs_test = data_pred_test[:, :, position, :]
		if gaussian[0] is not None:
			gaussian_t = [gaussian[0][:,:,position,:], gaussian[1][:,:,position, :]]
		else:
			gaussian_t = gaussian	
		# Uncertainty calibration
		results = recalibrate_and_test(this_pred_out_abs,data_gt[:,position,:2],this_pred_out_abs_test,data_gt_test[:,position,:2],methods,kde_size,resample_size,gaussian=gaussian_t)
		conf_levels=results[0]["confidence_levels"]
		# Metrics Calibration for data calibration
		logging.info("Calibration metrics (Calibration dataset)")
		for method in methods:
			generate_metrics_curves(conf_levels,results[method]["raw"]["onCalibrationData"],results[method]["recalibrated"]["onCalibrationData"],metrics_onCalibration[method],position,method,output_dirs,prediction_method_name)
		# Metrics Calibration for data test
		logging.info("Calibration evaluation (Test dataset)")
		for method in methods:
			generate_metrics_curves(conf_levels,results[method]["raw"]["onTestData"],results[method]["recalibrated"]["onTestData"],metrics_onTest[method],position,method,output_dirs, prediction_method_name, suffix='test')

	#--------------------- Save calibration metrics ---------------------------------	
	for method in methods:
		save_metrics(prediction_method_name,metrics_onCalibration[method],metrics_onTest[method],method,output_dirs)

 #---------------------------------------------------------------------------------------
