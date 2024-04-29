import numpy as np
import logging
from trajpred_unc.uncertainties.hdr_kde import get_alpha
from trajpred_unc.utils.constants import CALIBRATION_CONFORMAL_FVAL, CALIBRATION_CONFORMAL_FREL, CALIBRATION_CONFORMAL_ALPHA

def evaluate_quantile(gt_density_value, samples_density_values, alpha):
	"""
	Args:
		- gt_density_value: evaluations of ground truth on the KDE
		- samples_density_values: evaluations of samples on the KDE
		- alpha: confidence level
	Returns:
		- True/False for test whether the GT is within the R-alpha region
	"""
	# Sort samples p.d.f. values by decreasing order
	sorted_density_values    = np.array(sorted(samples_density_values, reverse=True))
	accum_density_values     = (sorted_density_values/sorted_density_values.sum()).cumsum()
	accum_density_values[-1] = 1.0
	# First index where accumulated density is superior to alpha
	ind                      = np.where(accum_density_values>=alpha)[0][0]
	# Index of alpha-th sample
	if ind==len(sorted_density_values):
		falpha = 0.0
	else:
		# The alpha-th largest element gives the threshold
		falpha = sorted_density_values[ind]
	return (gt_density_value>=falpha)

def evaluate_within_proportions(gt_density_values, samples_density_values, method, threshold, alpha):
	"""
	Args:
		- gt_density_values: evaluations of ground truth on the KDE
		- samples_density_values: evaluations of samples on the KDE
		- method: choice of the calibration method
		- threshold: calibration threshold
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
		within_unc_ = evaluate_quantile(gt_density_values[trajectory_id],samples_density_values[trajectory_id],alpha)
		within_unc.append(within_unc_)
		if method==CALIBRATION_CONFORMAL_FVAL:
			# Check if the GT density value is above the threshold
			within_cal.append((gt_density_values[trajectory_id]>=threshold))
		elif method==CALIBRATION_CONFORMAL_FREL:
			# Check if the GT relative density value is above the threshold
			within_cal.append((gt_density_values[trajectory_id]>=samples_density_values[trajectory_id].max()*1.5*threshold))
		elif method==CALIBRATION_CONFORMAL_ALPHA:
			# Check if the GT alpha value is above the threshold			
			within_cal.append(evaluate_quantile(gt_density_values[trajectory_id],samples_density_values[trajectory_id],threshold))
	return np.mean(np.array(within_unc)), np.mean(np.array(within_cal))

def recalibrate_density(gt_density_values, alpha):
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

def recalibrate_relative_density(gt_density_values, samples_density_values, alpha):
	"""
	Performs uncertainty calibration by using the relative density values as conformal scores
	Args:
		- gt_density_values: array of values of the density function at the GT points (for a set of trajectories)
		- samples_density_values: array of array of of values of the density function at some samples (for a set of trajectories)
		- alpha: confidence value to consider
	Returns:
		- Threshold on the relative density value to be used for marking confidence at least alpha
	"""
	# Evaluate the relative density values by dividing the GT density values by 1.5 times the maximum sample density value
	gt_relative_density_values = np.divide(gt_density_values.reshape(-1),1.5*samples_density_values.max(axis=1))
	# Sort GT relative density values by decreasing order
	sorted_relative_density_values = sorted(gt_relative_density_values, reverse=True)
	# Index of alpha-th sample
	ind = int(np.rint(len(sorted_relative_density_values)*alpha))
	if ind==len(sorted_relative_density_values):
		return 0.0
	# The alpha-th largest element gives the threshold
	return sorted_relative_density_values[ind]

def recalibrate_alpha_density(gt_density_values, samples_density_values, alpha):
	"""
	Performs uncertainty calibration by using the alpha-density values as conformal scores
	Args:
		- gt_density_values: array of values of the density function at the GT points (for a set of trajectories)
		- samples_density_values: array of array of of values of the density function at some samples (for a set of trajectories)
		- alpha: confidence value to consider
	Returns:
		- Threshold on the alpha value to be used for marking confidence at least alpha
	"""
	alphas_k          = []
	gt_density_values = gt_density_values.reshape(-1)
	# Cycle over the calibration dataset trajectories
	for trajectory_id in range(samples_density_values.shape[0]):
		# For each trajectory, get the alpha value: the proportion of samples with a density value below the GT density value
		alphas_k.append(get_alpha(samples_density_values[trajectory_id],gt_density_values[trajectory_id]))

	# Sort GT alpha values by increasing order
	sorted_alphas_density_values = sorted(alphas_k)
	# Index of alpha-th smallest sample
	ind = int(np.rint(len(sorted_alphas_density_values)*alpha))
	if ind==len(sorted_alphas_density_values):
		return 0.0
	# The alpha-th smallest element gives the threshold
	return sorted_alphas_density_values[ind]

def recalibrate_conformal(all_f_gt, all_f_samples,method,alpha):
	if method == CALIBRATION_CONFORMAL_FVAL:
		# Calibration based on the density values
		fa = recalibrate_density(all_f_gt, alpha)
	elif method == CALIBRATION_CONFORMAL_FREL:
		# Calibration based on the relative density values
		fa = recalibrate_relative_density(all_f_gt, all_f_samples, alpha)
	elif method == CALIBRATION_CONFORMAL_ALPHA:
		# Calibration using alpha values on the density values
		fa = recalibrate_alpha_density(all_f_gt, all_f_samples, alpha)
	else:
		logging.error("Calibration method not implemented")
		#raise()
		pass
	return fa
