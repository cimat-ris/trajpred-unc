import numpy as np
import pandas as pd
import os, logging
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns
import statistics

import torch
from scipy.stats import gaussian_kde

from scipy.stats import multivariate_normal,multinomial
from sklearn.metrics import auc
from sklearn.isotonic import IsotonicRegression

from utils.plot_utils import plot_calibration_curves, plot_HDR_curves, plot_calibration_pdf, plot_calibration_pdf_traj, plot_traj_world, plot_traj_img_kde
# Local utils helpers
from utils.directory_utils import Output_directories
# Local constants
from utils.constants import IMAGES_DIR, TRAINING_CKPT_DIR, SUBDATASETS_NAMES, CALIBRATION_CONFORMAL_FVAL, CALIBRATION_CONFORMAL_FREL
# HDR utils
from utils.hdr import sort_sample, get_alpha
# Calibration metrics
from utils.calibration_metrics import miscalibration_area,mean_absolute_calibration_error,root_mean_squared_calibration_error

from utils.plot_utils import plot_calibration_curves2

def gaussian_kde_from_gaussianmixture(displacement_prediction, sigmas_prediction, observations, trajectory_id, time_position, resample_size=0):
	"""
	Builds a KDE representation from a Gaussian mixture (output of one of the prediction algorithms)
	Args:
		- displacement_prediction: prediction of displacements
		- sigmas_prediction: covariances of the predictions
		- observations: observations (to translate the predictions)
		- trajectory_id: id of the trajectory
		- time_position: position in the time horizon
		- resample_size: number of samples to produce from the KDE
	Returns:
		- kde: PDF estimation
		- sample_kde: Sampled points (x,y) from PDF
	"""
	# This array will hold the parameters of each element of the mixture
	gaussian_mixture = []

	for idx_ensemble in range(sigmas_prediction.shape[0]):
		# Get means and standard deviations
		sigmas_samples_ensemble = sigmas_prediction[idx_ensemble,trajectory_id,:,:]
		sx, sy, cor = sigmas_samples_ensemble[:, 0], sigmas_samples_ensemble[:, 1], sigmas_samples_ensemble[:, 2]
		sx          = sx[time_position]
		sy          = sy[time_position]

		# Transform in absolute coordinates
		displacement        = displacement_prediction[idx_ensemble,trajectory_id,:,:]
		absolute_prediction = displacement + np.array([observations[trajectory_id,:,:][-1].numpy()])
		mean                = absolute_prediction[time_position, :]
		covariance          = np.array([[sx**2, 0],[0, sy**2]])
		gaussian_mixture.append(multivariate_normal(mean,covariance))
	# Performs the sampling
	pi                 = np.ones((len(gaussian_mixture),))/len(gaussian_mixture)
	partition          = multinomial(n=resample_size,p=pi).rvs(size=1)
	sample_pdf         = []
	for gaussian_id,gaussian in enumerate(gaussian_mixture):
		sample_pdf.append(gaussian.rvs(size=partition[0][gaussian_id]))
	sample_pdf = np.concatenate(sample_pdf,axis=0)
	# TODO: do the samples from the mixture, directly
	# Construimos la gaussiana de la mezcla
	# Mezcla de gaussianas
	# https://faculty.ucmerced.edu/mcarreira-perpinan/papers/cs-99-03.pdf
	# Mean of the mixture
	mean_mixture = np.zeros((2,))
	for j in range(len(gaussian_mixture)):
		mean_mixture += pi[j]*(gaussian_mixture[j].mean)
	# Covariance of the mixture
	cov_mixture = np.zeros((2,2))
	for j in range(len(gaussian_mixture)):
		sub_mean      = gaussian_mixture[j].mean.reshape(2,1) - mean_mixture.reshape(2,1)
		mult_sub_mean = sub_mean @ sub_mean.T
		cov_mixture  +=  pi[j]*(gaussian_mixture[j].cov + mult_sub_mean)

	sample_pdf = np.random.multivariate_normal(mean_mixture, cov_mixture, resample_size)
	# TODO: return a sklearn.KernelDensity or GaussianKDE instead?
	return multivariate_normal(mean_mixture, cov_mixture), sample_pdf

def get_kde(displacement_prediction, observations, trajectory_id, sigmas_samples, time_position=0, gaussian=False, resample_size=1000, relative_coords_flag=False):
	"""
	Builds a KDE representation from the prediction output
	Args:
		- displacement_prediction: prediction of displacements
		- observations: past observations (to translate the predictions)
		- trajectory_id: id of the trajectory
		- sigmas_prediction: covariances of the predictions
		- time_position: position in the time horizon
		- gaussian: flag to handle the prediction output as mean,variance
		- resample_size: number of samples to produce from the KDE
		- relative_coords_flag: to specify how the coordinates are going to be computed, relative (True) or absolute (False)
	Returns:
		- kde: PDF estimation
		- samples_kde: Sampled points (x,y) from PDF
	"""
	# Produces resample_size samples from the pdf
	if gaussian:
		# p.d.f. estimation given as a Gaussian. Sampling points (x,y) from PDF
		kde, samples_kde = gaussian_kde_from_gaussianmixture(displacement_prediction, sigmas_samples, observations, trajectory_id, time_position, resample_size=resample_size)
	else:
		# p.d.f. estimation given implicitly as samples
		if relative_coords_flag:
			samples_kde = displacement_prediction[:, trajectory_id, time_position, :]
		else:
			samples_kde = displacement_prediction[:, trajectory_id, time_position, :] + np.array([observations[trajectory_id,:,:][-1].numpy()])
		if sample_kde.shape[0]<2:
			raise Exception("Needs more than one sample to perform KDE")
		# Use KDE to get a representation of the p.d.f.
		# See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
		kde        = gaussian_kde(samples_kde.T)
		samples_kde = kde.resample(resample_size,0)
	return kde, samples_kde


def get_mass_below_gt(displacement_prediction, observations, ground_truth, sigmas_prediction, time_position=0, gaussian=False, resample_size=1000):
	"""
	Evaluates the probability mass in our predictive distribution below the GT
	Args:
		- displacement_prediction: prediction of displacements
		- observations: past observations (to translate the predictions)
		- sigmas_prediction: covariances of the predictions
		- time_position: position in the time horizon
		- gaussian: flag to handle the prediction output as mean,variance
		- resample_size: number of samples to produce from the KDE
	Returns:
		- Array of probability masses above the GT
	"""
	mass_below_gt = []
	# Traverse each trajectory of the batch
	for trajectory_id in range(displacement_prediction.shape[1]):
		# Ground Truth
		gt = ground_truth[trajectory_id,time_position,:].cpu()
		# KDE representation and samples
		kde, samples_kde = get_kde(displacement_prediction, observations, trajectory_id, sigmas_prediction, time_position=time_position, gaussian=gaussian, resample_size=resample_size)

		#----------------------------------------------------------
		# Evaluate these samples on the p.d.f.
		samples_pdf_values = kde.pdf(samples_kde)

		# Sort the samples in **decreasing order** of their p.d.f. value
		sample_pdf_zip = zip(samples_pdf_values, samples_pdf_values/np.sum(samples_pdf_values))
		sorted_samples_pdf_zip = sorted(sample_pdf_zip, key=lambda x: x[1], reverse=True)
		#----------------------------------------------------------
		# Evaluate the GT on the p.d.f.
		gt_pdf_value = kde.pdf(gt)
		# Select all samples for which the pdf value is above the one of GT
		index = np.where(np.array(sorted_samples_pdf_zip)[:,0] >= gt_pdf_value)[0]
		index = 0 if index.size == 0 else index[-1] # Validate that it is not the first largest element
		# TODO: to be like in the paper should we remove the 1-?
		# Sum all the normalized values for these indices
		alpha_pred = 1 - np.array(sorted_samples_pdf_zip)[:index+1,1].sum()
		mass_below_gt.append(alpha_pred)

	return mass_below_gt


def save_calibration_curves(tpred_samples_test, conf_levels, unc_pcts, cal_pcts, unc_pcts2, cal_pcts2, gaussian=False, idTest=0, position=0, output_dirs=None, show=False):
	"""
	Save calibration curves
	"""

	if gaussian:
		output_image_name = os.path.join(output_dirs.confidence, "confidence_level_cal_IsotonicReg_"+str(idTest)+"_"+str(position)+"_gaussian.pdf")
		plot_calibration_curves(conf_levels, unc_pcts, cal_pcts, output_image_name, show=show)
	else:
		output_image_name = os.path.join(output_dirs.confidence, "confidence_level_cal_IsotonicReg_"+str(idTest)+"_"+str(position)+".pdf")
		plot_calibration_curves(conf_levels, unc_pcts, cal_pcts, output_image_name, show=show)

	if tpred_samples_test is not None:
		if gaussian:
			output_image_name = os.path.join(output_dirs.confidence, "confidence_level_test_IsotonicReg_"+str(idTest)+"_"+str(position)+"_gaussian.pdf")
			plot_calibration_curves(conf_levels, unc_pcts2, cal_pcts2, output_image_name, show=show)
		else:
			output_image_name = os.path.join(output_dirs.confidence, "confidence_level_test_IsotonicReg_"+str(idTest)+"_"+str(position)+".pdf")
			plot_calibration_curves(conf_levels, unc_pcts2, cal_pcts2, output_image_name, show=show)

# TODO: I don't understand why we need two alpha arguments. Is it possible to use only one?
def get_fa(sorted_pdf_values, alpha, alpha_level):
	"""
	Given a set of sorted pdf values (unnormalized and normalized), determine the f_alpha value such that the pdf values above f_alpha sum to alpha.
	Args:
		- sorted_pdf_values: values of the pdf (unnormalized and normalized)
		- alpha: confidence level we are considering
		- alpha_level: confidence level we are considering
	Returns:
		- fa obtained from PDF samples
	"""
	sorted_unnormalized_pdf_values, sorted_normalized_pdf_values = zip(*sorted_pdf_values)
	# TODO: Is it sure 1-alpha?
	index = np.where(np.cumsum(sorted_normalized_pdf_values) >= (1.0-alpha_level))[0]
	if (index.shape[0] == 0) :
		fa = 0.0
	elif (list(index) == [len(sorted_pdf_values)-1]) and (alpha==0.0):
		fa = 0.0
	else:
		fa = sorted_unnormalized_pdf_values[index[0]]
	return fa


def get_gt_within_proportions(conf_levels, isotonic, displacement_prediction, ground_truth, observations, sigmas_prediction, time_position=0, gaussian=False, resample_size=1000):
	"""
	Args:
		- conf_levels: array of cofidence levels to probe
		- isotonic: the (isotonic) function that applies calibration
		- displacement_prediction: prediction of displacements
		- ground_truth: the ground_truth values for future positions
		- observations: past observations (to translate the predictions)
		- sigmas_prediction: covariances of the predictions
		- time_position: position in the time horizon
		- gaussian: flag to handle the prediction output as mean,variance
		- resample_size: number of samples to produce from the KDE
	Returns:
		- Array of probability masses blow the GT
	"""
	# TODO: I feel that this function do the same as get_mass_above_gt no?
	within_proportion_calibrated   = []
	within_proportion_uncalibrated = []
	for trajectory_id,alpha in enumerate(tqdm(conf_levels)):
		# Modified (calibrated) alpha
		new_alpha = isotonic.transform([alpha])
		logging.debug("alpha: {} -- new_alpha: {}".format(alpha,new_alpha))
		within_proportion_calibrated_   = []
		within_proportion_uncalibrated_ = []
		for trajectory_id in range(displacement_prediction.shape[1]):
			# Ground Truth
			gt = ground_truth[trajectory_id,time_position,:].cpu()
			# Get KDE representation
			kde, samples_kde = get_kde(displacement_prediction, observations, trajectory_id, sigmas_prediction, time_position=time_position, gaussian=gaussian, resample_size=resample_size)

			#--------
			# Steps to compute HDRs fa
			# Evaluate these samples over the p.d.f.
			samples_pdf_values = kde.pdf(samples_kde)

			# Sort the samples in decreasing order of their p.d.f. value
			samples_pdf_zip   = zip(samples_pdf_values, samples_pdf_values/np.sum(samples_pdf_values))
			sorted_samples_pdf= sorted(samples_pdf_zip, key=lambda x: x[1], reverse=True)
			fa     = get_fa(sorted_samples_pdf, alpha, new_alpha)
			fa_unc = get_fa(sorted_samples_pdf, alpha, alpha)

			# Evaluate the GT over the p.d.f.
			gt_pdf_value = kde.pdf(gt)
			# Evaluate whether the GT pdf value is above fa
			within_proportion_calibrated_.append(gt_pdf_value >= fa)
			within_proportion_uncalibrated_.append(gt_pdf_value >= fa_unc)
			#-----

		# Save batch results for an specific alpha
		within_proportion_calibrated.append(np.mean(within_proportion_calibrated_))
		within_proportion_uncalibrated.append(np.mean(within_proportion_uncalibrated_))

	return within_proportion_calibrated, within_proportion_uncalibrated


def calibration_IsotonicReg(tpred_samples_cal, data_cal, target_cal, sigmas_samples_cal, time_position = 0, idTest=0, gaussian=False, tpred_samples_test=None, data_test=None, target_test=None, sigmas_samples_test=None,resample_size=1000, output_dirs=None, show_plot=False):
	output_dirs = Output_directories()

	predicted_hdr = get_mass_below_gt(tpred_samples_cal, data_cal, target_cal, sigmas_samples_cal, time_position=time_position, gaussian=gaussian, resample_size=resample_size)

	# Empirical HDR
	empirical_hdr = np.zeros(len(predicted_hdr))

	for i, p in enumerate(predicted_hdr):
		# TODO: check whether < or <=
		empirical_hdr[i] = np.sum(predicted_hdr <= p)/len(predicted_hdr)

	#Visualization: Estimating HDR of Forecast
	output_image_name = os.path.join(output_dirs.calibration, "plot_uncalibrate_"+str(idTest)+".pdf")
	title = "Estimating HDR of Forecast"
	plot_HDR_curves(predicted_hdr, empirical_hdr, output_image_name, title, show=show_plot)

	#-----------------

	# Fit empirical_hdr to predicted_hdr with isotonic regression
	isotonic = IsotonicRegression(out_of_bounds='clip')
	isotonic.fit(empirical_hdr, predicted_hdr)

	# Visualization: Calibration with Isotonic Regression
	output_image_name = os.path.join(output_dirs.calibration, "plot_calibrate_"+str(idTest)+".pdf")
	title = "Calibration with Isotonic Regression"
	plot_HDR_curves(predicted_hdr, isotonic.predict(empirical_hdr), output_image_name, title, show=show_plot)

	#----------------

	conf_levels = np.arange(start=0.0, stop=1.025, step=0.05) # Valores de alpha

	cal_pcts, unc_pcts = get_gt_within_proportions(conf_levels, isotonic, tpred_samples_cal, target_cal, data_cal, sigmas_samples_cal, time_position=time_position, gaussian=gaussian, resample_size=resample_size)
	unc_pcts2 = []
	cal_pcts2 = []

	if tpred_samples_test is not None:
		cal_pcts2, unc_pcts2 = get_gt_within_proportions(conf_levels, isotonic, tpred_samples_test, target_test, data_test, sigmas_samples_test, time_position=time_position, gaussian=gaussian, resample_size=resample_size)

	save_calibration_curves(tpred_samples_test, conf_levels, unc_pcts, cal_pcts, unc_pcts2, cal_pcts2, gaussian=gaussian, idTest=idTest, position=time_position, output_dirs=output_dirs, show=show_plot)
	return 1-conf_levels, unc_pcts, cal_pcts, unc_pcts2, cal_pcts2, isotonic


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

def calibration_density(displacement_prediction, observations, target_test, target_test2, sigmas_prediction, time_position, alpha = 0.85, id_batch=-2, draw=False, gaussian=False, output_dirs=None):
	"""
	Performs uncertainty calibration by using the density values as conformal scores
	Args:
		- displacement_prediction: prediction of the displacements, according to the prediction algorithm
		- observations: past positions
		-
		-
		- sigmas_prediction: covariances of the predictions, according to the prediction algorithm
		- time_position: the position in the time horizon to consider
		- alpha: confidence value to consider
	Returns:
		- Threshold on the density value to be used for marking confidence at least alpha
	"""
	all_density_values = []
	for trajectory_id in range(displacement_prediction.shape[1]):
		# KDE density creation using provided samples
		kde, __ = get_kde(displacement_prediction, observations, trajectory_id, sigmas_prediction, time_position=time_position, gaussian=gaussian, resample_size=1000, relative_coords_flag=True)
		# Evaluates the GT over the KDE and keep the value  in  all_density_values
		gt_evaluation(target_test, target_test2, trajectory_id, time_position, kde, all_density_values, gaussian=gaussian)

	# Sort GT values by decreasing order
	sorted_density_values = sorted(all_density_values, reverse=True)

	# Index of alpha-th sample
	ind = int(len(sorted_density_values)*alpha)
	if ind==len(sorted_density_values):
		Sa = 0.0
	else:
		# The alpha-th largest element gives the threshold
		Sa = sorted_density_values[ind][0]

	# TODO: move this part in another function
	if draw:
		#-------------- For an specific id_batch ----------------------
		# Compute alpha that relates the new Sa in p.d.f.
		# Get sample of interest
		yi = tpred_samples[:, id_batch, time_position, :].T
		gt = target_test[id_batch, time_position,:].detach().numpy()

		# p.d.f creation and sample evaluation in it
		fk = gaussian_kde(yi.T)
		fk_yi = fk.pdf(yi.T)

		# Sort samples
		orden = sorted(fk_yi, reverse=True)
		ind = np.where(np.array(orden) >= Sa)[0]
		ind = 0 if ind.size == 0 else ind[-1] # Validamos que no sea el primer elemento mas grande
		alpha_fk = float(ind)/len(orden)

		output_image_name = os.path.join(output_dirs.hdr, "plot_hdr_%.2f_"%(alpha)+"_"+str(id_batch)+"_"+str(time_position)+"_gt.pdf")
		# Distribution visualization
		plot_calibration_pdf(yi, alpha_fk, gt, Sa, id_batch, output_image_name, alpha=alpha)

		yi = tpred_samples[:, id_batch, time_position, :]  + data_test[id_batch,-1,:].numpy()
		target_test_world = target_test[id_batch, :, :] + data_test[id_batch,-1,:].numpy()

		output_image_name = os.path.join(output_dirs.trajectories_kde, "trajectories_kde_%.2f_"%(alpha)+"_"+str(id_batch)+"_"+str(time_position)+".pdf")
		# Distribution visualization along trajectory
		plot_calibration_pdf_traj(yi, data_test, id_batch, target_test_world, Sa, output_image_name)

	return Sa

def calibration_relative_density(tpred_samples, data_test, target_test, target_test2, sigmas_samples, time_position, alpha = 0.85, id_batch=-2, draw=False, gaussian=False, output_dirs=None):

	list_fk = []
	s_xk_yk = []
	# KDE density creation using provided samples
	for k in range(tpred_samples.shape[1]):
		fk, yi = get_kde(tpred_samples, data_test, k, sigmas_samples, time_position=time_position, gaussian=gaussian, resample_size=1000, relative_coords_flag=True)
		fk_max = fk.pdf(yi).max()
		gt_evaluation(target_test, target_test2, k, time_position, fk, s_xk_yk, gaussian=gaussian, fk_max=fk_max)
		list_fk.append(fk)

	# Sort samples
	orden = sorted(s_xk_yk, reverse=True)
	#  Index of alpha-th sample
	ind = int(len(orden)*alpha)
	if ind==len(orden):
		Sa = 0.0
	else:
		Sa = orden[ind][0] # tomamos el valor del alpha-esimo elemento mas grande

	if draw:
		#-------------- For an specific id_batch ----------------------
		# Compute alpha that relates the new Sa in p.d.f.
		# Get sample of interest
		yi = tpred_samples[:, id_batch, time_position, :] # Seleccionamos las muestras de una trayectoria
		gt = target_test[id_batch, time_position,:].detach().numpy()

		# p.d.f creation and sample evaluation in it
		fk = gaussian_kde(yi.T)
		fk_yi = fk.pdf(yi.T)
		fk_max = fk_yi.max()

		# Sort samples
		orden = sorted(fk_yi, reverse=True)
		ind = np.where(np.array(orden) >= (fk_max*Sa))[0]
		ind = 0 if ind.size == 0 else ind[-1] # Validamos que no sea el primer elemento mas grande
		alpha_fk = float(ind)/len(orden)

		output_image_name = os.path.join(output_dirs.hdr2 , "plot_hdr_%.2f_"%(alpha)+"_"+str(id_batch)+"_"+str(time_position)+"_gt.pdf")
		# Distribution visualization
		plot_calibration_pdf(yi, alpha_fk, gt, Sa, id_batch, output_image_name, alpha=alpha)

	return Sa

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

def get_conformal_pcts(displacement_prediction, observations, target, target2, sigmas_prediction, alpha, fa, method, position=0, gaussian=False):
	"""
	Args:
		- displacement_prediction: prediction of displacements
		- observations: observations (to translate the predictions)
		- target:
		- target2:
		- sigmas_prediction: covariances of the predictions
		- alpha: confidence level
		- fa: calibration threshold
		- position: position in the time horizon
	Returns:
		- calibrated percentages for conformal calibration
		- uncalibrated percentages for conformal calibration
	"""
	perc_within_cal = []
	perc_within_unc = []
	# For each individual trajectory
	for trajectory_id in range(displacement_prediction.shape[1]):

		kde, sample_kde = get_kde(displacement_prediction, observations, trajectory_id, sigmas_prediction, time_position=position, gaussian=gaussian, resample_size=1000, relative_coords_flag=True)

		# Steps to compute HDRs fa
		# Evaluate these samples on the p.d.f.
		sample_pdf = kde.pdf(sample_kde)

		# TODO: instead of sorting the values (in n log n) we could simply count how many of these values are superior to f_pdf
		# Sort samples
		sorted_pdf = sorted(sample_pdf, reverse=True)

		# Index corresponding to the alpha-largest value
		ind = int(len(sorted_pdf)*alpha)
		if ind==len(sorted_pdf):
			fa_unc = 0.0
		else:
			fa_unc = sorted_pdf[ind] # tomamos el valor del alpha-esimo elemento mas grande
		# Why?
		if gaussian:
			gt = target2[trajectory_id,position,:].cpu()
		else:
			gt = target[trajectory_id,position,:].cpu()
		# GT evaluation on the pdf
		f_pdf = kde.pdf(gt)

		if method==CALIBRATION_CONFORMAL_FVAL:
			perc_within_cal.append(f_pdf >= fa)
		elif method==CALIBRATION_CONFORMAL_FREL:
			perc_within_cal.append(f_pdf >= sample_pdf.max()*fa)

		perc_within_unc.append(f_pdf >= fa_unc)

	return perc_within_cal, perc_within_unc


def calibration_Conformal(displacement_prediction_calibration, observations_calibration, gt_calibration, target_cal2, sigmas_samples_cal, position = 0, idTest=0, method=CALIBRATION_CONFORMAL_FVAL, gaussian=False, tpred_samples_test=None, data_test=None, target_test=None, target_test2=None, sigmas_samples_test=None, output_dirs=None, show_plot=False):
	"""
	Args:
		- displacement_prediction_calibration: prediction of displacements on the calibration dataset
		- observations_calibration: observations (to translate the predictions)
		- gt_calibration:
		- target2:
		- sigmas_prediction_calibration: covariances of the predictions
		- alpha: confidence level
		- fa: calibration threshold
		- position: position in the time horizon
	Returns:
		- calibrated percentages for conformal calibration
		- uncalibrated percentages for conformal calibration
	"""
	# Perform calibration for alpha values in the range [0,1]
	conf_levels = np.arange(start=0.0, stop=1.025, step=0.05)

	unc_pcts = []
	cal_pcts = []
	unc_pcts2 = []
	cal_pcts2 = []

	for i,alpha in enumerate(tqdm(conf_levels)):
		logging.debug("***** alpha: {}".format(alpha))
		# Use a conformal approach to obtain a threshold value fa
		if method==CALIBRATION_CONFORMAL_FVAL:
			fa = calibration_density(displacement_prediction_calibration, observations_calibration, gt_calibration, target_cal2, sigmas_samples_cal, position, alpha=alpha, gaussian=gaussian) # NOTE: Unique value for the whole calibration dataset
		elif method==CALIBRATION_CONFORMAL_FREL:
			fa = calibration_relative_density(displacement_prediction_calibration, observations_calibration, gt_calibration, target_cal2, sigmas_samples_cal, position, alpha=alpha, gaussian=gaussian) # NOTE: Unique value for the whole calibration dataset
		else:
			logging.error("Method not implemented.")
			return -1

		perc_within_cal, perc_within_unc = get_conformal_pcts(displacement_prediction_calibration, observations_calibration, gt_calibration, target_cal2, sigmas_samples_cal, alpha, fa, method, position=position, gaussian=gaussian)
		# Save batch results for an specific alpha
		cal_pcts.append(np.mean(perc_within_cal))
		unc_pcts.append(np.mean(perc_within_unc))

		if tpred_samples_test is not None:
			perc_within_cal, perc_within_unc = get_conformal_pcts(tpred_samples_test, data_test, target_test, target_test2, sigmas_samples_test, alpha, fa, method, position=position, gaussian=gaussian)
			# Save batch results for an specific alpha
			cal_pcts2.append(np.mean(perc_within_cal))
			unc_pcts2.append(np.mean(perc_within_unc))

	output_image_name = os.path.join(output_dirs.confidence, "confidence_level_cal_"+str(idTest)+"_conformal"+str(method)+"_"+str(position)+".pdf")
	plot_calibration_curves(conf_levels, unc_pcts, cal_pcts, output_image_name, cal_conformal=True,show=show_plot)

	if tpred_samples_test is not None:
		output_image_name = os.path.join(output_dirs.confidence , "confidence_level_test_"+str(idTest)+"_conformal"+str(method)+"_"+str(position)+".pdf")
		plot_calibration_curves(conf_levels, unc_pcts2, cal_pcts2, output_image_name, cal_conformal=True,show=show_plot)

	return conf_levels, unc_pcts, cal_pcts, unc_pcts2, cal_pcts2

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

def generate_metrics_calibration_IsotonicReg(tpred_samples_cal, data_cal, target_cal, sigmas_samples_cal, id_test, gaussian=False, tpred_samples_test=None, data_test=None, target_test=None, sigmas_samples_test=None, compute_nll=False, show_plot=False):

	#------------Calibration metrics-------------------
	metrics_calibration_data = [["","MACE","RMSCE","MA"]]
	metrics_test_data        = [["","MACE","RMSCE","MA"]]
	key_before = "Before Recalibration"
	key_after  = "After  Recalibration"
	output_dirs = Output_directories()

	# Recorremos cada posicion
	positions_to_test = [11]
	for position in positions_to_test:
		logging.info("Calibration metrics at position: {}".format(position))
		logging.info("Calibration method: Isotonic regression")
		# Apply isotonic regression
		exp_proportions, obs_proportions_unc, obs_proportions_cal, obs_proportions_unc2, obs_proportions_cal2 , isotonic = calibration_IsotonicReg(tpred_samples_cal, data_cal, target_cal, sigmas_samples_cal, time_position = position, idTest=id_test, gaussian=gaussian, tpred_samples_test=tpred_samples_test, data_test=data_test, target_test=target_test, sigmas_samples_test=sigmas_samples_test, output_dirs=output_dirs,show_plot=show_plot)

		# Calibration metrics
		logging.info("Calibration metrics (Calibration dataset)")
		compute_calibration_metrics(exp_proportions, obs_proportions_unc, metrics_calibration_data, position, key_before)
		compute_calibration_metrics(exp_proportions, obs_proportions_cal, metrics_calibration_data, position, key_after)


		if tpred_samples_test is not None:
			logging.info("Calibration evaluation (Test dataset)")
			# Metrics Calibration on testing data
			compute_calibration_metrics(exp_proportions, obs_proportions_unc2, metrics_test_data, position, key_before)
			compute_calibration_metrics(exp_proportions, obs_proportions_cal2, metrics_test_data, position, key_after)

		break

	# Save the metrics results: on calibration dataset
	df = pd.DataFrame(metrics_calibration_data)

	output_csv_name = os.path.join(output_dirs.metrics, "metrics_calibration_cal_IsotonicRegresion_"+str(id_test)+".csv")
	df.to_csv(output_csv_name)

	if tpred_samples_test is not None:
		# Save the metrics results: on test dataset
		df = pd.DataFrame(metrics_test_data)
		output_csv_name = os.path.join(output_dirs.metrics, "metrics_calibration_test_IsotonicRegresion_"+str(id_test)+".csv")
		df.to_csv(output_csv_name)

	if compute_nll:
		# Evaluation of NLL
		position = 11
		ll_cal = []
		ll_uncal = []

		for i in tqdm(range(tpred_samples_test.shape[1])):
			# Ground Truth
			gt = target_test[i,position,:].cpu()
			kde, sample_kde = get_kde(tpred_samples_test, data_test, i, sigmas_samples_test, position=position, gaussian=gaussian, resample_size=1000)

			# Evaluamos la muestra en la pdf
			sample_pdf = kde.pdf(sample_kde)

			sorted_samples  = sort_sample(sample_pdf)
			observed_alphas = np.array([get_alpha(sorted_samples,fk) for fk in sample_pdf ])

			modified_alphas = isotonic.transform(observed_alphas)
			fs_samples_new  = []
			for alpha in modified_alphas:
				fs_samples_new.append(get_falpha(sorted_samples,alpha))
			fs_samples_new    = np.array(fs_samples_new)
			sorted_samples_new= sort_sample(fs_samples_new)
			importance_weights= fs_samples_new/sample_pdf
			# TODO: sometimes transpose, someties not...
			if (sample_kde.shape[0]==importance_weights.shape[0]):
				sample_kde = sample_kde.T
			kernel = gaussian_kde(sample_kde, weights=importance_weights)
			ll_cal.append(kernel.logpdf(gt))
			ll_uncal.append(kde.logpdf(gt))
			#-----

		# Calculamos el Negative LogLikelihood
		nll_cal   = statistics.median(ll_cal)
		nll_uncal = statistics.median(ll_uncal)

		df = pd.DataFrame([["calibrated", "uncalibrated"],[nll_cal, nll_uncal]])
		output_csv_name = os.path.join(output_dirs.calibration, "nll_IsotonicRegresion_"+str(id_test)+".csv")
		df.to_csv(output_csv_name)
		print(df)



def generate_metrics_calibration_conformal(tpred_samples_cal, data_cal, targetrel_cal, target_cal, sigmas_samples_cal, id_test, gaussian=False, tpred_samples_test=None, data_test=None, targetrel_test=None, target_test=None, sigmas_samples_test=None, show_plot=False):
	#--------------------- Calculamos las metricas de calibracion ---------------------------------
	metrics2      = [["","MACE","RMSCE","MA"]]
	metrics3      = [["","MACE","RMSCE","MA"]]
	metrics2_test = [["","MACE","RMSCE","MA"]]
	metrics3_test = [["","MACE","RMSCE","MA"]]
	key_before    = "Before Recalibration"
	key_after     = "After  Recalibration"
	output_dirs   = Output_directories()
	# Recorremos cada posicion para calibrar
	for pos in range(tpred_samples_cal.shape[2]):
		pos = 11
		logging.info("Calibration metrics at position: {}".format(pos))
		gt      = np.cumsum(targetrel_cal, axis=1)
		gt_test = np.cumsum(targetrel_test, axis=1)
		# Uncertainty calibration
		logging.info("Calibration method: Conformal approach with density values")
		exp_proportions, obs_proportions_unc, obs_proportions_cal, obs_proportions_unc2, obs_proportions_cal2 = calibration_Conformal(tpred_samples_cal, data_cal, gt, target_cal, sigmas_samples_cal, position = pos, idTest=id_test, method=2, gaussian=gaussian, tpred_samples_test=tpred_samples_test, data_test=data_test, target_test=gt_test, target_test2=target_test, sigmas_samples_test=sigmas_samples_test, output_dirs=output_dirs, show_plot=show_plot)

		# Metrics Calibration
		logging.info("Calibration metrics (Calibration dataset)")
		compute_calibration_metrics(exp_proportions, obs_proportions_unc, metrics2, pos, key_before)
		compute_calibration_metrics(exp_proportions, obs_proportions_cal, metrics2, pos, key_after)

		if tpred_samples_test is not None:
			# Metrics Calibration Test
			logging.info("Calibration evaluation (Test dataset)")
			compute_calibration_metrics(exp_proportions, obs_proportions_unc2, metrics2_test, pos, key_before)
			compute_calibration_metrics(exp_proportions, obs_proportions_cal2, metrics2_test, pos, key_after)

		logging.info("Calibration method: Conformal approach with relative density values")
		exp_proportions, obs_proportions_unc, obs_proportions_cal, obs_proportions_unc2, obs_proportions_cal2 = calibration_Conformal(tpred_samples_cal, data_cal, gt, target_cal, sigmas_samples_cal, position = pos, idTest=id_test, method=3, gaussian=gaussian, tpred_samples_test=tpred_samples_test, data_test=data_test, target_test=gt_test, target_test2=target_test, sigmas_samples_test=sigmas_samples_test, output_dirs=output_dirs)

		# Metrics Calibration
		logging.info("Calibration metrics (Calibration dataset)")
		compute_calibration_metrics(exp_proportions, obs_proportions_unc, metrics3, pos, key_before)
		compute_calibration_metrics(exp_proportions, obs_proportions_cal, metrics3, pos, key_after)

		if tpred_samples_test is not None:
			# Metrics Calibration Test
			logging.info("Calibration evaluation (Test dataset)")
			compute_calibration_metrics(exp_proportions, obs_proportions_unc2, metrics3_test, pos, key_before)
			compute_calibration_metrics(exp_proportions, obs_proportions_cal2, metrics3_test, pos, key_after)

		break

	# Guardamos los resultados de las metricas
	df = pd.DataFrame(metrics2)
	output_csv_name = os.path.join(output_dirs.metrics, "metrics_calibration_cal_conformal2_"+str(id_test)+".csv")
	df.to_csv(output_csv_name)

	df = pd.DataFrame(metrics3)
	output_csv_name = os.path.join(output_dirs.metrics, "metrics_calibration_cal_conformal3_"+str(id_test)+".csv")
	df.to_csv(output_csv_name)

	if tpred_samples_test is not None:
		# Guardamos los resultados de las metricas de Test
		df = pd.DataFrame(metrics2_test)
		output_csv_name = os.path.join(output_dirs.metrics, "metrics_calibration_test_conformal2_"+str(id_test)+".csv")
		df.to_csv(output_csv_name)

		df = pd.DataFrame(metrics3_test)
		output_csv_name = os.path.join(output_dirs.metrics, "metrics_calibration_test_conformal3_"+str(id_test)+".csv")
		df.to_csv(output_csv_name)


def generate_newKDE(tpred_samples, data_test, targetrel_test, target_test, id_batch=25, position = 0, idTest=0, method=2, test_homography=None, bck=None):

	#-------------- para un id_batch ----------------------
	# Obtenemos la muestra de interes
	yi = tpred_samples[:, id_batch, position, :] # Seleccionamos las muestras de una trayectoria
	# Creamos la pdf para la muestra
	fk = gaussian_kde(yi.T)
	# Evaluamos las muestras en la pdf
	fk_yi = fk.pdf(yi.T) # Evaluamos en la funcion de densidad
	# Ordenamos las muestras
	orden = sorted(fk_yi, reverse=True) # Ordenamos

	#--------------------------------------------------------
	new_fk_yi = []
	# Encontramos el alpha correspondiente a cada muestra
	n = len(orden)
	for i in range(n):
		alpha = (i+1)/n
		logging.debug("***** alpha: {}".format(alpha))
		# Obtenemos el fa con el metodo conformal
		if method==2:
			Sa = calibration_density(tpred_samples, data_test, targetrel_test, None, None, position, alpha=alpha) # NOTA: Es unico para todo el dataset de calibracion
		elif method==3:
			Sa = calibration_relative_density(tpred_samples, data_test, targetrel_test, None, None, position, alpha=alpha) # NOTA: Es unico para todo el dataset de calibracion
		else:
			print("Método incorrecto, valores posibles 2 o 3.")
			return -1

		# Encontramos el nuevo alpha correspondiente
		ind = np.where(np.array(orden) >= Sa)[0]
		ind = 0 if ind.size == 0 else ind[-1] # Validamos que no sea el primer elemento mas grande
		print("ind_pdf: ", ind)
		new_fk_yi.append( orden[ind] )
		#alpha_fk = float(ind)/len(orden)
		#print("alpha_fk: ", alpha_fk)

	# convertimos a array
	new_fk_yi = np.array(new_fk_yi)
	#w_i = new_fk_yi/fk_yi
	w_i = new_fk_yi/new_fk_yi.sum()

	#--------------------------------------------------
	# Visualizamos la distribucion
	plt.figure()
	sns.kdeplot(x=yi[:,0], y=yi[:,1], label='KDE')
	plt.legend()
	plt.xlabel('x-position')
	plt.ylabel('y-position')
	plt.title("Highest Density Regions with Isotonic Regresion, id_batch=" + str(id_batch))
	plt.savefig("images/calib/plot_hdr"+str(method)+"_"+str(id_batch)+"_"+str(position)+"_transform_KDE.pdf")
	plt.close()

	plt.figure()
	sns.kdeplot(x=yi[:,0], y=yi[:,1], color="orange", weights=w_i, label='KDE Weights')
	plt.legend()
	plt.xlabel('x-position')
	plt.ylabel('y-position')
	plt.title("Highest Density Regions with Isotonic Regresion, id_batch=" + str(id_batch))
	plt.savefig("images/calibration/KDE/plot_hdr"+str(method)+"_"+str(id_batch)+"_"+str(position)+"_transform_KDEWeights.pdf")
	plt.close()

	#--------------------------------------------------
	# Visualizamos la distribucion sobre la imagen
	if test_homography is not None:
		# Graficamos la distribucón kde sobre la imagen
		plot_traj_img_kde(tpred_samples, data_test, target_test, test_homography, bck, id_batch, pos=position)
		# Graficamos la distribucón kde modificada sobre la imagen
		plot_traj_img_kde(tpred_samples, data_test, target_test, test_homography, bck, id_batch, pos=position, w_i=w_i)



def generate_uncertainty_evaluation_dataset(batched_test_data, model, num_samples, model_name, args, device=None, dim_pred=12, type="ensemble"):
	#----------- Dataset TEST -------------
	datarel_test_full   = []
	targetrel_test_full = []
	data_test_full      = []
	target_test_full    = []

	for batch_idx, (datarel_test, targetrel_test, data_test, target_test) in enumerate(batched_test_data):
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
 
 #---------------------------------------------------------------------------------------
 def generate_metrics_curves(conf_levels, unc_pcts, cal_pcts, metrics, position, method, output_dirs):
    # Metrics Calibration
    logging.info("Calibration metrics (Calibration dataset)")
    compute_calibration_metrics(conf_levels, unc_pcts, metrics, position, "Before Recalibration")
    compute_calibration_metrics(conf_levels, cal_pcts, metrics, position, "After  Recalibration")
    #print("position: ", position)

    # Save plot_calibration_curves
    output_image_name = os.path.join(output_dirs.confidence, "confidence_level_cal_method_"+str(method)+"_pos_"+str(position)+".pdf")
    plot_calibration_curves2(conf_levels, unc_pcts, cal_pcts, output_image_name)

def save_metrics(metrics_cal, metrics_test, method, output_dirs):
    # Guardamos con un data frame
    df = pd.DataFrame(metrics_cal)
    output_csv_name = os.path.join(output_dirs.metrics, "metrics_calibration_cal_method_" + str(method) + ".csv")
    df.to_csv(output_csv_name)
    print("Metricas del conjunto de calibracion:")
    print(df)

    # Guardamos con un data frame
    df = pd.DataFrame(metrics_test)
    output_csv_name = os.path.join(output_dirs.metrics, "metrics_calibration_test_method_" + str(method) + ".csv")
    df.to_csv(output_csv_name)
    print("Metricas del conjunto de test:")
    print(df)

def get_quantile(score, alpha):
    # Sort samples
    orden = sorted(score, reverse=True)

    # Index of alpha-th sample
    ind = int(np.round(len(orden)*alpha)-1)
    if ind == -1:
        fa = orden[0]*10 # forzamos a que sea muy grande
    else:
        fa = orden[ind]
    #print("ind, fa: ", ind, fa)
    return fa

def get_alpha2(score, fa):
    # Sort samples
    orden = sorted(score, reverse=True)

    # Select all samples for which the pdf value is above the one of GT
    ind = np.where(orden < fa)[0]
    if ind.shape[0] > 0:
        alpha_fa = ind[0]/len(orden)
    else:
        alpha_fa = 1.0
    return alpha_fa

def eval_gaussianmixture(displacement_prediction, sigmas_prediction, resample_size=1000):
    """
    Builds a KDE representation from a Gaussian mixture (output of one of the prediction algorithms)
    Args:
      - displacement_prediction: prediction of displacements
      - sigmas_prediction: covariances of the predictions
      - resample_size: number of samples to produce from the KDE
    Returns:
      - kde: PDF estimation
      - sample_kde: Sampled points (x,y) from PDF
    """
    # This array will hold the parameters of each element of the mixture
    gaussian_mixture = []

    for idx_ensemble in range(sigmas_prediction.shape[0]):
        # Get means and standard deviations
        sigmas_samples_ensemble = sigmas_prediction[idx_ensemble,:]
        sx, sy, cor = sigmas_samples_ensemble[0], sigmas_samples_ensemble[1], sigmas_samples_ensemble[2]

        # Transform in absolute coordinates
        mean                = displacement_prediction[idx_ensemble, :]
        covariance          = np.array([[sx**2, 0],[0, sy**2]])
        gaussian_mixture.append(multivariate_normal(mean,covariance))
    # Performs the sampling
    pi                 = np.ones((len(gaussian_mixture),))/len(gaussian_mixture)
    partition          = multinomial(n=resample_size,p=pi).rvs(size=1)
    sample_pdf         = []
    for gaussian_id,gaussian in enumerate(gaussian_mixture):
        sample_pdf.append(gaussian.rvs(size=partition[0][gaussian_id]))
    sample_pdf = np.concatenate(sample_pdf,axis=0)
    # TODO: do the samples from the mixture, directly
    # Construimos la gaussiana de la mezcla
    # Mezcla de gaussianas
    # https://faculty.ucmerced.edu/mcarreira-perpinan/papers/cs-99-03.pdf
    # Mean of the mixture
    mean_mixture = np.zeros((2,))
    for j in range(len(gaussian_mixture)):
        mean_mixture += pi[j]*(gaussian_mixture[j].mean)
    # Covariance of the mixture
    cov_mixture = np.zeros((2,2))
    for j in range(len(gaussian_mixture)):
        sub_mean      = gaussian_mixture[j].mean.reshape(2,1) - mean_mixture.reshape(2,1)
        mult_sub_mean = sub_mean @ sub_mean.T
        cov_mixture  +=  pi[j]*(gaussian_mixture[j].cov + mult_sub_mean)

    sample_pdf = np.random.multivariate_normal(mean_mixture, cov_mixture, resample_size)
    # TODO: return a sklearn.KernelDensity or GaussianKDE instead?
    return multivariate_normal(mean_mixture, cov_mixture), sample_pdf
    
def eval_density(displacement_prediction, ground_truth, resample_size=1000, sigmas_prediction=None):

    if sigmas_prediction is not None:
        # Creamos la funcion de densidad
        f_density, samples = eval_gaussianmixture(displacement_prediction, sigmas_prediction, resample_size=1000)
	
	# Evaluamos el gt en la funcion de densidad
	f_gt = f_density.pdf(ground_truth)

    else:
        # Creamos la funcion de densidad
        f_density = gaussian_kde(displacement_prediction.T)

        # Muestreamos de la funcion de densidad
        samples = f_density.resample(resample_size,0)

	# Evaluamos el gt en la funcion de densidad
        f_gt = f_density.pdf(ground_truth)[0]

    # Evaluamos las muetras en la funcion de densidad
    f_samples = f_density.pdf(samples)

    return f_gt, f_samples
    
def regresion_isotonic_fit(this_pred_out_abs, data_gt, position, resample_size=1000, sigmas_prediction=None):
    predicted_hdr = []
    # Recorremos todo el conjunto de calibracion (batch)
    for k in range(this_pred_out_abs.shape[1]):

        if sigmas_prediction is not None:
            # Creamos la funcion de densidad, evaluamos el gt y muestreamos
            f_gt0, f_samples = eval_density(this_pred_out_abs[:,k,:], data_gt[k, position, :], resample_size, sigmas_prediction=sigmas_prediction[:, k, position, :])
        else:
          # Creamos la funcion de densidad, evaluamos el gt y muestreamos
            f_gt0, f_samples = eval_density(this_pred_out_abs[:,k,:], data_gt[k, position, :], resample_size)

        predicted_hdr.append( get_alpha2(f_samples, f_gt0) )

    # Empirical HDR
    empirical_hdr = np.zeros(len(predicted_hdr))

    for i, p in enumerate(predicted_hdr):
        # TODO: check whether < or <=
        empirical_hdr[i] = np.sum(np.array(predicted_hdr) <= p)/len(predicted_hdr)

    # Fit empirical_hdr to predicted_hdr with isotonic regression
    isotonic = IsotonicRegression(out_of_bounds='clip')
    isotonic.fit(empirical_hdr, predicted_hdr)

    return isotonic
    
def calibration_test(this_pred_out_abs, data_gt, this_pred_out_abs_test, data_gt_test, position, method, resample_size=1000, gaussian=[None,None]):

    # Perform calibration for alpha values in the range [0,1]
    conf_levels = np.arange(start=0.0, stop=1.025, step=0.05)

    cal_pcts = []
    unc_pcts = []
    cal_pcts_test = []
    unc_pcts_test = []

    if method == 2: # Isotonic
        # Regresion isotonic training
        isotonic = regresion_isotonic_fit(this_pred_out_abs, data_gt, position, resample_size, sigmas_prediction=gaussian[0])

    # Recorremos cada valor de alpha
    #for i,alpha in enumerate(tqdm(conf_levels)):
    for i,alpha in enumerate(conf_levels):
        #logging.debug("***** alpha: {}".format(alpha))
#        print("\n***** alpha: {}".format(alpha))

        # ------------------------------------------------------------

        f_gt = []
        fa_unc = []
        fa_new = []
        f_density_max = []

        f_gt_test = []
        fa_unc_test = []
        fa_new_test = []
        f_density_max_test = []
        # Recorremos todo el conjunto de calibracion (batch)
        for k in range(this_pred_out_abs.shape[1]):

            if gaussian[0] is not None:
                # Creamos la funcion de densidad, evaluamos el gt y muestreamos
                f_gt0, f_samples = eval_density(this_pred_out_abs[:,k,:], data_gt[k, position, :], resample_size, sigmas_prediction=gaussian[0][:, k, position, :])
                f_gt0_test, f_samples_test = eval_density(this_pred_out_abs_test[:,k,:], data_gt_test[k, position, :], resample_size, sigmas_prediction=gaussian[1][:, k, position, :])
            else:
                # Creamos la funcion de densidad, evaluamos el gt y muestreamos
                #f_gt0, f_samples = eval_density(this_pred_out_abs[:,k,:], data_gt[k, position, :], resample_size, sigmas_prediction=gaussian[0])
                f_gt0, f_samples = eval_density(this_pred_out_abs[:,k,:], data_gt[k, position, :], resample_size)
                f_gt0_test, f_samples_test = eval_density(this_pred_out_abs_test[:,k,:], data_gt_test[k, position, :], resample_size)
                
            f_gt.append( f_gt0 )
            f_gt_test.append( f_gt0_test )

            # Obtenemos el cuantil alpha de la distribucion de las muestras
            fa_unc.append( get_quantile(f_samples, alpha) )
            fa_unc_test.append( get_quantile(f_samples_test, alpha) )

            # Verificamos la version del metodo conformal a utilizar
            if method == 1: # conformal relative
                f_density_max.append( f_samples.max() )
                f_density_max_test.append( f_samples_test.max() )
            elif method == 2: # Isotonic
                # Modified (calibrated) alpha
                new_alpha = isotonic.transform([alpha])
                # Obtenemos el cuantil alpha de la distribucion de las muestras
                fa_new.append( get_quantile(f_samples, new_alpha) )
                fa_new_test.append( get_quantile(f_samples_test, new_alpha) )
              
            
        # ------------------------------------------------------------
        f_gt = np.array(f_gt)
        f_gt_test = np.array(f_gt_test)
        
        if method == 1: # conformal relative
            # Obtenemos el cuantil alpha del score
            f_density_max = np.array(f_density_max)
            f_density_max_test = np.array(f_density_max_test)
            fa = get_quantile(f_gt/f_density_max, alpha)

            # Verificamos con la evaluacion del gt
            perc_within_cal = f_gt >= fa*f_density_max
            perc_within_cal_test = f_gt_test >= fa*f_density_max_test

        elif method == 2: # Isotonic
            # Verificamos con la evaluacion del gt
            perc_within_cal = f_gt >= np.array(fa_new)
            perc_within_cal_test = f_gt_test >= np.array(fa_new)

        else:
            # Obtenemos el cuantil alpha del score
            fa = get_quantile(f_gt, alpha)
            #print(position, "    fa: ", fa)

            # Verificamos con la evaluacion del gt
            perc_within_cal = f_gt >= fa
            perc_within_cal_test = f_gt_test >= fa

        # Verificamos con la evaluacion del gt
        perc_within_unc = f_gt >= np.array(fa_unc)
        perc_within_unc_test = f_gt_test >= np.array(fa_unc_test)
        
        # ------------------------------------------------------------

#        print("np.mean(perc_within_cal): ", np.mean(perc_within_cal))
#        print("np.mean(perc_within_unc): ", np.mean(perc_within_unc))
#        print(perc_within_cal.tolist())
#        print(perc_within_unc.tolist())
#        print("np.mean(perc_within_cal_test): ", np.mean(perc_within_cal_test))
#        print("np.mean(perc_within_unc_test): ", np.mean(perc_within_unc_test))
#        print(perc_within_cal_test.tolist())
#        print(perc_within_unc_test.tolist())

        cal_pcts.append(np.mean(perc_within_cal))
        unc_pcts.append(np.mean(perc_within_unc))
        cal_pcts_test.append(np.mean(perc_within_cal_test))
        unc_pcts_test.append(np.mean(perc_within_unc_test))

    return conf_levels, cal_pcts, unc_pcts, cal_pcts_test, unc_pcts_test
    
def generate_metrics_calibration(data_pred, data_obs, data_gt, data_pred_test, data_obs_test, data_gt_test, methods=[0], resample_size=1000, gaussian=[None,None], relative_coords_flag=True):
    # Calculamos para cada metodo
    for method in methods:
        print("\nProcesando metodo: ", method)
        #--------------------- Calculamos las metricas de calibracion ---------------------------------
        metrics_cal  = [["","MACE","RMSCE","MA"]]
        metrics_test = [["","MACE","RMSCE","MA"]]
        output_dirs   = Output_directories()
        # Recorremos cada posicion para calibrar
        for position in range(data_pred.shape[2]):
            if relative_coords_flag:
                # Convert it to absolute (starting from the last observed position)
                this_pred_out_abs      = data_pred[:, :, position, :] + data_obs[:, -1, :].numpy()
                this_pred_out_abs_test = data_pred_test[:, :, position, :] + data_obs_test[:, -1, :].numpy()
            else:
                this_pred_out_abs      = data_pred[:, :, position, :]
                this_pred_out_abs_test = data_pred_test[:, :, position, :]

            # Uncertainty calibration
            logging.info("Calibration metrics at position: {}".format(position))
            conf_levels, cal_pcts, unc_pcts, cal_pcts_test, unc_pcts_test = calibration_test(this_pred_out_abs, data_gt, this_pred_out_abs_test, data_gt_test, position, method, resample_size, gaussian=gaussian)

            # Metrics Calibration for data calibration
            logging.info("Calibration metrics (Calibration dataset)")
            generate_metrics_curves(conf_levels, unc_pcts, cal_pcts, metrics_cal, position, method, output_dirs)

            # Metrics Calibration for data test
            logging.info("Calibration evaluation (Test dataset)")
            generate_metrics_curves(conf_levels, unc_pcts_test, cal_pcts_test, metrics_test, position, method, output_dirs)
        
        #--------------------- Guardamos las metricas de calibracion ---------------------------------
        save_metrics(metrics_cal, metrics_test, method, output_dirs)

 #---------------------------------------------------------------------------------------
 
