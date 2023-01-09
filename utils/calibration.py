import numpy as np
import pandas as pd
import os, logging
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns
import statistics

import torch
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity

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

#-----------------------------------------------------------------------------------
def generate_metrics_curves(conf_levels, unc_pcts, cal_pcts, metrics, position, method, output_dirs, suffix="cal"):
	# Evaluate metrics before/after calibration
	compute_calibration_metrics(conf_levels, unc_pcts, metrics, position, "Before Recalibration")
	compute_calibration_metrics(conf_levels, cal_pcts, metrics, position, "After  Recalibration")
	# Save plot_calibration_curves
	output_image_name = os.path.join(output_dirs.confidence, "confidence_level_"+suffix+"_method_"+str(method)+"_pos_"+str(position)+".pdf")
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

def gaussian_kde_from_gaussianmixture(prediction, sigmas_prediction, resample_size=1000):
	"""
	Builds a KDE representation from a Gaussian mixture (output of one of the prediction algorithms)
	Args:
	  - prediction: set of position predictions
	  - sigmas_prediction: covariances of the predictions
	  - resample_size: number of samples to produce from the KDE
	Returns:
	  - kde: PDF estimate through KDE
	  - sample_kde: Sampled points (x,y) from PDF
	"""
	# This array will hold the parameters of each element of the mixture
	gaussian_mixture = []
	np.random.seed(2846)
	# Form the Gaussian mixture
	for idx_ensemble in range(sigmas_prediction.shape[0]):
		# Get means and standard deviations
		sigmas_samples_ensemble = sigmas_prediction[idx_ensemble,:]
		sx, sy, cor = sigmas_samples_ensemble[0], sigmas_samples_ensemble[1], sigmas_samples_ensemble[2]
		# Predictions arrive here in **absolute coordinates**
		mean                = prediction[idx_ensemble, :]
		covariance          = np.array([[sx, 0],[0, sy]])
		gaussian_mixture.append(multivariate_normal(mean,covariance))
	# Performs sampling on the Gaussian mixture
	pi                 = np.ones((len(gaussian_mixture),))/len(gaussian_mixture)
	partition          = multinomial(n=resample_size,p=pi).rvs(size=1)
	sample_pdf         = []
	for gaussian_id,gaussian in enumerate(gaussian_mixture):
		sample_pdf.append(gaussian.rvs(size=partition[0][gaussian_id]))
	sample_pdf = np.concatenate(sample_pdf,axis=0)
	# Use the samples to generate a KDE
	f_density = gaussian_kde(sample_pdf.T)
	return f_density, sample_pdf

def evaluate_kde(prediction, sigmas_prediction, ground_truth, resample_size=1000):
	"""
	Builds a KDE representation for the prediction and evaluate the ground truth on it
	Args:
	  - prediction: set of predicted position
	  - sigmas_prediction: set of covariances on the predicted position
	  - ground_truth: set of ground truth positions
	  - resample_size: number of samples to produce from the KDE
	Returns:
	  - f_ground_truth: PDF values at the ground truth points
	  - f_samples: PDF values at the samples
	"""
	if sigmas_prediction is not None:
		# In this case, we use a Gaussian output and create a KDE representation from it
		f_density, samples = gaussian_kde_from_gaussianmixture(prediction,sigmas_prediction,resample_size=resample_size)
	else:
		# In this case, we just have samples and create the KDE from them
		f_density = gaussian_kde(prediction.T)
		# Then we sample from the obtained representation
		samples = f_density.resample(resample_size,0)

	# Evaluate the GT on the obtained KDE
	f_ground_truth = f_density.pdf(ground_truth.T)
	# Evaluate our samples
	f_samples = f_density.pdf(samples.T)
	return f_density, f_ground_truth, f_samples, samples

def regresion_isotonic_fit(this_pred_out_abs, data_gt, position, resample_size=1000, sigmas_prediction=None):
	predicted_hdr = []
	# Recorremos todo el conjunto de calibracion (batch)
	for k in range(this_pred_out_abs.shape[1]):

		if sigmas_prediction is not None:
			# Creamos la funcion de densidad, evaluamos el gt y muestreamos
			__,f_gt0,f_samples,__ = evaluate_kde(this_pred_out_abs[:,k,:], sigmas_prediction[:, k, position, :], data_gt[k, position, :], resample_size)
		else:
		  # Creamos la funcion de densidad, evaluamos el gt y muestreamos
			__,f_gt0,f_samples,__ = evaluate_kde(this_pred_out_abs[:,k,:],[None,None],data_gt[k, position, :], resample_size)

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

def calibrate_density(gt_density_values, alpha):
	"""
	Performs uncertainty calibration by using the density values as conformal scores
	Args:
		- predictions: prediction of the future positions
		- sigmas_prediction: covariances of the predictions, according to the prediction algorithm
		- time_position: the position in the time horizon to consider
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
	Performs uncertainty calibration by using the density values as conformal scores
	Args:
		- predictions: prediction of the future positions
		- sigmas_prediction: covariances of the predictions, according to the prediction algorithm
		- time_position: the position in the time horizon to consider
		- alpha: confidence value to consider
	Returns:
		- Threshold on the density value to be used for marking confidence at least alpha
	"""
	gt_relative_density_values = []
	gt_density_values          = gt_density_values.reshape(-1)
	# Cycle over the calibration dataset trajectories
	for trajectory_id in range(gt_density_values.shape[0]):
		# KDE density creation using provided samples
		gt_relative_density_values.append(min(1.0,gt_density_values[trajectory_id]/samples_density_values[trajectory_id].max()))
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
    
        - predictions: prediction of the future positions
        - sigmas_prediction: covariances of the predictions, according to the prediction algorithm
        - time_position: the position in the time horizon to consider
        - alpha: confidence value to consider
        - isotonic: optional, model for isotonic regression
    Returns:
        - Threshold on the density value to be used for marking confidence at least alpha
    """
    alphas_k = []
    gt_density_values          = gt_density_values.reshape(-1)
    # Cycle over the calibration dataset trajectories
    for trajectory_id in range(samples_density_values.shape[0]):
        #print(samples_density_values[trajectory_id].shape)
        #print(gt_density_values[trajectory_id])
        #print(get_alpha2(samples_density_values[trajectory_id], gt_density_values[trajectory_id]))
        alphas_k.append( get_alpha2(samples_density_values[trajectory_id], gt_density_values[trajectory_id]) )
    
    # Sort GT values by growing order
    sorted_alphas_density_values = sorted(alphas_k)
    #print(sorted_alphas_density_values)
    #aaaa
    # Index of alpha-th sample
    ind = int(np.rint(len(sorted_alphas_density_values)*alpha))
    if ind==len(sorted_alphas_density_values):
        return 0.0
    # The alpha-th largest element gives the threshold
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
		# Get quantile value
		within_unc_ = check_quantile(gt_density_values[trajectory_id],samples_density_values[trajectory_id],alpha)
		within_unc.append(within_unc_)
		if method==CALIBRATION_CONFORMAL_FVAL:
			within_cal.append((gt_density_values[trajectory_id]>=fa))
		elif method==CALIBRATION_CONFORMAL_FREL:
			within_cal.append((gt_density_values[trajectory_id]>=samples_density_values[trajectory_id].max()*fa))
		elif method==CALIBRATION_CONFORMAL_ALPHA:
			# Sort GT values by decreasing order
			sorted_relative_density_values = sorted(samples_density_values[trajectory_id], reverse=True)
			# Index of alpha-th sample
			ind = int(np.rint(len(sorted_relative_density_values)*fa))
			if ind==len(sorted_relative_density_values):
				fa_new = 0.0
			# The alpha-th largest element gives the threshold
			fa_new = sorted_relative_density_values[ind-1]
			within_cal.append((gt_density_values[trajectory_id]>=fa_new))
            
	return np.mean(np.array(within_unc)), np.mean(np.array(within_cal))


def calibration_test(prediction,groundtruth,prediction_test,groundtruth_test,time_position,method,resample_size=1000, gaussian=[None,None]):

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
				f_kde, f_gt, f_samples,samples = evaluate_kde(prediction[:,k,:],gaussian[0][:,k,time_position,:],groundtruth[k,time_position,:], resample_size)
				__, f_gt_test, f_samples_test,__ = evaluate_kde(prediction_test[:,k,:],gaussian[1][:, k, time_position, :],groundtruth_test[k, time_position, :], resample_size, )
			else:
				# Estimate a KDE, produce samples and evaluate the groundtruth on it
				f_kde, f_gt, f_samples, samples = evaluate_kde(prediction[:,k,:],[None,None],groundtruth[k,time_position,:],resample_size)
				__, f_gt_test, f_samples_test,__ = evaluate_kde(prediction_test[:,k,:],[None,None],groundtruth_test[k, time_position, :], resample_size)
			all_f_samples.append(f_samples)
			all_f_samples_test.append(f_samples_test)
			all_f_gt.append(f_gt)
			all_f_gt_test.append(f_gt_test)
			if False:
				# Here temporarily only :)
				xmin = samples[:,0].min()
				xmax = samples[:,0].max()
				ymin = samples[:,1].min()
				ymax = samples[:,1].max()
				X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
				positions = np.vstack([X.ravel(), Y.ravel()])
				Z = np.reshape(f_kde(positions).T, X.shape)
				fig, ax = plt.subplots()
				ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,extent=[xmin, xmax, ymin, ymax])
				ax.plot(groundtruth[k,time_position,0],groundtruth[k,time_position,1],'ro')
				ax.plot(samples[:,0],samples[:,1],'g+',alpha=0.05)
				ax.set_xlim([xmin, xmax])
				ax.set_ylim([ymin, ymax])
				ax.axis('equal')
				plt.show()


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
		proportion_uncalibrated_test,proportion_calibrated_test = get_within_proportions(all_f_gt_test, all_f_samples_test, method, fa, alpha)
		# ------------------------------------------------------------
		unc_pcts.append(proportion_uncalibrated)
		cal_pcts.append(proportion_calibrated)
		unc_pcts_test.append(proportion_uncalibrated_test)
		cal_pcts_test.append(proportion_calibrated_test)

	return conf_levels, cal_pcts, unc_pcts, cal_pcts_test, unc_pcts_test

def generate_metrics_calibration(predictions_calibration, observations_calibration, data_gt, data_pred_test, data_obs_test, data_gt_test, methods=[0], resample_size=1000, gaussian=[None,None], relative_coords_flag=True):
	# Cycle over methods
	for method in methods:
		logging.info("Evaluating calibration method: {}".format(method))
		#--------------------- Calculamos las metricas de calibracion ---------------------------------
		metrics_cal  = [["","MACE","RMSCE","MA"]]
		metrics_test = [["","MACE","RMSCE","MA"]]
		output_dirs   = Output_directories()
		# Recorremos cada posicion para calibrar
		for position in [3,7,11]:
			if relative_coords_flag:
				# Convert it to absolute (starting from the last observed position)
				this_pred_out_abs      = predictions_calibration[:,:,position,:]+observations_calibration[:,-1,:]
				this_pred_out_abs_test = data_pred_test[:, :, position, :] + data_obs_test[:, -1, :]
			else:
				this_pred_out_abs      = predictions_calibration[:, :, position, :]
				this_pred_out_abs_test = data_pred_test[:, :, position, :]

			# Uncertainty calibration
			logging.info("Calibration metrics at position: {}".format(position))
			conf_levels, cal_pcts, unc_pcts, cal_pcts_test, unc_pcts_test = calibration_test(this_pred_out_abs, data_gt, this_pred_out_abs_test, data_gt_test, position, method, resample_size, gaussian=gaussian)

			# Metrics Calibration for data calibration
			logging.info("Calibration metrics (Calibration dataset)")
			generate_metrics_curves(conf_levels, unc_pcts, cal_pcts, metrics_cal, position, method, output_dirs)
			print(unc_pcts)
			# Metrics Calibration for data test
			logging.info("Calibration evaluation (Test dataset)")
			generate_metrics_curves(conf_levels, unc_pcts_test, cal_pcts_test, metrics_test, position, method, output_dirs, suffix='test')

		#--------------------- Guardamos las metricas de calibracion ---------------------------------
		save_metrics(metrics_cal, metrics_test, method, output_dirs)

 #---------------------------------------------------------------------------------------
