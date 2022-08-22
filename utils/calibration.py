import numpy as np
import pandas as pd
import os, logging
from matplotlib import pyplot as plt
import seaborn as sns
import statistics

import torch
from scipy.stats import gaussian_kde

from scipy.stats import multivariate_normal
from sklearn.metrics import auc
from sklearn.isotonic import IsotonicRegression

from utils.plot_utils import plot_calibration_curves, plot_HDR_curves, plot_traj_world, plot_traj_img_kde
# Local utils helpers
from utils.directory_utils import mkdir_p
# Local constants
from utils.constants import IMAGES_DIR
# HDR utils
from utils.hdr import sort_sample, get_alpha
# Calibration metrics
from utils.calibration_metrics import miscalibration_area,mean_absolute_calibration_error,root_mean_squared_calibration_error


def gaussian_kde2(pred, sigmas_samples, data_test, target_test, i, position, resample_size=0 , display=False, idTest=0):

	# Estimamos la gaussiana con los parametros que salen del modelo
	param_gaussiana = []
	if display:
		plt.figure(figsize=(12,10))

	for ind_ensemble in range(sigmas_samples.shape[0]):
		# Procesamos las medias y sigmas [2, 16, 12, 3]
		# Extraemos los valores para la covarianza
		sigmas_samples_ensemble = sigmas_samples[ind_ensemble, i,:,:]
		sx, sy, cor = sigmas_samples_ensemble[:, 0], sigmas_samples_ensemble[:, 1], sigmas_samples_ensemble[:, 2]

		# Exponential to get a positive value for std dev
		sx   = sx[position]
		sy   = sy[position]

		# Coordenadas absolutas
		displacement = pred[ind_ensemble, i,:,:]
		this_pred_out_abs = displacement + np.array([data_test[i,:,:][-1].numpy()])

		mean = this_pred_out_abs[position, :]
		cov = np.array([[sx**2, 0],[0, sy**2]])

		if display:
			label4, = plt.plot(mean[0], mean[1], "*", color="red", label = "Means from Gaussian Mix")
			label1, label2, label3 = plot_traj_world(pred[ind_ensemble,i,:,:], data_test[i,:,:], target_test[i,:,:])

	param_gaussiana.append([mean,cov])

	# Construimos la gaussiana de la mezcla
	# Mezcla de gaussianas
	# https://faculty.ucmerced.edu/mcarreira-perpinan/papers/cs-99-03.pdf

	pi = np.ones((len(param_gaussiana),))/len(param_gaussiana)
	# Calculamos la media de mezcla
	mean_mix = np.zeros((2,))
	for j in range(len(param_gaussiana)):
		mean_mix += pi[j]*(param_gaussiana[j][0])

	# Calculamos la covarianza de la mezcla
	cov_mix = np.zeros((2,2))
	for j in range(len(param_gaussiana)):
		sub_mean = param_gaussiana[j][0].reshape(2,1) - mean_mix.reshape(2,1)
		mult_sub_mean = sub_mean @ sub_mean.T
		cov_mix +=  pi[j]*(param_gaussiana[j][1] + mult_sub_mean)


	sample_pdf = np.random.multivariate_normal(mean_mix, cov_mix, resample_size)
	# Create directory if does not exists
	output_dir = os.path.join(IMAGES_DIR, "trajectories")
	mkdir_p(output_dir)

	if display:
		label5, = plt.plot(sample_pdf[:,0], sample_pdf[:,1], ".", color="blue", alpha=0.2, label = "Gaussian Mix Samples")
		plt.title("Trajectory Plot")
		plt.legend(handles=[label1, label2, label3, label4, label5 ])
		plt.savefig(os.path.join(output_dir , "traj_samples_cov_"+str(idTest)+"_"+str(i)+".pdf"))
		#plt.show()
		plt.close()

	return multivariate_normal(mean_mix, cov_mix), sample_pdf

def get_kde(tpred_samples, data, target, i, sigmas_samples, position=0, idTest=0, gaussian=False, resample_size=1000):
	"""
	Args:
	Returns:
		- kde: PDF estimation
		- sample_kde: Sampled points (x,y) from PDF
	"""
	# Produce resample_size samples from the pdf
	if gaussian:
		# p.d.f. estimation. Sampling points (x,y) from PDF
		kde, sample_kde = gaussian_kde2(tpred_samples, sigmas_samples, data, target, i, position, resample_size=resample_size, display=False, idTest=idTest)
	else:
		# TODO: Here also, the coordinates may be absolute or relative
		# depending on the prediction method
		sample_kde = tpred_samples[:, i, position, :] + np.array([data[i,:,:][-1].numpy()])
		# Use KDE to get a representation of the p.d.f.
		# See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
		kde        = gaussian_kde(sample_kde.T)
		sample_kde = kde.resample(resample_size,0)

	return kde, sample_kde


def get_predicted_hdr(tpred_samples_cal, data_cal, target_cal, sigmas_samples_cal, position=0, idTest=0, gaussian=False, resample_size=1000):
	"""
	Args:
	Returns:
		- predicted HDR
	"""
	predicted_hdr = []
	# Traverse each trajectory of the batch
	for i in range(tpred_samples_cal.shape[1]):
		# Ground Truth
		gt = target_cal[i,position,:].cpu()
		kde, sample_kde = get_kde(tpred_samples_cal, data_cal, target_cal, i, sigmas_samples_cal, position=position, idTest=idTest, gaussian=gaussian, resample_size=resample_size)

		#----------------------------------------------------------
		# Evaluate these samples on the p.d.f.
		sample_pdf = kde.pdf(sample_kde)

		# Sort the samples in decreasing order of their p.d.f. value
		sample_pdf_zip = zip(sample_pdf, sample_pdf/np.sum(sample_pdf))
		orden          = sorted(sample_pdf_zip, key=lambda x: x[1], reverse=True)
		#----------------------------------------------------------

		# Ealuate the GT on the p.d.f.
		f_pdf = kde.pdf(gt)

		# Predicted HDR
		ind = np.where(np.array(orden)[:,0] >= f_pdf)[0]
		ind = 0 if ind.size == 0 else ind[-1] # Validate that it is not the first largest element
		alpha_pred = 1 - np.array(orden)[:ind+1,1].sum()
		predicted_hdr.append(alpha_pred)

	return predicted_hdr


def save_calibration_curves(output_calibration_dir, tpred_samples_test, conf_levels, unc_pcts, cal_pcts, unc_pcts2, cal_pcts2, gaussian=False, idTest=0, position=0):
	"""
	Save calibration curves
	"""
	# Create confidence level directory if does not exists
	output_confidence_dir = os.path.join(output_calibration_dir, "confidence_level")
	mkdir_p(output_confidence_dir)

	if gaussian:
		output_image_name = os.path.join(output_confidence_dir , "confidence_level_cal_IsotonicReg_"+str(idTest)+"_"+str(position)+"_gaussian.pdf")
		plot_calibration_curves(conf_levels, unc_pcts, cal_pcts, output_image_name)
	else:
		output_image_name = os.path.join(output_confidence_dir , "confidence_level_cal_IsotonicReg_"+str(idTest)+"_"+str(position)+".pdf")
		plot_calibration_curves(conf_levels, unc_pcts, cal_pcts, output_image_name)

	if tpred_samples_test is not None:
		if gaussian:
			output_image_name = os.path.join(output_confidence_dir , "confidence_level_test_IsotonicReg_"+str(idTest)+"_"+str(position)+"_gaussian.pdf")
			plot_calibration_curves(conf_levels, unc_pcts2, cal_pcts2, output_image_name)
		else:
			output_image_name = os.path.join(output_confidence_dir , "confidence_level_test_IsotonicReg_"+str(idTest)+"_"+str(position)+".pdf")
			plot_calibration_curves(conf_levels, unc_pcts2, cal_pcts2, output_image_name)


def get_fa(orden, alpha, ind_alpha):
	"""
	Args:
		- orden
		- alpha: alpha value to be used at elif
		- ind_alpha: alpha value to be used to compute index
	Returns:
		- fa obtained from PDF samples
	"""
	orden_i, orden_val = zip(*orden)
	ind = np.where(np.cumsum(orden_val) >= (1.0-ind_alpha))[0]
	if (ind.shape[0] == 0) :
		fa = 0.0
	elif (list(ind) == [len(orden)-1]) and (alpha==0.0):
		fa = 0.0
	else:
		fa = orden_i[ind[0]]
	return fa


def get_calibrated_uncalibrated_pcts(conf_levels, isotonic, tpred_samples, target, data, sigmas_samples, position=0, idTest=0, gaussian=False, resample_size=1000):
	"""
	Args:
	Returns:
		- calibrated percentages
		- uncalibrated percentages
	"""
	unc_pcts = []
	cal_pcts = []
	for alpha in conf_levels:
		new_alpha = isotonic.transform([alpha])
		print("alpha: ", alpha, " -- new_alpha: ", new_alpha)

		perc_within_cal = []
		perc_within_unc = []
		for i in range(tpred_samples.shape[1]):
			# Ground Truth
			gt = target[i,position,:].cpu()
			kde, sample_kde = get_kde(tpred_samples, data, target, i, sigmas_samples, position=position, idTest=idTest, gaussian=gaussian, resample_size=resample_size)

			#--------
			# Steps to compute HDRs fa
			# Evaluate these samples on the p.d.f.
			sample_pdf = kde.pdf(sample_kde)

			# Sort the samples in decreasing order of their p.d.f. value
			sample_pdf_zip = zip(sample_pdf, sample_pdf/np.sum(sample_pdf))
			orden = sorted(sample_pdf_zip, key=lambda x: x[1], reverse=True)

			fa = get_fa(orden, alpha, new_alpha)
			fa_unc = get_fa(orden, alpha, alpha)

			f_pdf = kde.pdf(gt)
			perc_within_cal.append(f_pdf >= fa)
			perc_within_unc.append(f_pdf >= fa_unc)
			#-----

		# Save batch results for an specific alpha
		cal_pcts.append(np.mean(perc_within_cal))
		unc_pcts.append(np.mean(perc_within_unc))

	return cal_pcts, unc_pcts


def calibration_IsotonicReg(tpred_samples_cal, data_cal, target_cal, sigmas_samples_cal, position = 0, idTest=0, gaussian=False, tpred_samples_test=None, data_test=None, target_test=None, sigmas_samples_test=None,resample_size=1000):

	predicted_hdr = get_predicted_hdr(tpred_samples_cal, data_cal, target_cal, sigmas_samples_cal, position=position, idTest=idTest, gaussian=gaussian, resample_size=resample_size)

	# Empirical HDR
	empirical_hdr = np.zeros(len(predicted_hdr))
	for i, p in enumerate(predicted_hdr):
		# TODO: check whether < or <=
		empirical_hdr[i] = np.sum(predicted_hdr <= p)/len(predicted_hdr)

	# Create calibration directory if does not exists
	output_calibration_dir = os.path.join(IMAGES_DIR, "calibration")
	mkdir_p(output_calibration_dir)

	#Visualization: Estimating HDR of Forecast
	output_image_name = os.path.join(output_calibration_dir , "plot_uncalibrate_"+str(idTest)+".pdf")
	title = "Estimating HDR of Forecast"
	plot_HDR_curves(predicted_hdr, empirical_hdr, output_image_name, title)

	#-----------------

	# Fit empirical_hdr to predicted_hdr with isotonic regression
	isotonic = IsotonicRegression(out_of_bounds='clip')
	isotonic.fit(empirical_hdr, predicted_hdr)

	# Visualization: Calibration with Isotonic Regression
	output_image_name = os.path.join(output_calibration_dir , "plot_calibrate_"+str(idTest)+".pdf")
	title = "Calibration with Isotonic Regression"
	plot_HDR_curves(predicted_hdr, isotonic.predict(empirical_hdr), output_image_name, title)

	#----------------

	conf_levels = np.arange(start=0.0, stop=1.025, step=0.05) # Valores de alpha

	cal_pcts, unc_pcts = get_calibrated_uncalibrated_pcts(conf_levels, isotonic, tpred_samples_cal, target_cal, data_cal, sigmas_samples_cal, position=position, idTest=idTest, gaussian=gaussian, resample_size=resample_size)
	unc_pcts2 = []
	cal_pcts2 = []

	if tpred_samples_test is not None:
		cal_pcts2, unc_pcts2 = get_calibrated_uncalibrated_pcts(conf_levels, isotonic, tpred_samples_test, target_test, data_test, sigmas_samples_test, position=position, idTest=idTest, gaussian=gaussian, resample_size=resample_size)

	save_calibration_curves(output_calibration_dir, tpred_samples_test, conf_levels, unc_pcts, cal_pcts, unc_pcts2, cal_pcts2, gaussian=gaussian, idTest=idTest, position=position)

	return 1-conf_levels, unc_pcts, cal_pcts, unc_pcts2, cal_pcts2, isotonic


def calibration_pdf22(tpred_samples, data_test, target_test, target_test2, sigmas_samples, position, alpha = 0.85, id_batch=-2, draw=False, gaussian=False):

	list_fk = []
	s_xk_yk = []
	# Creamos un KDE density con las muestras # tpred_samples[ind_ensemble, id_batch, position, xy]
	for k in range(tpred_samples.shape[1]): # Recorremos cada trayectoria del batch

		if gaussian:
			# Estimamos la pdf y muestreamos puntos (x,y) de la pdf
			fk, yi = gaussian_kde2(tpred_samples, sigmas_samples, data_test, target_test2, k, position, resample_size=1000, display=False, idTest=2)
		else:
			# Muestra de la distribución bayessiana
			yi = tpred_samples[:, k, position, :].T

			# Creamos la función de densidad con KDE, references: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
			fk = gaussian_kde(yi)
			yi = fk.resample(1000,0)

		# Evaluamos el GT
		if gaussian:
			gt = target_test2[k,position,:].detach().numpy()
			fk_yi = np.array([fk.pdf(gt)])

			# Guardamos en una lista
			s_xk_yk.append(np.array([fk_yi]))
		else:
			gt = target_test[k,position,:].detach().numpy()
			fk_yi = fk.pdf(gt)

			# Guardamos en una lista
			s_xk_yk.append(fk_yi)
		list_fk.append(fk)

	# Ordenamos las muestras
	orden = sorted(s_xk_yk, reverse=True) # Ordenamos
	#print(orden[:30])
	#print(type(orden))

	ind = int(len(orden)*alpha) # Encontramos el indice del alpha-esimo elemento
	#print(ind)
	if ind==len(orden):
		Sa = 0.0
	else:
		Sa = orden[ind][0] # tomamos el valor del alpha-esimo elemento mas grande
	#print("alpha: ", alpha)
	#print("ind: ", ind)
	#print("Sa encontrado: ", Sa)

	#--------------------------------------------------
	if draw:
		#-------------- para un id_batch ----------------------
		# Encontramos el alpha correspondiente al nuevo Sa en la funcion de densidad
		# Obtenemos la muestra de interes
		yi = tpred_samples[:, id_batch, position, :].T # Seleccionamos las muestras de una trayectoria
		gt = target_test[id_batch, position,:].detach().numpy()
		#print("yi: ", yi)
		#print("gt: ", gt)

		# Creamos la pdf para la muestra
		fk = gaussian_kde(yi.T)
		# Evaluamos la muestra en la pdf
		fk_yi = fk.pdf(yi.T) # Evaluamos en la funcion de densidad
		fk_gt = fk.pdf(gt.T)[0]
		#print("f(GT)", fk_gt)
		#print("f(GT) > Sa: ", fk_gt>Sa)

		# Ordenamos las muestras
		orden = sorted(fk_yi, reverse=True) # Ordenamos
		#print(orden)
		ind = np.where(np.array(orden) >= Sa)[0]
		ind = 0 if ind.size == 0 else ind[-1] # Validamos que no sea el primer elemento mas grande
		#print("ind_pdf: ", ind)
		alpha_fk = float(ind)/len(orden)
		#print("alpha_fk: ", alpha_fk)

		#--------------------------------------------------
		# Visualizamos la distribucion

		plt.figure()
		sns.kdeplot(x=yi[:,0], y=yi[:,1], label='KDE')
		sns.kdeplot(x=yi[:,0], y=yi[:,1], levels=[1-alpha], label=r'$\alpha$'+"=%.2f"%(alpha)) # Para colocar bien el Sa debemos usar el alpha
		sns.kdeplot(x=yi[:,0], y=yi[:,1], levels=[1-alpha_fk], label=r'$\alpha_{new}$'+"=%.2f"%(alpha_fk))
		plt.scatter(gt[0], gt[1], marker='^', color="blue", linewidth=3, label="GT")

		plt.legend()
		plt.xlabel('x-position')
		plt.ylabel('y-position')
		plt.title("Conformal Highest Density Regions with GT, S"+r'$_\alpha$'+"=%.2f"%(Sa)+", id_batch=" + str(id_batch))
		plt.savefig("images/HDR1/plot_hdr_%.2f_"%(alpha)+"_"+str(id_batch)+"_"+str(position)+"_gt.pdf")
		plt.close()

		# Visualizamos la distribucion sobre la trayectoria
		yi = tpred_samples[:, id_batch, position, :]  + data_test[id_batch,-1,:].numpy() # Seleccionamos las muestras de una trayectoria

		plt.figure()
		sns.kdeplot(x=yi[:,0], y=yi[:,1], label='KDE', fill=True, cmap="viridis_r", alpha=0.8)
		target_test_world = target_test[id_batch, :, :] + data_test[id_batch,-1,:].numpy()

		plt.plot(data_test[id_batch,:,0].numpy(),data_test[id_batch,:,1].numpy(),"-b", linewidth=2, label="Observations") # Observations
		plt.plot(target_test_world[:, 0], target_test_world[:, 1], '-*r', linewidth=2, label="Ground truth") # GT
		plt.legend()

		plt.legend()
		plt.xlabel('x-position')
		plt.ylabel('y-position')
		plt.title("Conformal Highest Density Regions with GT, S"+r'$_\alpha$'+"=%.2f"%(Sa)+", id_batch=" + str(id_batch))
		plt.savefig("images/trajectories_kde/trajectories_kde_%.2f_"%(alpha)+"_"+str(id_batch)+"_"+str(position)+".pdf")
		plt.close()
		#--------------------------------------------------


	return Sa

def calibration_pdf32(tpred_samples, data_test, target_test, target_test2, sigmas_samples, position, alpha = 0.85, id_batch=-2, draw=False, gaussian=False):

	list_fk = []
	s_xk_yk = []
	# Creamos un KDE density con las muestras # tpred_samples[ind_ensemble, id_batch, position, xy]
	for k in range(tpred_samples.shape[1]): # Recorremos cada trayectoria del batch

		if gaussian:
			# Estimamos la pdf y muestreamos puntos (x,y) de la pdf
			fk, yi = gaussian_kde2(tpred_samples, sigmas_samples, data_test, target_test2, k, position, resample_size=1000, display=False, idTest=2)
		else:
			# Muestra de la distribución bayessiana
			yi = tpred_samples[:, k, position, :].T

			# Creamos la función de densidad con KDE, references: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
			fk = gaussian_kde(yi)
			yi = fk.resample(1000,0)

		# Evaluamos el GT
		fk_max = fk.pdf(yi).max()
		if gaussian:
			# Evaluamos el GT
			gt = target_test2[k,position,:].detach().numpy()
			fk_yi = np.array([fk.pdf(gt)])

			# Guardamos en una lista
			s_xk_yk.append(np.array([fk_yi/fk_max]))
		else:
			# Evaluamos el GT
			gt = target_test[k,position,:].detach().numpy()
			fk_yi = fk.pdf(gt)

			# Guardamos en una lista
			s_xk_yk.append(fk_yi/fk_max)
		# Guardamos en una lista
		list_fk.append(fk)

	# Ordenamos las muestras
	orden = sorted(s_xk_yk, reverse=True) # Ordenamos
	#print(orden[:30])
	ind = int(len(orden)*alpha) # Encontramos el indice del alpha-esimo elemento
	if ind==len(orden):
		Sa = 0.0
	else:
		Sa = orden[ind][0] # tomamos el valor del alpha-esimo elemento mas grande
	#print("alpha: ", alpha)
	#print("ind: ", ind)
	#print("Sa encontrado: ", Sa)


	#--------------------------------------------------
	if draw:
		#-------------- para un id_batch ----------------------
		# Encontramos el alpha correspondiente al nuevo Sa en la funcion de densidad
		# Obtenemos la muestra de interes
		yi = tpred_samples[:, id_batch, position, :] # Seleccionamos las muestras de una trayectoria
		gt = target_test[id_batch, position,:].detach().numpy()

		# Creamos la pdf para la muestra
		fk = gaussian_kde(yi.T)
		# Evaluamos la muestra en la pdf
		fk_yi = fk.pdf(yi.T) # Evaluamos en la funcion de densidad
		fk_max = fk_yi.max()
		fk_gt = fk.pdf(gt.T)[0]
		#print("f(GT)", fk_gt)
		#print("f(GT) > Sa: ", fk_gt > (fk_max*Sa))

		# Ordenamos las muestras
		orden = sorted(fk_yi, reverse=True) # Ordenamos
		print(orden)
		ind = np.where(np.array(orden) >= (fk_max*Sa))[0]
		ind = 0 if ind.size == 0 else ind[-1] # Validamos que no sea el primer elemento mas grande
		#print("ind_pdf: ", ind)
		alpha_fk = float(ind)/len(orden)
		#print("alpha_fk: ", alpha_fk)

		#--------------------------------------------------
		# Visualizamos la distribucion

		plt.figure()
		sns.kdeplot(x=yi[:,0], y=yi[:,1], label='KDE')
		sns.kdeplot(x=yi[:,0], y=yi[:,1], levels=[1-alpha], label=r'$\alpha$'+"=%.2f"%(alpha)) # Para colocar bien el Sa debemos usar el alpha
		sns.kdeplot(x=yi[:,0], y=yi[:,1], levels=[1-alpha_fk], label=r'$\alpha_{new}$'+"=%.2f"%(alpha_fk))
		plt.scatter(gt[0], gt[1], marker='^', color="blue", linewidth=3, label="GT")

		plt.legend()
		plt.xlabel('x-position')
		plt.ylabel('y-position')
		plt.title("Conformal Highest Density Regions with GT, S"+r'$_\alpha$'+"=%.2f"%(Sa)+", id_batch=" + str(id_batch))
		plt.savefig("images/HDR2/plot_hdr_%.2f_"%(alpha)+"_"+str(id_batch)+"_"+str(position)+"_gt.pdf")
		plt.close()

	return Sa

# Given a value of alpha, deduce the value of the density
def get_falpha(orden, alpha):
	# We find f_gamma(HDR) from the pdf samples
	orden_idx, orden_val = zip(*orden)
	ind = np.where(np.cumsum(orden_val) >= alpha)[0]
	if ind.shape[0] == 0:
		fa = orden[-1][0]
	else:
		fa = orden_idx[ind[0]]
	return fa

def get_samples_pdfWeight(pdf,num_sample):
	# Muestreamos de la nueva función de densidad pesada
	return pdf.resample(num_sample)

def calibration_Conformal(tpred_samples_cal, data_cal, target_cal, target_cal2, sigmas_samples_cal, position = 0, idTest=0, method=2, gaussian=False, tpred_samples_test=None, data_test=None, target_test=None, target_test2=None, sigmas_samples_test=None):
	#----------------

	conf_levels = np.arange(start=0.0, stop=1.025, step=0.05) # Valores de alpha

	unc_pcts = []
	cal_pcts = []
	unc_pcts2 = []
	cal_pcts2 = []

	#for alpha in conf_level_lower_bounds:
	for alpha in conf_levels:

		print("***** alpha: ", alpha)
		# Obtenemos el fa con el metodo conformal
		if method==2:
			fa = calibration_pdf22(tpred_samples_cal, data_cal, target_cal, target_cal2, sigmas_samples_cal, position, alpha=alpha, gaussian=gaussian) # NOTA: Es unico para todo el dataset de calibracion
		elif method==3:
			fa = calibration_pdf32(tpred_samples_cal, data_cal, target_cal, target_cal2, sigmas_samples_cal, position, alpha=alpha, gaussian=gaussian) # NOTA: Es unico para todo el dataset de calibracion
		else:
			print("Método incorrecto, valores posibles 2 o 3.")
			return -1

		perc_within_cal = []
		perc_within_unc = []
		for i in range(tpred_samples_cal.shape[1]):

			if gaussian:
				# Estimamos la pdf y muestreamos puntos (x,y) de la pdf
				kde, sample_kde = gaussian_kde2(tpred_samples_cal, sigmas_samples_cal, data_cal, target_cal2, i, position, resample_size=1000, display=False, idTest=2)
			else:
				# Estimamos la pdf
				sample_kde = tpred_samples_cal[:, i, position, :].T # Seleccionamos las muestras de una trayectoria
				# Creamos la pdf para la muestra
				kde = gaussian_kde(sample_kde)
				sample_kde = kde.resample(1000,0)

			#--------
			# Pasos para calcular fa del HDR

			# Evaluamos la muestra en la pdf
			sample_pdf = kde.pdf(sample_kde)

			# Ordenamos las muestras
			orden = sorted(sample_pdf, reverse=True) # Ordenamos
			##print(orden[:30])
			ind = int(len(orden)*alpha) # Encontramos el indice del alpha-esimo elemento
			if ind==len(orden):
				fa_unc = 0.0
			else:
				fa_unc = orden[ind] # tomamos el valor del alpha-esimo elemento mas grande
			##print("alpha: ", alpha)
			##print("ind: ", ind)
			##print("fa_unc encontrado: ", fa_unc)

			if gaussian:
				# Ground Truth
				gt = target_cal2[i,position,:].cpu()
			else:
				# Ground Truth
				gt = target_cal[i,position,:].cpu()
			# Evaluamos el Ground truth
			f_pdf = kde.pdf(gt)

			if method==2:
				perc_within_cal.append(f_pdf >= fa)
			elif method==3:
				perc_within_cal.append(f_pdf >= sample_pdf.max()*fa)

			perc_within_unc.append(f_pdf >= fa_unc)
			#-----

		# Guardamos los resultados de todo el batch para un alpha especifico
		cal_pcts.append(np.mean(perc_within_cal))
		unc_pcts.append(np.mean(perc_within_unc))

		if tpred_samples_test is not None:
			print("-- procesamos el datatest...")
			##print(tpred_samples_test.shape)
			##print(data_test.shape)
			##print(target_test.shape)
			#aaaa

			perc_within_cal = []
			perc_within_unc = []
			ll = 0.0
			for i in range(tpred_samples_test.shape[1]):

				if gaussian:
					# Estimamos la pdf y muestreamos puntos (x,y) de la pdf
					kde, sample_kde = gaussian_kde2(tpred_samples_test, sigmas_samples_test, data_test, target_test2, i, position, resample_size=1000, display=False, idTest=2)
				else:
					# Estimamos la pdf
					sample_kde = tpred_samples_test[:, i, position, :].T # Seleccionamos las muestras de una trayectoria
					# Creamos la pdf para la muestra
					kde = gaussian_kde(sample_kde)
					sample_kde = kde.resample(1000,0)

				#--------
				# Pasos para calcular fa del HDR

				# Evaluamos la muestra en la pdf
				sample_pdf = kde.pdf(sample_kde)

				# Ordenamos las muestras
				orden = sorted(sample_pdf, reverse=True) # Ordenamos
				##print(orden[:30])
				ind = int(len(orden)*alpha) # Encontramos el indice del alpha-esimo elemento
				if ind==len(orden):
					fa_unc = 0
				else:
					fa_unc = orden[ind] # tomamos el valor del alpha-esimo elemento mas grande

				if gaussian:
					gt = target_test2[i,position,:].cpu()
				else:
					# Ground Truth
					gt = target_test[i,position,:].cpu()
				# Evaluamos el Ground truth
				f_pdf = kde.pdf(gt)

				if method==2:
					perc_within_cal.append(f_pdf >= fa)
				elif method==3:
					perc_within_cal.append(f_pdf >= sample_pdf.max()*fa)

				perc_within_unc.append(f_pdf >= fa_unc)
				#-----

			# Guardamos los resultados de todo el batch para un alpha especifico
			cal_pcts2.append(np.mean(perc_within_cal))
			unc_pcts2.append(np.mean(perc_within_unc))





	plt.figure(figsize=(10,7))
	plt.plot([0,1],[0,1],'--', color='grey')
	plt.plot(conf_levels, unc_pcts, '-o', color='purple', label='Uncalibrated')
	plt.plot(conf_levels, cal_pcts, '-o', color='red', label='Calibrated')
	plt.legend(fontsize=14)
	#plt.title('Calibration Plot on Calibration Data ('+str(idTest)+')', fontsize=17)
	plt.xlabel(r'$\alpha$', fontsize=17)
	plt.ylabel(r'$\hat{P}_\alpha$', fontsize=17)

	# Create confidence level directory if does not exists
	output_confidence_dir = os.path.join(IMAGES_DIR, "calibration", "confidence_level")
	mkdir_p(output_confidence_dir)

	output_image_name = os.path.join(output_confidence_dir , "confidence_level_cal_"+str(idTest)+"_conformal"+str(method)+"_"+str(position)+".pdf")
	plt.savefig(output_image_name)
	plt.show()

	if tpred_samples_test is not None:
		plt.figure(figsize=(10,7))
		plt.plot([0,1],[0,1],'--', color='grey')
		plt.plot(conf_levels, unc_pcts2, '-o', color='purple', label='Uncalibrated')
		plt.plot(conf_levels, cal_pcts2, '-o', color='red', label='Calibrated')
		plt.legend(fontsize=14)
		#plt.title('Calibration Plot on Test Data ('+str(idTest)+')', fontsize=17)
		plt.xlabel(r'$\alpha$', fontsize=17)
		plt.ylabel(r'$\hat{P}_\alpha$', fontsize=17)

		output_image_name = os.path.join(output_confidence_dir , "confidence_level_test_"+str(idTest)+"_conformal"+str(method)+"_"+str(position)+".pdf")
		plt.savefig(output_image_name)
		plt.show()
	#----------------------------------------------------------------------
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
	print(key + ":  ", end="")
	print("MACE: {:.5f}, RMSCE: {:.5f}, MA: {:.5f}".format(mace,rmsce,ma))

def generate_metrics_calibration_IsotonicReg(tpred_samples_cal, data_cal, target_cal, sigmas_samples_cal, id_test, gaussian=False, tpred_samples_test=None, data_test=None, target_test=None, sigmas_samples_test=None):

	#------------Calibration metrics-------------------
	metrics_calibration_data = [["","MACE","RMSCE","MA"]]
	metrics_test_data        = [["","MACE","RMSCE","MA"]]
	key_before = "Before Recalibration"
	key_after = "After Recalibration"

	# Recorremos cada posicion
	positions_to_test = [11]
	for position in positions_to_test:
		logging.info("Calibration metrics at position: {}".format(position))
		# Apply isotonic regression
		exp_proportions, obs_proportions_unc, obs_proportions_cal, obs_proportions_unc2, obs_proportions_cal2 , isotonic = calibration_IsotonicReg(tpred_samples_cal, data_cal, target_cal, sigmas_samples_cal, position = position, idTest=id_test, gaussian=gaussian, tpred_samples_test=tpred_samples_test, data_test=data_test, target_test=target_test, sigmas_samples_test=sigmas_samples_test)

		# TODO: move?
		plt.show()

		# Calibration metrics
		compute_calibration_metrics(exp_proportions, obs_proportions_unc, metrics_calibration_data, position, key_before)

		compute_calibration_metrics(exp_proportions, obs_proportions_cal, metrics_calibration_data, position, key_after)


		if tpred_samples_test is not None:
			# Metrics Calibration on testing data
			compute_calibration_metrics(exp_proportions, obs_proportions_unc2, metrics_test_data, position, key_before)

			compute_calibration_metrics(exp_proportions, obs_proportions_cal2, metrics_test_data, position, key_after)

		break

	# Save the metrics results: on calibration dataset
	df = pd.DataFrame(metrics_calibration_data)

	# Create metrics directory if does not exists
	output_metrics_dir = os.path.join(IMAGES_DIR, "calibration", "metrics")
	mkdir_p(output_metrics_dir)

	output_csv_name = os.path.join(output_metrics_dir, "metrics_calibration_cal_IsotonicRegresion_"+str(id_test)+".csv")
	df.to_csv(output_csv_name)

	if tpred_samples_test is not None:
		# Save the metrics results: on test dataset
		df = pd.DataFrame(metrics_test_data)
		output_csv_name = os.path.join(output_metrics_dir, "metrics_calibration_test_IsotonicRegresion_"+str(id_test)+".csv")
		df.to_csv(output_csv_name)

	# Evaluation of NLL
	position = 11
	ll_cal = []
	ll_uncal = []

	for i in range(tpred_samples_test.shape[1]):
		# Ground Truth
		gt = target_test[i,position,:].cpu()

		# TODO: here, it depends on the predition system whether the output is
		# TODO: relative or absolute
		this_pred_out_abs = tpred_samples_test[:, i, position, :] + np.array([data_test[i,:,:][-1].numpy()]) # ABSOLUTE?
		if gaussian:
			# Estimamos la pdf y muestreamos puntos (x,y) de la pdf
			kde, sample_kde = gaussian_kde2(tpred_samples_test, sigmas_samples_test, data_test, target_test, i, position, resample_size=1000, display=False, idTest=id_test)
		else:
			sample_kde = this_pred_out_abs.T
			kde = gaussian_kde(sample_kde)
			sample_kde = kde.resample(1000,0)

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
	output_csv_name = os.path.join(output_metrics_dir, "nll_IsotonicRegresion_"+str(id_test)+".csv")
	df.to_csv(output_csv_name)
	print(df)



def generate_metrics_calibration_conformal(tpred_samples_cal, data_cal, targetrel_cal, target_cal, sigmas_samples_cal, id_test, gaussian=False, tpred_samples_test=None, data_test=None, targetrel_test=None, target_test=None, sigmas_samples_test=None):
	#--------------------- Calculamos las metricas de calibracion ---------------------------------
	metrics2 = [["","MACE","RMSCE","MA"]]
	metrics3 = [["","MACE","RMSCE","MA"]]
	metrics2_test = [["","MACE","RMSCE","MA"]]
	metrics3_test = [["","MACE","RMSCE","MA"]]
	key_before = "Before Recalibration"
	key_after = "After Recalibration"
	# Recorremos cada posicion para calibrar
	for pos in range(tpred_samples_cal.shape[2]):
		pos = 11
		print("--------------------------------")
		print("Procesamos para posicion: ", pos)
		gt = np.cumsum(targetrel_cal, axis=1)
		gt_test = np.cumsum(targetrel_test, axis=1)
		# HDR y Calibracion
		print("------- calibration_pdf2 ")
		exp_proportions, obs_proportions_unc, obs_proportions_cal, obs_proportions_unc2, obs_proportions_cal2 = calibration_Conformal(tpred_samples_cal, data_cal, gt, target_cal, sigmas_samples_cal, position = pos, idTest=id_test, method=2, gaussian=gaussian, tpred_samples_test=tpred_samples_test, data_test=data_test, target_test=gt_test, target_test2=target_test, sigmas_samples_test=sigmas_samples_test)

		# Metrics Calibration
		compute_calibration_metrics(exp_proportions, obs_proportions_unc, metrics2, pos, key_before)

		compute_calibration_metrics(exp_proportions, obs_proportions_cal, metrics2, pos, key_after)

		if tpred_samples_test is not None:
			# Metrics Calibration Test
			compute_calibration_metrics(exp_proportions, obs_proportions_unc2, metrics2_test, pos, key_before)

			compute_calibration_metrics(exp_proportions, obs_proportions_cal2, metrics2_test, pos, key_after)

		print("------- calibration_pdf3 ")
		exp_proportions, obs_proportions_unc, obs_proportions_cal, obs_proportions_unc2, obs_proportions_cal2 = calibration_Conformal(tpred_samples_cal, data_cal, gt, target_cal, sigmas_samples_cal, position = pos, idTest=id_test, method=3, gaussian=gaussian, tpred_samples_test=tpred_samples_test, data_test=data_test, target_test=gt_test, target_test2=target_test, sigmas_samples_test=sigmas_samples_test)

		# Metrics Calibration
		compute_calibration_metrics(exp_proportions, obs_proportions_unc, metrics3, pos, key_before)

		compute_calibration_metrics(exp_proportions, obs_proportions_cal, metrics3, pos, key_after)

		if tpred_samples_test is not None:
			# Metrics Calibration Test
			compute_calibration_metrics(exp_proportions, obs_proportions_unc2, metrics3_test, pos, key_before)

			compute_calibration_metrics(exp_proportions, obs_proportions_cal2, metrics3_test, pos, key_after)

		break

	# Create metrics directory if does not exists
	output_metrics_dir = os.path.join(IMAGES_DIR, "calibration", "metrics")
	mkdir_p(output_metrics_dir)

	# Guardamos los resultados de las metricas
	df = pd.DataFrame(metrics2)
	output_csv_name = os.path.join(output_metrics_dir, "metrics_calibration_cal_conformal2_"+str(id_test)+".csv")
	df.to_csv(output_csv_name)

	df = pd.DataFrame(metrics3)
	output_csv_name = os.path.join(output_metrics_dir, "metrics_calibration_cal_conformal3_"+str(id_test)+".csv")
	df.to_csv(output_csv_name)

	if tpred_samples_test is not None:
		# Guardamos los resultados de las metricas de Test
		df = pd.DataFrame(metrics2_test)
		output_csv_name = os.path.join(output_metrics_dir, "metrics_calibration_test_conformal2_"+str(id_test)+".csv")
		df.to_csv(output_csv_name)

		df = pd.DataFrame(metrics3_test)
		output_csv_name = os.path.join(output_metrics_dir, "metrics_calibration_test_conformal3_"+str(id_test)+".csv")
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
		print("***** alpha: ", alpha)
		# Obtenemos el fa con el metodo conformal
		if method==2:
			Sa = calibration_pdf22(tpred_samples, data_test, targetrel_test, None, None, position, alpha=alpha) # NOTA: Es unico para todo el dataset de calibracion
		elif method==3:
			Sa = calibration_pdf32(tpred_samples, data_test, targetrel_test, None, None, position, alpha=alpha) # NOTA: Es unico para todo el dataset de calibracion
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



def generate_one_batch_test(batched_test_data, model, num_samples, TRAINING_CKPT_DIR, model_name, id_test=2, device=None, dim_pred=12, type="ensemble"):
	#----------- Dataset TEST -------------
	datarel_test_full = []
	targetrel_test_full = []
	data_test_full = []
	target_test_full = []

	for batch_idx, (datarel_test, targetrel_test, data_test, target_test) in enumerate(batched_test_data):
		if batch_idx==0:
			continue

		 # Batchs saved into array respectively
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
			model.load_state_dict(torch.load(TRAINING_CKPT_DIR+"/"+model_name+"_"+str(ind)+"_"+str(id_test)+".pth"))
			model.eval()

		if torch.cuda.is_available():
			datarel_test_full  = datarel_test_full.to(device)

		# Model prediction obtained
		pred, sigmas = model.predict(datarel_test_full, dim_pred=12)

		# Sample saved
		tpred_samples_full.append(pred)
		sigmas_samples_full.append(sigmas)

	tpred_samples_full = np.array(tpred_samples_full)
	sigmas_samples_full = np.array(sigmas_samples_full)

	return datarel_test_full, targetrel_test_full, data_test_full, target_test_full, tpred_samples_full, sigmas_samples_full
