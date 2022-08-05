import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import seaborn as sns
import statistics

import torch
from scipy.stats import gaussian_kde
#from sklearn.neighbors import KernelDensity

from scipy.stats import multivariate_normal
from sklearn.metrics import auc
from sklearn.isotonic import IsotonicRegression

from shapely.geometry import Polygon, LineString
from shapely.ops import polygonize, unary_union

from utils.plot_utils import plot_traj_world, plot_traj_img_kde
# Local utils helpers
from utils.directory_utils import mkdir_p
# Local constants
from utils.constants import IMAGES_DIR


def gaussian_kde2(pred, sigmas_samples, data_test, target_test, i, position, resample_size=0 , display=False, idTest=0):

    # Estimamos la gaussiana con los parametros que salen del modelo
    param_gaussiana = []
    if display:
        plt.figure(figsize=(12,10))

    for ind_ensemble in range(sigmas_samples.shape[0]):
        # Procesamos las medias y sigmas [2, 16, 12, 3]
        # Extraemos los valores para la covarianza
        #cum_sigmas_samples = np.cumsum(sigmas_samples[ind_ensemble, i,:,:], axis=0)
        sigmas_samples_ensemble = sigmas_samples[ind_ensemble, i,:,:]
        #sx, sy, cor = sigmas_samples[ind_ensemble, i, position, 0], sigmas_samples[ind_ensemble, i, position, 1], sigmas_samples[ind_ensemble, i, position, 2]
        #sx, sy, cor = cum_sigmas_samples[position, 0], cum_sigmas_samples[position, 1], cum_sigmas_samples[position, 2]
        sx, sy, cor = sigmas_samples_ensemble[:, 0], sigmas_samples_ensemble[:, 1], sigmas_samples_ensemble[:, 2]

        # Exponential to get a positive value for std dev
        #sx = np.exp(sx)
        #sy = np.exp(sy)
        #sx   = np.exp(sx)+1e-2
        #sy   = np.exp(sy)+1e-2
        #sx   = np.cumsum(sx)[position]
        #sy   = np.cumsum(sy)[position]
        sx   = sx[position]
        sy   = sy[position]

        # tanh to get a value between [-1, 1] for correlation
        #cor = np.tanh(cor)

        # Coordenadas absolutas
        #displacement = np.cumsum(pred[ind_ensemble, i,:,:], axis=0)
        displacement = pred[ind_ensemble, i,:,:]
        this_pred_out_abs = displacement + np.array([data_test[i,:,:][-1].numpy()])

        mean = this_pred_out_abs[position, :]
        #cov = np.array([[sx**2, cor*sx*sy],
        #                [cor*sx*sy, sy**2]])
        cov = np.array([[sx**2, 0],
                        [0, sy**2]])

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


def calibration_IsotonicReg(tpred_samples_cal, data_cal, target_cal, sigmas_samples_cal, position = 0, idTest=0, gaussian=False, tpred_samples_test=None, data_test=None, target_test=None, sigmas_samples_test=None):

    predicted_hdr = []
    # Recorremos cada trayectoria del batch
    for i in range(tpred_samples_cal.shape[1]):
        # Ground Truth
        gt = target_cal[i,position,:].cpu()
        #print("GT: ", gt)

        if gaussian:
            # Estimamos la pdf y muestreamos puntos (x,y) de la pdf
            kde, sample_kde = gaussian_kde2(tpred_samples_cal, sigmas_samples_cal, data_cal, target_cal, i, position, resample_size=1000, display=False, idTest=idTest)
        else:
            # Muestra de la distribución bayessiana
            this_pred_out_abs = tpred_samples_cal[:, i, position, :] + np.array([data_cal[i,:,:][-1].numpy()])
            sample_kde = this_pred_out_abs.T
            # Creamos la función de densidad con KDE, references: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
            kde = gaussian_kde(sample_kde)
            sample_kde = kde.resample(1000,0)
            print("NLL: ", kde.pdf(gt))

        #----------------------------------------------------------
        # Pasos para calcular fa del HDR

        # Evaluamos la muestra en la pdf
        sample_pdf = kde.pdf(sample_kde)

        # Ordenamos de forma descendente las muestras de pdf
        sample_pdf_zip = zip(sample_pdf, sample_pdf/np.sum(sample_pdf))
        orden = sorted(sample_pdf_zip, key=lambda x: x[1], reverse=True)
        #----------------------------------------------------------


        # Evaluamos el Ground Truth (ultima posicion) en la distribucion
        f_pdf = kde.pdf(gt)

        # Predicted HDR
        ind = np.where(np.array(orden)[:,0] >= f_pdf)[0]
        ind = 0 if ind.size == 0 else ind[-1] # Validamos que no sea el primer elemento mas grande
        alpha_pred = 1 - np.array(orden)[:ind+1,1].sum()
        predicted_hdr.append(alpha_pred)

    # Empirical HDR
    empirical_hdr = np.zeros(len(predicted_hdr))
    for i, p in enumerate(predicted_hdr):
        empirical_hdr[i] = np.sum(predicted_hdr <= p)/len(predicted_hdr) # En este paso, p toma los valores de alpha, por lo que no estoy seguro  si debe ser menor e igual o mayor e igual.

    #Visualization
    plt.figure(figsize=(10,7))
    plt.scatter(predicted_hdr, empirical_hdr, alpha=0.7)
    plt.plot([0,1],[0,1],'--', color='grey', label='Perfect calibration')
    plt.xlabel('Predicted HDR', fontsize=17)
    plt.ylabel('Empirical HDR', fontsize=17)
    plt.title('Estimating HDR of Forecast', fontsize=17)
    plt.legend(fontsize=17)
    plt.grid("on")

    plt.savefig("images/calibration/plot_uncalibrate_"+str(idTest)+".pdf")
    plt.show()

    #-----------------

    # fit the isotonic regression
    isotonic = IsotonicRegression(out_of_bounds='clip')
    isotonic.fit(empirical_hdr, predicted_hdr)

    #Visualization
    plt.figure(figsize=(10,7))
    plt.scatter(predicted_hdr, isotonic.predict(empirical_hdr), alpha=0.7)
    plt.plot([0,1],[0,1],'--', color='grey', label='Perfect calibration')
    plt.xlabel('Predicted HDR', fontsize=17)
    plt.ylabel('Empirical HDR', fontsize=17)
    plt.title('Calibration with Isotonic Regression', fontsize=17)
    plt.legend(fontsize=17)
    plt.grid("on")

    plt.savefig("images/calibration/plot_calibrate_"+str(idTest)+".pdf")
    plt.show()

    #----------------

    conf_levels = np.arange(start=0.0, stop=1.025, step=0.05) # Valores de alpha

    unc_pcts = []
    cal_pcts = []
    unc_pcts2 = []
    cal_pcts2 = []

    for alpha in conf_levels:
        new_alpha = isotonic.transform([alpha])
        print("alpha: ", alpha, " -- new_alpha: ", new_alpha)
        #print("hdr: ", 1-alpha, " -- new_hdr: ", 1-new_alpha)

        perc_within_cal = []
        perc_within_unc = []
        for i in range(tpred_samples_cal.shape[1]):
            # Ground Truth
            gt = target_cal[i,position,:].cpu()

            if gaussian:
                # Estimamos la pdf
                kde, sample_kde = gaussian_kde2(tpred_samples_cal, sigmas_samples_cal, data_cal, target_cal, i, position, resample_size=1000, idTest=idTest)
            else:
                # Muestra de la distribución bayessiana
                this_pred_out_abs = tpred_samples_cal[:, i, position, :] + np.array([data_cal[i,:,:][-1].numpy()])
                sample_kde = this_pred_out_abs.T
                # Creamos la función de densidad con KDE, references: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
                kde = gaussian_kde(sample_kde)
                sample_kde = kde.resample(1000,0)

            #--------
            # Pasos para calcular fa del HDR

            # Evaluamos la muestra en la pdf
            sample_pdf = kde.pdf(sample_kde)

            # Ordenamos de forma descendente las muestras de pdf
            sample_pdf_zip = zip(sample_pdf, sample_pdf/np.sum(sample_pdf))
            orden = sorted(sample_pdf_zip, key=lambda x: x[1], reverse=True)

            # Encontramos fa a partir de las muestras de pdf
            orden_i, orden_val = zip(*orden)
#            print(np.cumsum(orden_val) >= (1.0-new_alpha))
#            print(np.where(np.cumsum(orden_val) >= (1.0-new_alpha)))
#            print(np.where(np.cumsum(orden_val) >= (1.0-new_alpha))[0])
            ind = np.where(np.cumsum(orden_val) >= (1.0-new_alpha))[0]
            if (ind.shape[0] == 0) :
#                print("ind1: ", -1)
                #fa = orden[-1][0]
                fa = 0.0
            elif (list(ind) == [len(orden)-1]) and (alpha==0.0):
                fa = 0.0
            else:
#                print("ind1: ", ind)
                fa = orden_i[ind[0]]
            #sum = 0
            #fa = orden[-1][0]
            #for ii, val in enumerate(orden):
            #    sum += val[1]
            #    #if sum >= new_alpha:
            #    if sum >= (1.0-new_alpha):
            #        fa = val[0]
            #        break

            # Encontramos fa a partir de las muestras de pdf
            orden_i, orden_val = zip(*orden)
            #print(orden_val[::-1][:10])
            #print(np.cumsum(orden_val)[::-1][:10])
            #print("1.0-alpha: ", 1.0-alpha)
            ind = np.where(np.cumsum(orden_val) >= (1.0-alpha))[0]
            if (ind.shape[0] == 0) :
                #print("ind: ", ind)
                #fa_unc = orden[-1][0]
                fa_unc = 0.0
            elif (list(ind) == [len(orden)-1]) and (alpha==0.0):
                #print("ind: ", ind)
                fa_unc = 0.0
            else:
                #print("ind: ", ind[0])
                fa_unc = orden_i[ind[0]]
            #print("fa_unc: ", fa_unc)
            #sum = 0
            #fa_unc = orden[-1][0]
            #for ii, val in enumerate(orden):
            #    sum += val[1]
            #    if sum >= (1.0-alpha):
            #        fa_unc = val[0]
            #        break

            f_pdf = kde.pdf(gt)
            #print("gt_pdf: ", f_pdf)
            #print("f_pdf >= fa_unc: ", f_pdf >= fa_unc)
            perc_within_cal.append(f_pdf >= fa)
            perc_within_unc.append(f_pdf >= fa_unc)
            #-----

        # Guardamos los resultados de todo el batch para un alpha especifico
        #print(perc_within_unc)
        #print(np.mean(perc_within_unc))
        cal_pcts.append(np.mean(perc_within_cal))
        unc_pcts.append(np.mean(perc_within_unc))

    if tpred_samples_test is not None:
        for alpha in conf_levels:
            new_alpha = isotonic.transform([alpha])

            perc_within_cal = []
            perc_within_unc = []
            for i in range(tpred_samples_test.shape[1]):
                #i = 1
                # Ground Truth
                gt = target_test[i,position,:].cpu()

                if gaussian:
                    # Estimamos la pdf
                    kde, sample_kde = gaussian_kde2(tpred_samples_test, sigmas_samples_test, data_test, target_test, i, position, resample_size=1000, idTest=idTest)
                else:
                    # Muestra de la distribución bayessiana
                    this_pred_out_abs = tpred_samples_test[:, i, position, :] + np.array([data_test[i,:,:][-1].numpy()])
                    sample_kde = this_pred_out_abs.T
                    # Creamos la función de densidad con KDE, references: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
                    kde = gaussian_kde(sample_kde)
                    sample_kde = kde.resample(1000,0)

                #--------
                # Pasos para calcular fa del HDR

                # Evaluamos la muestra en la pdf
                sample_pdf = kde.pdf(sample_kde)

                # Ordenamos de forma descendente las muestras de pdf
                sample_pdf_zip = zip(sample_pdf, sample_pdf/np.sum(sample_pdf))
                orden = sorted(sample_pdf_zip, key=lambda x: x[1], reverse=True)

                # Encontramos fa a partir de las muestras de pdf
                orden_i, orden_val = zip(*orden)
                ind = np.where(np.cumsum(orden_val) >= (1.0-new_alpha))[0]
    #            print(ind)
                if (ind.shape[0] == 0) :
    #                print("ind1: ", -1)
                    #fa = orden[-1][0]
                    fa = 0.0
                elif (list(ind) == [len(orden)-1]) and (alpha==0.0):
                    fa = 0.0
                else:
    #                print("ind1: ", ind)
                    fa = orden_i[ind[0]]
    #            print(fa)

                # Encontramos fa a partir de las muestras de pdf
                orden_i, orden_val = zip(*orden)
                ind = np.where(np.cumsum(orden_val) >= (1.0-alpha))[0]
    #            print(ind)
    #            print(ind.shape)
                if (ind.shape[0] == 0):
                    #fa_unc = orden[-1][0]
                    fa_unc = 0.0
                elif (list(ind) == [len(orden)-1]) and (alpha==0.0):
                    fa_unc = 0.0
                else:
                    fa_unc = orden_i[ind[0]]

                f_pdf = kde.pdf(gt)
                perc_within_cal.append(f_pdf >= fa)
                perc_within_unc.append(f_pdf >= fa_unc)
                #-----

            # Guardamos los resultados de todo el batch para un alpha especifico
            cal_pcts2.append(np.mean(perc_within_cal))
            unc_pcts2.append(np.mean(perc_within_unc))


    #--------------------- Guardamos las curvas de calibracion ---------------------------

    plt.figure(figsize=(10,7))
    plt.plot([0,1],[0,1],'--', color='grey')
    plt.plot(1-conf_levels, unc_pcts, '-o', color='purple', label='Uncalibrated')
    plt.plot(1-conf_levels, cal_pcts, '-o', color='red', label='Calibrated')
    plt.legend(fontsize=14)
    #plt.title('Calibration Plot on Testing Data ('+str(idTest)+')', fontsize=17)
    plt.xlabel(r'$\alpha$', fontsize=17)
    plt.ylabel(r'$\hat{P}_\alpha$', fontsize=17)

    if gaussian:
        plt.savefig("images/calibration/confidence_level/confidence_level_cal_IsotonicReg_"+str(idTest)+"_"+str(position)+"_gaussian.pdf")
    else:
        plt.savefig("images/calibration/confidence_level/confidence_level_cal_IsotonicReg_"+str(idTest)+"_"+str(position)+".pdf")
    plt.show()

    if tpred_samples_test is not None:
        plt.figure(figsize=(10,7))
        plt.plot([0,1],[0,1],'--', color='grey')
        plt.plot(1-conf_levels, unc_pcts2, '-o', color='purple', label='Uncalibrated')
        plt.plot(1-conf_levels, cal_pcts2, '-o', color='red', label='Calibrated')
        plt.legend(fontsize=14)
        #plt.title('Calibration Plot on Testing Data ('+str(idTest)+')', fontsize=17)
        plt.xlabel(r'$\alpha$', fontsize=17)
        plt.ylabel(r'$\hat{P}_\alpha$', fontsize=17)

        if gaussian:
            plt.savefig("images/calibration/confidence_level/confidence_level_test_IsotonicReg_"+str(idTest)+"_"+str(position)+"_gaussian.pdf")
        else:
            plt.savefig("images/calibration/confidence_level/confidence_level_test_IsotonicReg_"+str(idTest)+"_"+str(position)+".pdf")
        plt.show()

    #----------------------------------------------------------------------
    return 1-conf_levels, unc_pcts, cal_pcts, unc_pcts2, cal_pcts2, isotonic

def calibration_IsotonicReg2(tpred_samples_cal, data_cal, target_cal, sigmas_samples_cal, position = 0, idTest=0, gaussian=False, tpred_samples_test=None, data_test=None, target_test=None, sigmas_samples_test=None):

    predicted_hdr = []
    # Recorremos cada trayectoria del batch
    for i in range(tpred_samples_cal.shape[1]):
        # Ground Truth
        gt = target_cal[i,position,:].cpu()
        print("GT: ", gt)

        if gaussian:
            # Estimamos la pdf y muestreamos puntos (x,y) de la pdf
            kde, sample_kde = gaussian_kde2(tpred_samples_cal, sigmas_samples_cal, data_cal, target_cal, i, position, resample_size=1000, display=False, idTest=idTest)
        else:
            # Muestra de la distribución bayessiana
            this_pred_out_abs = tpred_samples_cal[:, i, position, :] + np.array([data_cal[i,:,:][-1].numpy()])
            sample_kde = this_pred_out_abs.T
            # Creamos la función de densidad con KDE, references: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
            kde = gaussian_kde(sample_kde)
            sample_kde = kde.resample(1000,0)
            #print("deterministic: ", kde.pdf(gt))

        #----------------------------------------------------------
        # Pasos para calcular fa del HDR

        # Evaluamos la muestra en la pdf
        sample_pdf = kde.pdf(sample_kde)
        print(sample_kde)

        # Ordenamos de forma descendente las muestras de pdf
        #sample_pdf_zip = zip(sample_pdf, sample_pdf/np.sum(sample_pdf))
        #orden = sorted(sample_pdf_zip, key=lambda x: x[1], reverse=True)
        #----------------------------------------------------------


        # Evaluamos el Ground Truth (ultima posicion) en la distribucion
        f_pdf = kde.pdf(gt)

        # Predicted HDR
        #ind = np.where(np.array(orden)[:,0] >= f_pdf)[0]
        #ind = 0 if ind.size == 0 else ind[-1] # Validamos que no sea el primer elemento mas grande
        #alpha_pred = 1 - np.array(orden)[:ind+1,1].sum()
        orden = sorted(sample_pdf, reverse=True) # Ordenamos
        ind = np.where(np.array(orden) >= f_pdf)[0]
        ind = -1 if ind.size == 0 else ind[-1] # Validamos que no sea el primer elemento mas grande
        #if ind == (len(orden)-1):
        #    alpha_pred = 0
        #else:
        #alpha_pred = 1 - (float(ind+1)/len(orden))
        alpha_pred = 1 - (float(ind)/len(orden))
        print(ind, f_pdf, orden[0], orden[-1], alpha_pred)

        predicted_hdr.append(alpha_pred)


    print(predicted_hdr)
    print(np.unique(predicted_hdr, return_counts=True))
    #print(sorted(list(set(predicted_hdr))))
    predicted_hdr = np.array(predicted_hdr)
    print(len(predicted_hdr))
    #aaaa
    # Empirical HDR
    empirical_hdr = np.zeros(len(predicted_hdr))
    for i, p in enumerate(predicted_hdr):
        empirical_hdr[i] = np.sum(predicted_hdr <= p)/len(predicted_hdr) # En este paso, p toma los valores de alpha, por lo que no estoy seguro  si debe ser menor e igual o mayor e igual.
        print(p, empirical_hdr[i], np.sum(predicted_hdr <= p))

    #Visualization
    plt.figure(figsize=(10,7))
    plt.scatter(predicted_hdr, empirical_hdr, alpha=0.7)
    plt.plot([0,1],[0,1],'--', color='grey', label='Perfect calibration')
    plt.xlabel('Predicted HDR', fontsize=17)
    plt.ylabel('Empirical HDR', fontsize=17)
    plt.title('Estimating HDR of Forecast', fontsize=17)
    plt.legend(fontsize=17)
    plt.grid("on")

    plt.savefig("images/plot_uncalibrate_"+str(idTest)+".pdf")
    plt.show()
    aaaaa
    #-----------------

    # fit the isotonic regression
    isotonic = IsotonicRegression(out_of_bounds='clip')
    isotonic.fit(empirical_hdr, predicted_hdr)

    #Visualization
    plt.figure(figsize=(10,7))
    plt.scatter(predicted_hdr, isotonic.predict(empirical_hdr), alpha=0.7)
    plt.plot([0,1],[0,1],'--', color='grey', label='Perfect calibration')
    plt.xlabel('Predicted HDR', fontsize=17)
    plt.ylabel('Empirical HDR', fontsize=17)
    plt.title('Calibration with Isotonic Regression', fontsize=17)
    plt.legend(fontsize=17)
    plt.grid("on")

    plt.savefig("images/calibration/plot_calibrate_"+str(idTest)+".pdf")
    plt.show()

    #----------------

    conf_levels = np.arange(start=0.0, stop=1.025, step=0.05) # Valores de alpha

    unc_pcts = []
    cal_pcts = []
    unc_pcts2 = []
    cal_pcts2 = []

    for alpha in conf_levels:
        #alpha = 0.4
        new_alpha = isotonic.transform([alpha])
        print("alpha: ", alpha, " -- new_alpha: ", new_alpha)
        #print("hdr: ", 1-alpha, " -- new_hdr: ", 1-new_alpha)

        perc_within_cal = []
        perc_within_unc = []
        for i in range(tpred_samples_cal.shape[1]):
            # Ground Truth
            gt = target_cal[i,position,:].cpu()

            if gaussian:
                # Estimamos la pdf
                kde, sample_kde = gaussian_kde2(tpred_samples_cal, sigmas_samples_cal, data_cal, target_cal, i, position, resample_size=1000, idTest=idTest)
            else:
                # Muestra de la distribución bayessiana
                this_pred_out_abs = tpred_samples_cal[:, i, position, :] + np.array([data_cal[i,:,:][-1].numpy()])
                sample_kde = this_pred_out_abs.T
                # Creamos la función de densidad con KDE, references: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
                kde = gaussian_kde(sample_kde)
                sample_kde = kde.resample(1000,0)

            #--------
            # Pasos para calcular fa del HDR

            # Evaluamos la muestra en la pdf
            sample_pdf = kde.pdf(sample_kde)

            # Ordenamos de forma descendente las muestras de pdf
            # Ordenamos las muestras
            orden = sorted(sample_pdf, reverse=True) # Ordenamos
            print(orden[:30])
            ind = int(len(orden)*(1.0-new_alpha)) # Encontramos el indice del alpha-esimo elemento
            print("new_alpha: ", new_alpha)
            print("1-new_alpha: ", 1-new_alpha)
            print("ind: ", ind)
            #if (ind==len(orden)) or (ind==(len(orden)-1)) or (alpha==0):
            if (ind==len(orden)) or (alpha==0):
                print("si es igual")
                fa = 0.0
            else:
                fa = orden[ind] # tomamos el valor del alpha-esimo elemento mas grande
            print("fa: ", fa)

            #sum = 0
            #fa = orden[-1][0]
            #for ii, val in enumerate(orden):
            #    sum += val[1]
            #    #if sum >= new_alpha:
            #    if sum >= (1.0-new_alpha):
            #        fa = val[0]
            #        break

            # Encontramos fa a partir de las muestras de pdf
            # Ordenamos las muestras
            orden = sorted(sample_pdf, reverse=True) # Ordenamos
            #print(orden[:30])
            ind = int(len(orden)*(1.0-alpha)) # Encontramos el indice del alpha-esimo elemento
            #if (ind==len(orden)) or (ind==(len(orden)-1)) or (alpha==0):
            if (ind==len(orden)) or (alpha==0):
                fa_unc = 0.0
            else:
                fa_unc = orden[ind] # tomamos el valor del alpha-esimo elemento mas grande
            print("fa_unc: ", fa_unc)
            #sum = 0
            #fa_unc = orden[-1][0]
            #for ii, val in enumerate(orden):
            #    sum += val[1]
            #    if sum >= (1.0-alpha):
            #        fa_unc = val[0]
            #        break

            f_pdf = kde.pdf(gt)
            print("gt_pdf: ", f_pdf)
            print("f_pdf >= fa_unc: ", f_pdf >= fa_unc)
            print("f_pdf >= fa: ", f_pdf >= fa)
            perc_within_cal.append(f_pdf >= fa)
            perc_within_unc.append(f_pdf >= fa_unc)
            #-----

        # Guardamos los resultados de todo el batch para un alpha especifico
        print(perc_within_unc)
        print(np.mean(perc_within_unc))
        print(perc_within_cal)
        print(np.mean(perc_within_cal))
        cal_pcts.append(np.mean(perc_within_cal))
        unc_pcts.append(np.mean(perc_within_unc))
        #aaaaa

    print(conf_levels)
    print(isotonic.transform(conf_levels))
    print(1-isotonic.transform(conf_levels))

    if tpred_samples_test is not None:
        for alpha in conf_levels:
            new_alpha = isotonic.transform([alpha])

            perc_within_cal = []
            perc_within_unc = []
            for i in range(tpred_samples_test.shape[1]):
                #i = 1
                # Ground Truth
                gt = target_test[i,position,:].cpu()

                if gaussian:
                    # Estimamos la pdf
                    kde, sample_kde = gaussian_kde2(tpred_samples_test, sigmas_samples_test, data_test, target_test, i, position, resample_size=1000, idTest=idTest)
                else:
                    # Muestra de la distribución bayessiana
                    this_pred_out_abs = tpred_samples_test[:, i, position, :] + np.array([data_test[i,:,:][-1].numpy()])
                    sample_kde = this_pred_out_abs.T
                    # Creamos la función de densidad con KDE, references: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
                    kde = gaussian_kde(sample_kde)
                    sample_kde = kde.resample(1000,0)

                #--------
                # Pasos para calcular fa del HDR

                # Evaluamos la muestra en la pdf
                sample_pdf = kde.pdf(sample_kde)

                # Ordenamos de forma descendente las muestras de pdf
                sample_pdf_zip = zip(sample_pdf, sample_pdf/np.sum(sample_pdf))
                orden = sorted(sample_pdf_zip, key=lambda x: x[1], reverse=True)

                # Encontramos fa a partir de las muestras de pdf
                orden_i, orden_val = zip(*orden)
                ind = np.where(np.cumsum(orden_val) >= (1.0-new_alpha))[0]
    #            print(ind)
                if (ind.shape[0] == 0) :
    #                print("ind1: ", -1)
                    #fa = orden[-1][0]
                    fa = 0.0
                elif (list(ind) == [len(orden)-1]) and (alpha==0.0):
                    fa = 0.0
                else:
    #                print("ind1: ", ind)
                    fa = orden_i[ind[0]]
    #            print(fa)

                # Encontramos fa a partir de las muestras de pdf
                orden_i, orden_val = zip(*orden)
                ind = np.where(np.cumsum(orden_val) >= (1.0-alpha))[0]
    #            print(ind)
    #            print(ind.shape)
                if (ind.shape[0] == 0):
                    #fa_unc = orden[-1][0]
                    fa_unc = 0.0
                elif (list(ind) == [len(orden)-1]) and (alpha==0.0):
                    fa_unc = 0.0
                else:
                    fa_unc = orden_i[ind[0]]

                f_pdf = kde.pdf(gt)
                perc_within_cal.append(f_pdf >= fa)
                perc_within_unc.append(f_pdf >= fa_unc)
                #-----

            # Guardamos los resultados de todo el batch para un alpha especifico
            cal_pcts2.append(np.mean(perc_within_cal))
            unc_pcts2.append(np.mean(perc_within_unc))


    #--------------------- Guardamos las curvas de calibracion ---------------------------

    plt.figure(figsize=(10,7))
    plt.plot([0,1],[0,1],'--', color='grey')
    plt.plot(1-conf_levels, unc_pcts, '-o', color='purple', label='Uncalibrated')
    plt.plot(1-conf_levels, cal_pcts, '-o', color='red', label='Calibrated')
    plt.legend(fontsize=14)
    #plt.title('Calibration Plot on Testing Data ('+str(idTest)+')', fontsize=17)
    plt.xlabel(r'$\alpha$', fontsize=17)
    plt.ylabel(r'$\hat{P}_\alpha$', fontsize=17)

    if gaussian:
        plt.savefig("images/calibration/confidence_level/confidence_level_cal_IsotonicReg_"+str(idTest)+"_"+str(position)+"_gaussian.pdf")
    else:
        plt.savefig("images/calibration/confidence_level/confidence_level_cal_IsotonicReg_"+str(idTest)+"_"+str(position)+".pdf")
    plt.show()

    if tpred_samples_test is not None:
        plt.figure(figsize=(10,7))
        plt.plot([0,1],[0,1],'--', color='grey')
        plt.plot(1-conf_levels, unc_pcts2, '-o', color='purple', label='Uncalibrated')
        plt.plot(1-conf_levels, cal_pcts2, '-o', color='red', label='Calibrated')
        plt.legend(fontsize=14)
        #plt.title('Calibration Plot on Testing Data ('+str(idTest)+')', fontsize=17)
        plt.xlabel(r'$\alpha$', fontsize=17)
        plt.ylabel(r'$\hat{P}_\alpha$', fontsize=17)

        if gaussian:
            plt.savefig("images/calibration/confidence_level/confidence_level_test_IsotonicReg_"+str(idTest)+"_"+str(position)+"_gaussian.pdf")
        else:
            plt.savefig("images/calibration/confidence_level/confidence_level_test_IsotonicReg_"+str(idTest)+"_"+str(position)+".pdf")
        plt.show()

    #----------------------------------------------------------------------
    return 1-conf_levels, unc_pcts, cal_pcts, unc_pcts2, cal_pcts2, isotonic


def calibration_pdf(tpred_samples, isotonic, position, draw=False):

    list_new_position = []

    # Creamos un KDE density con las muestras # tpred_samples[ind_ensemble, id_batch, position, xy]
    for k in range(tpred_samples.shape[1]): # Recorremos cada trayectoria del batch

        # Muestra de la distribución bayessiana
        yi = tpred_samples[:, k, position, :]

        # Creamos la función de densidad con KDE, references: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
        fk = gaussian_kde(yi.T)

        # Evaluamos la muestra en la pdf
        fk_yi = fk.pdf(yi.T)
        #print(fk_yi)

        # Ordenamos las muestras
        orden = sort_sample(fk_yi)

        # Buscamos alpha (HDR)
        alpha = [ get_alpha( orden, fk_alpha) for fk_alpha in fk_yi ]
        #print("alpha: ", alpha)

        # Calibramos el HDR
        gamma_alpha = isotonic.transform(alpha)
        #print("gamma_alpha: ", gamma_alpha)

        # Encontramos f_gamma(HDR) a partir de las muestras de pdf
        fk_gamma_alpha = [ get_falpha(orden, alpha_i) for alpha_i in gamma_alpha]
        #print("fk_gamma_alpha: ", fk_gamma_alpha)

        # Creamos el importance weight
        w_i = np.array(fk_gamma_alpha)/fk_yi
        #print("w_i: ", w_i)

        # Creamos la nueva función de densidad pesada
        fk_pdf = gaussian_kde(yi.T, weights=w_i)

        plt.figure()
        sns.kdeplot(x=yi[:,0], y=yi[:,1], label='KDE')
        #sns.kdeplot(x=yi[:,0], y=yi[:,1], weights=w_i, label='KDE Weights')
        plt.legend()
        plt.xlabel('x-position')
        plt.ylabel('y-position')
        plt.title("Highest Density Regions with Isotonic Regresion, id_batch=" + str(k))
        plt.savefig("images/HDR0/plot_hdr_"+str(k)+"_"+str(position)+"_KDE.pdf")
        #plt.savefig("images/calib/kdeplot_"+str(k)+".png")
        plt.close()

        plt.figure()
        sns.kdeplot(x=yi[:,0], y=yi[:,1], color="orange", weights=w_i, label='KDE Weights')
        plt.legend()
        plt.xlabel('x-position')
        plt.ylabel('y-position')
        plt.title("Highest Density Regions with Isotonic Regresion, id_batch=" + str(k))
        plt.savefig("images/HDR0/plot_hdr_"+str(k)+"_"+str(position)+"_KDEWeights.pdf")
        #plt.savefig("images/calib/kdeplot_"+str(k)+".png")
        plt.close()


        # Muestreamos de la nueva función de densidad pesada
        num_sample = tpred_samples.shape[0]
        new_position = get_samples_pdfWeight(fk_pdf, num_sample)
        list_new_position.append(new_position.T)

    return np.array(list_new_position)

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


def calibration_pdf2(tpred_samples, data_test, target_test, position, alpha = 0.85, id_batch=-2, draw=False):

    list_fk = []
    s_xk_yk = []
    # Creamos un KDE density con las muestras # tpred_samples[ind_ensemble, id_batch, position, xy]
    for k in range(tpred_samples.shape[1]): # Recorremos cada trayectoria del batch

        # Muestra de la distribución bayessiana
        yi = tpred_samples[:, k, position, :]

        # Creamos la función de densidad con KDE, references: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
        fk = gaussian_kde(yi.T)

        # Evaluamos el GT
        gt = target_test[k,position,:].detach().numpy()
        fk_yi = -fk.pdf(gt.T)

        # Guardamos en una lista
        s_xk_yk.append(fk_yi)
        list_fk.append(fk)

    # Ordenamos las muestras
    s_xk_yk = np.array(s_xk_yk)
    orden = sort_sample(s_xk_yk)

    # Encontramos el S_alpha (1 − alpha quantile )
    Sa = get_falpha(orden, alpha)
    #print("Sa encontrado: ", -Sa)


    #--------------------------------------------------
    if draw:
        #-------------- para un id_batch ----------------------
        # Encontramos el alpha correspondiente al nuevo Sa en la funcion de densidad
        # Obtenemos la muestra de interes
        yi = tpred_samples[:, id_batch, position, :] # Seleccionamos las muestras de una trayectoria
        gt = target_test[id_batch, position,:].detach().numpy()
        #print("yi: ", yi)
        #print("gt: ", gt)

        # Creamos la pdf para la muestra
        fk = gaussian_kde(yi.T)
        # Evaluamos la muestra en la pdf
        fk_yi = fk.pdf(yi.T) # Evaluamos en la funcion de densidad

        # Ordenamos las muestras
        orden_fk = sort_sample(fk_yi)
        # Encontramos el alpha
        alpha_fk = get_alpha( orden_fk, -Sa)

        #print("alpha: ", alpha)
        #print("Sa: ", Sa)
        #print("ind: ", ind)
        #print("pos_Sa: ", pos_Sa)
        #print("alpha_fk: ", alpha_fk)

        #--------------------------------------------------
        # Visualizamos la distribucion

        plt.figure()
        sns.kdeplot(x=yi[:,0], y=yi[:,1], label='KDE')
        sns.kdeplot(x=yi[:,0], y=yi[:,1], levels=[alpha], label=r'$\alpha$'+"=%.2f"%(alpha)) # Para colocar bien el Sa debemos usar el alpha
        sns.kdeplot(x=yi[:,0], y=yi[:,1], levels=[alpha_fk], label=r'$\alpha_{new}$'+"=%.2f"%(alpha_fk))
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
        #sns.kdeplot(x=yi[:,0], y=yi[:,1], levels=[alpha], label=r'$\alpha$'+"=%.2f"%(alpha)) # Para colocar bien el Sa debemos usar el alpha
        #sns.kdeplot(x=yi[:,0], y=yi[:,1], levels=[alpha_fk], label=r'$\alpha_{new}$'+"=%.2f"%(alpha_fk))
        #plt.scatter(gt[0], gt[1], marker='^', color="blue", linewidth=3, label="GT")
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

    return -Sa

def calibration_pdf3(tpred_samples, target_test, position, alpha = 0.85, id_batch=-2, draw=False):

    list_fk = []
    s_xk_yk = []
    # Creamos un KDE density con las muestras # tpred_samples[ind_ensemble, id_batch, position, xy]
    for k in range(tpred_samples.shape[1]): # Recorremos cada trayectoria del batch

        # Muestra de la distribución bayessiana
        yi = tpred_samples[:, k, position, :]

        # Creamos la función de densidad con KDE, references: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
        fk = gaussian_kde(yi.T)

        # Evaluamos el GT
        gt = target_test[k,position,:].detach().numpy()
        fk_max = fk.pdf(yi.T).max()
        #print("fk samples: ", fk.pdf(yi.T).shape)
        #print("fk_max: ", fk_max)
        fk_yi = fk.pdf(gt.T)

        # Guardamos en una lista
        s_xk_yk.append(-fk_yi/fk_max)
        list_fk.append(fk)

    # Ordenamos las muestras
    s_xk_yk = np.array(s_xk_yk)
    orden = sort_sample(s_xk_yk)

    # Encontramos el S_alpha (1 − alpha quantile )
    Sa = get_falpha(orden, alpha)
    fk_max_Sa = fk_max*Sa
    #print("Sa encontrado: ", -Sa, -fk_max_Sa)


    #--------------------------------------------------
    if draw:
        #-------------- para un id_batch ----------------------
        # Encontramos el alpha correspondiente al nuevo Sa en la funcion de densidad
        # Obtenemos la muestra de interes
        yi = tpred_samples[:, id_batch, position, :] # Seleccionamos las muestras de una trayectoria
        gt = target_test[id_batch, position,:].detach().numpy()
        #print("yi: ", yi)
        #print("gt: ", gt)

        # Creamos la pdf para la muestra
        fk = gaussian_kde(yi.T)
        # Evaluamos la muestra en la pdf
        fk_yi = fk.pdf(yi.T) # Evaluamos en la funcion de densidad
        # Ordenamos las muestras
        orden_fk = sort_sample(fk_yi)
        # Encontramos el alpha
        alpha_fk = get_alpha( orden_fk, -fk_max_Sa)
        #if alpha_fk < 1e-10:
        #    alpha_fk = 0.0

        #print("alpha: ", alpha)
        #print("Sa: ", Sa)
        #print("f_max Sa: ", fk_max_Sa)
        #print("alpha_fk: ", alpha_fk)

        #--------------------------------------------------
        # Visualizamos la distribucion

        plt.figure()
        sns.kdeplot(x=yi[:,0], y=yi[:,1], label='KDE')
        sns.kdeplot(x=yi[:,0], y=yi[:,1], levels=[alpha], label=r'$\alpha$'+"=%.2f"%(alpha)) # Para colocar bien el Sa debemos usar el alpha
        sns.kdeplot(x=yi[:,0], y=yi[:,1], levels=[alpha_fk], label=r'$\alpha_{new}$'+"=%.2f"%(alpha_fk))
        plt.scatter(gt[0], gt[1], marker='^', color="blue", linewidth=3, label="GT")

        plt.legend()
        plt.xlabel('x-position')
        plt.ylabel('y-position')
        plt.title("Conformal Highest Density Regions with GT, S"+r'$_\alpha$'+"=%.2f"%(Sa)+", id_batch=" + str(id_batch))
        plt.savefig("images/HDR2/plot_hdr_%.2f_"%(alpha)+"_"+str(id_batch)+"_"+str(position)+"_gt.pdf")
        plt.close()

    return -fk_max_Sa


# Sort samples with respect to their density value
def sort_sample(sample_pdf):
    # Samples from the pdf are sorted in decreasing order
    sample_pdf_zip = zip(sample_pdf, sample_pdf/np.sum(sample_pdf))
    return sorted(sample_pdf_zip, key=lambda x: x[1], reverse=True)

# Given a value on the pdf, deduce alpha
def get_alpha(orden, f_pdf):
    # Predicted HDR
    ind = np.where(np.array(orden)[:,0] >= f_pdf)[0]
    ind = 0 if ind.size == 0 else ind[-1] # Validamos que no sea el primer elemento mas grande
    alpha = np.array(orden)[:ind+1,1].sum()
    return alpha

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
                ##print("alpha: ", alpha)
                ##print("ind: ", ind)
                ##print("fa_unc encontrado: ", fa_unc)

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

    plt.savefig("images/calibration/confidence_level/confidence_level_cal_"+str(idTest)+"_conformal"+str(method)+"_"+str(position)+".pdf")
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

        plt.savefig("images/calibration/confidence_level/confidence_level_test_"+str(idTest)+"_conformal"+str(method)+"_"+str(position)+".pdf")
        plt.show()
    #----------------------------------------------------------------------
    return conf_levels, unc_pcts, cal_pcts, unc_pcts2, cal_pcts2

def generate_metrics_calibration_IsotonicReg(tpred_samples_cal, data_cal, target_cal, sigmas_samples_cal, id_test, gaussian=False, tpred_samples_test=None, data_test=None, target_test=None, sigmas_samples_test=None):

    #--------------------- Calculamos las metricas de calibracion ---------------------------------
    metrics1 = [["","MACE","RMSCE","MA"]]
    metrics2_test = [["","MACE","RMSCE","MA"]]

    # Recorremos cada posicion
    for pos in range(tpred_samples_cal.shape[2]):
        pos = 11
        #pos = 0 # borrar, solo para pruebas
        print("Procesamos para posicion: ", pos)

        # HDR y Calibracion
        exp_proportions, obs_proportions_unc, obs_proportions_cal, obs_proportions_unc2, obs_proportions_cal2 , isotonic = calibration_IsotonicReg(tpred_samples_cal, data_cal, target_cal, sigmas_samples_cal, position = pos, idTest=id_test, gaussian=gaussian, tpred_samples_test=tpred_samples_test, data_test=data_test, target_test=target_test, sigmas_samples_test=sigmas_samples_test) # Se hace sobre la posicion absoluta
        plt.show()
        #aaaaa

        # Modificamos la distribución y graficamos
        #new_positions = calibration_pdf(tpred_samples, isotonic, pos)

        # Metricas de Calibration
        ma1    = miscalibration_area(exp_proportions, obs_proportions_unc)
        mace1  = mean_absolute_calibration_error(exp_proportions, obs_proportions_unc)
        rmsce1 = root_mean_squared_calibration_error(exp_proportions, obs_proportions_unc)
        metrics1.append(["Before Recalibration pos "+str(pos), mace1, rmsce1, ma1])
        print("Before Recalibration:  ", end="")
        print("MACE: {:.5f}, RMSCE: {:.5f}, MA: {:.5f}".format(mace1, rmsce1, ma1))

        ma2 = miscalibration_area(exp_proportions, obs_proportions_cal)
        mace2 = mean_absolute_calibration_error(exp_proportions, obs_proportions_cal)
        rmsce2 = root_mean_squared_calibration_error(exp_proportions, obs_proportions_cal)
        metrics1.append(["After Recalibration pos "+str(pos), mace2, rmsce2, ma2])
        print("After Recalibration:  ", end="")
        print("MACE: {:.5f}, RMSCE: {:.5f}, MA: {:.5f}".format(mace2, rmsce2, ma2))

        #break # borrar solo para pruebas

        if tpred_samples_test is not None:
            # Metrics Calibration Test
            ma3    = miscalibration_area(exp_proportions, obs_proportions_unc2)
            mace3  = mean_absolute_calibration_error(exp_proportions, obs_proportions_unc2)
            rmsce3 = root_mean_squared_calibration_error(exp_proportions, obs_proportions_unc2)
            metrics2_test.append(["Before Recalibration pos "+str(pos), mace3, rmsce3, ma3])
            print("Before Recalibration:  ", end="")
            print("MACE: {:.5f}, RMSCE: {:.5f}, MA: {:.5f}".format(mace3, rmsce3, ma3))

            ma4 = miscalibration_area(exp_proportions, obs_proportions_cal2)
            mace4 = mean_absolute_calibration_error(exp_proportions, obs_proportions_cal2)
            rmsce4 = root_mean_squared_calibration_error(exp_proportions, obs_proportions_cal2)
            metrics2_test.append(["After Recalibration pos "+str(pos), mace4, rmsce4, ma4])
            print("After Recalibration:  ", end="")
            print("MACE: {:.5f}, RMSCE: {:.5f}, MA: {:.5f}".format(mace4, rmsce4, ma4))
        break

    # Guardamos los resultados de las metricas
    df = pd.DataFrame(metrics1)
    df.to_csv("images/calibration/metrics/metrics_calibration_cal_IsotonicRegresion_"+str(id_test)+".csv")

    if tpred_samples_test is not None:
        # Guardamos los resultados de las metricas de Test
        df = pd.DataFrame(metrics2_test)
        df.to_csv("images/calibration/metrics/metrics_calibration_test_IsotonicRegresion_"+str(id_test)+".csv")

    # ---------------- Solo caso deterministico
    # Agregamos el calculo de la nll
    position = 11
    ll_cal = []
    ll_uncal = []
    for i in range(tpred_samples_test.shape[1]):
        # Ground Truth
        gt = target_test[i,position,:].cpu()

        # Muestra de la distribución bayessiana
        this_pred_out_abs = tpred_samples_test[:, i, position, :] + np.array([data_test[i,:,:][-1].numpy()])
        sample_kde = this_pred_out_abs.T
        kde = gaussian_kde(sample_kde)
        sample_kde = kde.resample(1000,0)

        #--------
        # Pasos para calcular fa del HDR

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
        kernel = gaussian_kde(sample_kde, weights=importance_weights)
        ll_cal.append(kernel.logpdf(gt))
        ll_uncal.append(kde.logpdf(gt))
        print("neg. log: ", -kernel.logpdf(gt), -kde.logpdf(gt))
        #-----

    # Calculamos el Negative LogLikelihood
    nll_cal = statistics.median(ll_cal)
    nll_uncal = statistics.median(ll_uncal)

    df = pd.DataFrame([["calibrated", "uncalibrated"],[nll_cal, nll_uncal]])
    df.to_csv("images/calibration/metrics/nll_IsotonicRegresion_"+str(id_test)+".csv")
    print(df)



def generate_metrics_calibration_conformal(tpred_samples_cal, data_cal, targetrel_cal, target_cal, sigmas_samples_cal, id_test, gaussian=False, tpred_samples_test=None, data_test=None, targetrel_test=None, target_test=None, sigmas_samples_test=None):
    #--------------------- Calculamos las metricas de calibracion ---------------------------------
    metrics2 = [["","MACE","RMSCE","MA"]]
    metrics3 = [["","MACE","RMSCE","MA"]]
    metrics2_test = [["","MACE","RMSCE","MA"]]
    metrics3_test = [["","MACE","RMSCE","MA"]]
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
        ma1    = miscalibration_area(exp_proportions, obs_proportions_unc)
        mace1  = mean_absolute_calibration_error(exp_proportions, obs_proportions_unc)
        rmsce1 = root_mean_squared_calibration_error(exp_proportions, obs_proportions_unc)
        metrics2.append(["Before Recalibration pos "+str(pos), mace1, rmsce1, ma1])
        print("Before Recalibration:  ", end="")
        print("MACE: {:.5f}, RMSCE: {:.5f}, MA: {:.5f}".format(mace1, rmsce1, ma1))

        ma2 = miscalibration_area(exp_proportions, obs_proportions_cal)
        mace2 = mean_absolute_calibration_error(exp_proportions, obs_proportions_cal)
        rmsce2 = root_mean_squared_calibration_error(exp_proportions, obs_proportions_cal)
        metrics2.append(["After Recalibration pos "+str(pos), mace2, rmsce2, ma2])
        print("After Recalibration:  ", end="")
        print("MACE: {:.5f}, RMSCE: {:.5f}, MA: {:.5f}".format(mace2, rmsce2, ma2))

        if tpred_samples_test is not None:
            # Metrics Calibration Test
            ma3    = miscalibration_area(exp_proportions, obs_proportions_unc2)
            mace3  = mean_absolute_calibration_error(exp_proportions, obs_proportions_unc2)
            rmsce3 = root_mean_squared_calibration_error(exp_proportions, obs_proportions_unc2)
            metrics2_test.append(["Before Recalibration pos "+str(pos), mace3, rmsce3, ma3])
            print("Before Recalibration:  ", end="")
            print("MACE: {:.5f}, RMSCE: {:.5f}, MA: {:.5f}".format(mace3, rmsce3, ma3))

            ma4 = miscalibration_area(exp_proportions, obs_proportions_cal2)
            mace4 = mean_absolute_calibration_error(exp_proportions, obs_proportions_cal2)
            rmsce4 = root_mean_squared_calibration_error(exp_proportions, obs_proportions_cal2)
            metrics2_test.append(["After Recalibration pos "+str(pos), mace4, rmsce4, ma4])
            print("After Recalibration:  ", end="")
            print("MACE: {:.5f}, RMSCE: {:.5f}, MA: {:.5f}".format(mace4, rmsce4, ma4))

        print("------- calibration_pdf3 ")
        exp_proportions, obs_proportions_unc, obs_proportions_cal, obs_proportions_unc2, obs_proportions_cal2 = calibration_Conformal(tpred_samples_cal, data_cal, gt, target_cal, sigmas_samples_cal, position = pos, idTest=id_test, method=3, gaussian=gaussian, tpred_samples_test=tpred_samples_test, data_test=data_test, target_test=gt_test, target_test2=target_test, sigmas_samples_test=sigmas_samples_test)

        # Metrics Calibration
        ma1    = miscalibration_area(exp_proportions, obs_proportions_unc)
        mace1  = mean_absolute_calibration_error(exp_proportions, obs_proportions_unc)
        rmsce1 = root_mean_squared_calibration_error(exp_proportions, obs_proportions_unc)
        metrics3.append(["Before Recalibration pos "+str(pos), mace1, rmsce1, ma1])
        print("Before Recalibration:  ", end="")
        print("MACE: {:.5f}, RMSCE: {:.5f}, MA: {:.5f}".format(mace1, rmsce1, ma1))

        ma2 = miscalibration_area(exp_proportions, obs_proportions_cal)
        mace2 = mean_absolute_calibration_error(exp_proportions, obs_proportions_cal)
        rmsce2 = root_mean_squared_calibration_error(exp_proportions, obs_proportions_cal)
        metrics3.append(["After Recalibration pos "+str(pos), mace2, rmsce2, ma2])
        print("After Recalibration:  ", end="")
        print("MACE: {:.5f}, RMSCE: {:.5f}, MA: {:.5f}".format(mace2, rmsce2, ma2))

        if tpred_samples_test is not None:
            # Metrics Calibration Test
            ma3    = miscalibration_area(exp_proportions, obs_proportions_unc2)
            mace3  = mean_absolute_calibration_error(exp_proportions, obs_proportions_unc2)
            rmsce3 = root_mean_squared_calibration_error(exp_proportions, obs_proportions_unc2)
            metrics3_test.append(["Before Recalibration pos "+str(pos), mace3, rmsce3, ma3])
            print("Before Recalibration:  ", end="")
            print("MACE: {:.5f}, RMSCE: {:.5f}, MA: {:.5f}".format(mace3, rmsce3, ma3))

            ma4 = miscalibration_area(exp_proportions, obs_proportions_cal2)
            mace4 = mean_absolute_calibration_error(exp_proportions, obs_proportions_cal2)
            rmsce4 = root_mean_squared_calibration_error(exp_proportions, obs_proportions_cal2)
            metrics3_test.append(["After Recalibration pos "+str(pos), mace4, rmsce4, ma4])
            print("After Recalibration:  ", end="")
            print("MACE: {:.5f}, RMSCE: {:.5f}, MA: {:.5f}".format(mace4, rmsce4, ma4))

        break

    # Guardamos los resultados de las metricas
    df = pd.DataFrame(metrics2)
    df.to_csv("images/calibration/metrics/metrics_calibration_cal_confolmal2_"+str(id_test)+".csv")

    df = pd.DataFrame(metrics3)
    df.to_csv("images/calibration/metrics/metrics_calibration_cal_confolmal3_"+str(id_test)+".csv")

    if tpred_samples_test is not None:
        # Guardamos los resultados de las metricas de Test
        df = pd.DataFrame(metrics2_test)
        df.to_csv("images/calibration/metrics/metrics_calibration_test_confolmal2_"+str(id_test)+".csv")

        df = pd.DataFrame(metrics3_test)
        df.to_csv("images/calibration/metrics/metrics_calibration_test_confolmal3_"+str(id_test)+".csv")



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


def miscalibration_area(
    exp_proportions: np.ndarray,
    obs_proportions: np.ndarray
) -> float:
    """Miscalibration area.
    This is identical to mean absolute calibration error and ECE, however
    the integration here is taken by tracing the area between curves.
    In the limit of num_bins, miscalibration area and
    mean absolute calibration error will converge to the same value.
    Args:

    Returns:
        A single scalar which calculates the miscalibration area.
    """

    # Compute approximation to area between curves
    polygon_points = []
    for point in zip(exp_proportions, obs_proportions):
        polygon_points.append(point)
    for point in zip(reversed(exp_proportions), reversed(exp_proportions)):
        polygon_points.append(point)
    polygon_points.append((exp_proportions[0], obs_proportions[0]))
    polygon = Polygon(polygon_points)
    x, y = polygon.exterior.xy
    ls = LineString(np.c_[x, y])
    lr = LineString(ls.coords[:] + ls.coords[0:1])
    mls = unary_union(lr)
    polygon_area_list = [poly.area for poly in polygonize(mls)]
    miscalibration_area = np.asarray(polygon_area_list).sum()

    return miscalibration_area

def mean_absolute_calibration_error(
     exp_proportions: np.ndarray,
     obs_proportions: np.ndarray
) -> float:
    """Mean absolute calibration error; identical to ECE.
    Args:

    Returns:
        A single scalar which calculates the mean absolute calibration error.
    """

    abs_diff_proportions = np.abs(exp_proportions - obs_proportions)
    mace = np.mean(abs_diff_proportions)

    return mace

def root_mean_squared_calibration_error(
     exp_proportions: np.ndarray,
     obs_proportions: np.ndarray
) -> float:
    """Root mean squared calibration error.
    Args:

    Returns:
        A single scalar which calculates the root mean squared calibration error.
    """

    squared_diff_proportions = np.square(exp_proportions - obs_proportions)
    rmsce = np.sqrt(np.mean(squared_diff_proportions))

    return rmsce

def generate_one_batch_test_ensembles(batched_test_data, model, num_samples, TRAINING_CKPT_DIR, model_name, id_test=2, device=None, dim_pred=12):

    #----------- Dataset TEST -------------
    datarel_test_full = []
    targetrel_test_full = []
    data_test_full = []
    target_test_full = []
    for batch_idx, (datarel_test, targetrel_test, data_test, target_test) in enumerate(batched_test_data):
        if batch_idx==0:
            continue

        # Guardamos en una lista los batches
        datarel_test_full.append(datarel_test)
        targetrel_test_full.append(targetrel_test)
        data_test_full.append(data_test)
        target_test_full.append(target_test)

    # Concatenamos los batches para formar uno solo
    datarel_test_full = torch.cat(datarel_test_full, dim=0)
    targetrel_test_full = torch.cat( targetrel_test_full, dim=0)
    data_test_full = torch.cat( data_test_full, dim=0)
    target_test_full = torch.cat( target_test_full, dim=0)

    # Obtenemos las predicciones para el batch unico
    tpred_samples_full = []
    sigmas_samples_ful = []
    # Muestreamos con cada modelo
    for ind in range(num_samples):

        # Cargamos el Modelo
        model.load_state_dict(torch.load(TRAINING_CKPT_DIR+"/"+model_name+"_"+str(ind)+"_"+str(id_test)+".pth"))
        model.eval()

        # Para utlizar cuda
        if torch.cuda.is_available():
            datarel_test_full  = datarel_test_full.to(device)

        # Obtenemos la predicción con el modelo
        pred, sigmas = model.predict(datarel_test_full, dim_pred=12)

        # Guardamos la muestra
        tpred_samples_full.append(pred)
        sigmas_samples_ful.append(sigmas)

    # Convertimos a array np
    tpred_samples_full = np.array(tpred_samples_full)
    sigmas_samples_ful = np.array(sigmas_samples_ful)
    #---------------------------------------

    return datarel_test_full, targetrel_test_full, data_test_full, target_test_full, tpred_samples_full, sigmas_samples_ful

def generate_one_batch_test_dropout(batched_test_data, model, num_samples, TRAINING_CKPT_DIR, model_name, id_test=2, device=None, dim_pred=12):

    #----------- Dataset TEST -------------
    datarel_test_full = []
    targetrel_test_full = []
    data_test_full = []
    target_test_full = []
    for batch_idx, (datarel_test, targetrel_test, data_test, target_test) in enumerate(batched_test_data):
        if batch_idx==0:
            continue

        # Guardamos en una lista los batches
        datarel_test_full.append(datarel_test)
        targetrel_test_full.append(targetrel_test)
        data_test_full.append(data_test)
        target_test_full.append(target_test)

    # Concatenamos los batches para formar uno solo
    datarel_test_full = torch.cat(datarel_test_full, dim=0)
    targetrel_test_full = torch.cat( targetrel_test_full, dim=0)
    data_test_full = torch.cat( data_test_full, dim=0)
    target_test_full = torch.cat( target_test_full, dim=0)

    # Obtenemos las predicciones para el batch unico
    tpred_samples_full = []
    sigmas_samples_ful = []
    # Muestreamos con cada modelo
    for ind in range(num_samples):

        # Para utlizar cuda
        if torch.cuda.is_available():
            datarel_test_full  = datarel_test_full.to(device)

        # Obtenemos la predicción con el modelo
        pred, sigmas = model.predict(datarel_test_full, dim_pred=12)

        # Guardamos la muestra
        tpred_samples_full.append(pred)
        sigmas_samples_ful.append(sigmas)

    # Convertimos a array np
    tpred_samples_full = np.array(tpred_samples_full)
    sigmas_samples_ful = np.array(sigmas_samples_ful)
    #---------------------------------------

    return datarel_test_full, targetrel_test_full, data_test_full, target_test_full, tpred_samples_full, sigmas_samples_ful
