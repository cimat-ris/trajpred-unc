import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from scipy.stats import multivariate_normal
from sklearn.metrics import auc
from sklearn.isotonic import IsotonicRegression

from shapely.geometry import Polygon, LineString
from shapely.ops import polygonize, unary_union

from utils.plot_utils import plot_traj_world

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
    if display:
        label5, = plt.plot(sample_pdf[:,0], sample_pdf[:,1], ".", color="blue", alpha=0.2, label = "Gaussian Mix Samples")
        plt.title("Trajectory Plot")
        plt.legend(handles=[label1, label2, label3, label4, label5 ])
        plt.savefig("images/trajectories/traj_samples_cov_"+str(idTest)+"_"+str(i)+".pdf")
        #plt.show()
        plt.close()

    return multivariate_normal(mean_mix, cov_mix), sample_pdf


def calibration(tpred_samples, data_test, target_test, sigmas_samples, position = 0, alpha = 0.05, idTest=0):

    predicted_hdr = []
    for i in range(tpred_samples.shape[1]):
        # Ground Truth
        gt = target_test[i,position,:].cpu()

        # Estimamos la pdf y muestreamos puntos (x,y) de la pdf
        kde, sample_kde = gaussian_kde2(tpred_samples, sigmas_samples, data_test, target_test, i, position, resample_size=1000, display=True, idTest=idTest)

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

    plt.savefig("images/plot_uncalibrate_"+str(idTest)+".pdf")
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

    plt.savefig("images/plot_calibrate_"+str(idTest)+".pdf")
    plt.show()

    #----------------

    conf_levels = np.arange(start=0.0, stop=1.025, step=0.05) # Valores de alpha

    unc_pcts = []
    cal_pcts = []

    #for alpha in conf_level_lower_bounds:
    for alpha in conf_levels:
        new_alpha = isotonic.transform([alpha])

        perc_within_cal = []
        perc_within_unc = []
        for i in range(tpred_samples.shape[1]):
            #i = 1
            # Ground Truth
            gt = target_test[i,position,:].cpu()

            # Estimamos la pdf
            kde, sample_kde = gaussian_kde2(tpred_samples, sigmas_samples, data_test, target_test, i, position, resample_size=1000, idTest=idTest)

            #--------
            # Pasos para calcular fa del HDR

            # Evaluamos la muestra en la pdf
            sample_pdf = kde.pdf(sample_kde)

            # Ordenamos de forma descendente las muestras de pdf
            sample_pdf_zip = zip(sample_pdf, sample_pdf/np.sum(sample_pdf))
            orden = sorted(sample_pdf_zip, key=lambda x: x[1], reverse=True)

            # Encontramos fa a partir de las muestras de pdf
            sum = 0
            fa = orden[-1][0]
            for ii, val in enumerate(orden):
                sum += val[1]
                #if sum >= new_alpha:
                if sum >= (1.0-new_alpha):
                    fa = val[0]
                    break

            # Encontramos fa a partir de las muestras de pdf
            sum = 0
            fa_unc = orden[-1][0]
            for ii, val in enumerate(orden):
                sum += val[1]
                if sum >= (1.0-alpha):
                    fa_unc = val[0]
                    break

            f_pdf = kde.pdf(gt)
            perc_within_cal.append(f_pdf >= fa)
            perc_within_unc.append(f_pdf >= fa_unc)
            #-----

        # Guardamos los resultados de todo el batch para un alpha especifico
        cal_pcts.append(np.mean(perc_within_cal))
        unc_pcts.append(np.mean(perc_within_unc))

    plt.figure(figsize=(10,7))
    plt.plot([0,1],[0,1],'--', color='grey')
    plt.plot(1-conf_levels, unc_pcts, '-o', color='purple', label='Uncalibrated')
    plt.plot(1-conf_levels, cal_pcts, '-o', color='red', label='Calibrated')
    plt.legend(fontsize=14)
    plt.title('Calibration Plot on Testing Data ('+str(idTest)+')', fontsize=17)
    plt.xlabel('Confidence Level (1-'+r'$\alpha$'+')', fontsize=17)
    plt.ylabel('Observed Confidence Level', fontsize=17)

    plt.savefig("images/confidence_level_"+str(idTest)+".pdf")
    plt.show()



    # Calculamos el area de  cada curva con respecto a la linea perfecta
    abs_area_cal = np.abs(np.array(cal_pcts)+conf_levels-1) # Se le resta la linea perfecta
    abs_area_unc = np.abs(np.array(unc_pcts)+conf_levels-1) # Se le resta la linea perfecta
    #  Calculamos el area
    auc_cal = auc(1-conf_levels, abs_area_cal)
    auc_unc = auc(1-conf_levels, abs_area_unc)

    # Visualizamos en una tabla
    tabla = pd.DataFrame( data={"area":[auc_unc, auc_cal]}, index=pd.Index(['Uncalibrated', 'Calibrated']) )
    print(tabla)

    return auc_cal, auc_unc, 1-conf_levels, unc_pcts, cal_pcts

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
