import argparse
import logging, sys, random
import numpy as np
import torch
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

sys.path.append('.')


from utils.calibration_utils import get_data_for_calibration
from utils.calibration import generate_metrics_calibration, generate_metrics_calibration_all
from utils.constants import SUBDATASETS_NAMES, BITRAP, BITRAP_BT_SDD, DETERMINISTIC_GAUSSIAN, DETERMINISTIC_GAUSSIAN_SDD, DROPOUT, DROPOUT_SDD, ENSEMBLES, ENSEMBLES_SDD, VARIATIONAL, VARIATIONAL_SDD

from utils.plot_utils import plot_calibration_curves2
from utils.directory_utils import Output_directories
from utils.calibration import compute_calibration_metrics, save_metrics

# Parser arguments
parser = argparse.ArgumentParser(description='')
parser.add_argument('--calibration-conformal', action='store_true', help='generates metrics using calibration conformal')
parser.add_argument('--absolute-coords', action='store_true', help='')
parser.add_argument('--gaussian-output', default=False, action='store_true', help='gaussian var to be used to compute calibration metrics')
parser.add_argument('--show-plot', default=False, action='store_true', help='show the calibration plots')
parser.add_argument('--nll', default=False, action='store_true', help='Compute the values of GT negative log likelihood before and after calibration')
parser.add_argument('--test-name', type=str, default='deterministicGaussian', help='Test data to be load (default: deterministic gaussian test)')
parser.add_argument('--id-dataset',type=str, default=0, metavar='N',
					help='id of the dataset to use. 0 is ETH-UCY, 1 is SDD (default: 0)')
parser.add_argument('--id-test',type=int, default=2, metavar='N',
					help='id of the subdataset to use as test (default: 2)')
parser.add_argument('--seed',type=int, default=1,help='Random seed for all randomized functions')
parser.add_argument('--log-level',type=int, default=20,help='Log level (default: 20)')
parser.add_argument('--log-file',default='',help='Log file (default: standard output)')
args = parser.parse_args()

valid_test_names = {
	"deterministicGaussian": DETERMINISTIC_GAUSSIAN,
	"ensembles":             ENSEMBLES,
	"dropout":               DROPOUT,
	"bitrap":                BITRAP,
	"variational":           VARIATIONAL,
	"deterministicGaussianSDD": DETERMINISTIC_GAUSSIAN_SDD,
	"ensemblesSDD":             ENSEMBLES_SDD,
	"dropoutSDD":               DROPOUT_SDD,
	"bitrapSDD":                BITRAP_BT_SDD,
	"variationalSDD":           VARIATIONAL_SDD
}

def get_test_name():
	"""
	Args:
	Returns:
		- test_name
	"""
	if args.test_name not in valid_test_names.keys():
		return "ERROR: INVALID TEST NAME!!"
	return valid_test_names[args.test_name]+"_"+str(SUBDATASETS_NAMES[args.id_dataset][args.id_test])+"_calibration"


def calib_dbscan(X, gt=None, iter_max=1000, alpha=0.85, min_samples=10, TOL=5):
		
	#print("In calib_dbscan: ", X.shape)
	n_samples = X.shape[0]
	min_samples = int(n_samples*alpha)
	print('n_samples: ', n_samples)
	print('alpha: ', alpha)
	print('min_samples: ', min_samples)
	
	scaler = StandardScaler()
	X = scaler.fit_transform(X)
	for eps_step in range(iter_max):

		#eps_step = 0.1+eps_step*0.001
		eps_step = 0.001+eps_step*0.001
		#db     = DBSCAN(eps=eps_step, min_samples=min_samples).fit(X)
		db     = DBSCAN(eps=eps_step, min_samples=min_samples).fit(X)
		labels = db.labels_
		#print(eps_step)

		# Number of clusters in labels, ignoring noise if present.
		n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
		n_noise_    = list(labels).count(-1)

		#print("Estimated number of clusters: %d" % n_clusters_)
		#print("Estimated number of noise points: %d" % n_noise_)
		#print("paro: {}".format((1-alpha)*n_samples))
		#print("Within: {}".format(1.0-n_noise_/n_samples))
		###if n_noise_<=(1-alpha)*n_samples:
		if np.abs(n_noise_ - (1-alpha)*n_samples)<=TOL:
			#if n_clusters_ >0: #1
			print("---> ", eps_step)
			print("Estimated number of clusters: %d" % n_clusters_)
			print("Estimated number of noise points: %d" % n_noise_)
			print("Estimated number of points in cluster: ", n_samples-n_noise_)
			print(alpha, 1.0-n_noise_/n_samples)
			print("paro: {}".format((1-alpha)*n_samples))
			break

	unique_labels = set(labels)
	core_samples_mask = np.zeros_like(labels, dtype=bool)
	core_samples_mask[db.core_sample_indices_] = True

	hdr_samples_mask = np.zeros_like(labels, dtype=bool)
	hdr_samples_mask[db.labels_ !=-1 ] = True
	
	# Evaluamos si el gt pertenece al HDR
	if not gt is None:
		# Aplicamos la misma normalizacion que en el dbscan
		gt = scaler.transform(gt.reshape(1,-1))
		
		# Medimos la distancia con los core points
		error = db.components_ - gt
		# Distancia euclideana
		error = np.sqrt(np.sum(error**2, axis=1))
		
		# Verificamos si el punto esta dentro del HDR
		comp = np.sum(error < eps_step) > 0
		#print(comp)
		#if comp:
		#	aaaaaa

	
	#print(db.components_)
	return comp, core_samples_mask, hdr_samples_mask



def calib_dbscan_binsearch(X, gt=None, alpha=0.85, min_samples=10, TOL=0.0001):
#def calib_dbscan(X, gt=None, alpha=0.85, min_samples=10, TOL=0.001):
	step = 10
	delta_epsilon = TOL
	epsilon_min = 0.01
	epsilon_max = 0.99
		
	#print("In calib_dbscan: ", X.shape)
	n_samples = X.shape[0]
	#min_samples = int(n_samples*alpha)
	min_samples = n_samples//20
#	print('-- BIN SEARCH --')
#	print('n_samples: ', n_samples)
#	print('alpha: ', alpha)
#	print('min_samples: ', min_samples)
	
	scaler = StandardScaler()
	X = scaler.fit_transform(X)

	# Binary search version
	n_outs_ = n_samples
	while (abs(n_outs_-(1.0-alpha)*n_samples)>delta_epsilon and epsilon_max-epsilon_min>delta_epsilon):
		epsilon= 0.5*(epsilon_min+epsilon_max)
		db     = DBSCAN(eps=epsilon, min_samples=min_samples).fit(X)
		labels = db.labels_

		# Number of clusters in labels, ignoring noise if present.
		n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
		n_outs_    = list(labels).count(-1)

		#print("Epsilon: %f" % epsilon)
		#print("Epsilon min: %f" % epsilon_min)
		#print("Epsilon max: %f" % epsilon_max)
		#print("Estimated number of clusters: %d" % n_clusters_)
		#print("Estimated number of noise points: %d" % n_outs_)
		if (n_outs_<(1.0-alpha)*n_samples):
			epsilon_max = epsilon
		else:
			if (n_outs_>(1.0-alpha)*n_samples):
				epsilon_min = epsilon

	unique_labels = set(labels)
	core_samples_mask = np.zeros_like(labels, dtype=bool)
	core_samples_mask[db.core_sample_indices_] = True

	hdr_samples_mask = np.zeros_like(labels, dtype=bool)
	hdr_samples_mask[db.labels_ !=-1 ] = True

#	print("---> ", epsilon)
#	print("Estimated number of clusters: %d" % n_clusters_)
#	print("Estimated number of noise points: %d" % n_outs_)
#	print("Estimated number of points in cluster: ", n_samples-n_outs_)
#	print(alpha, 1.0-n_outs_/n_samples)
#	print("paro: {}".format((1-alpha)*n_samples))
	
	# Evaluamos si el gt pertenece al HDR
	if not gt is None:
		# Aplicamos la misma normalizacion que en el dbscan
		gt = scaler.transform(gt.reshape(1,-1))
		
		# Medimos la distancia con los core points
		error = db.components_ - gt
		# Distancia euclideana
		error = np.sqrt(np.sum(error**2, axis=1))
		
		# Verificamos si el punto esta dentro del HDR
		comp = np.sum(error < epsilon) > 0
		#print(comp)
		#if comp:
		#	aaaaaa

	
	#print(db.components_)
	return comp, core_samples_mask, hdr_samples_mask

	
def get_alpha_calibration(predictions_calibration, groundtruth_calibration, alpha, alpha_step=0.01, join_positions=False):

	# Inicializamos la variable iterativa
	alpha_update = alpha
	error_cal = []
	list_alpha_update = []
	list_Pa_hat = []
		
	# Iteramos el ciclo de calibracion
	for it in range(int(1/alpha_step)):
		
		# --------------------------------------------------------------------------
		pa = []
		# Recorremos todas las trayectorias del dataset
		for i in range(predictions_calibration.shape[1]):
				
			# Procesamos de acuerdo a la dimension de los datos a considerar
			if join_positions:
				# Consideramos una dimensión de num_positions*axis_coord
				dim1, dim2, dim3, dim4 = predictions_calibration.shape
				X = predictions_calibration.reshape(dim1,dim2,dim3*dim4)
				y = groundtruth_calibration.reshape(dim2,dim3*dim4)
				#print("New dim for dbscan: ", X.shape)
					
				# Clusterizamos con dbscan
				#comp, core_samples_mask, hdr_samples_mask = calib_dbscan(X[:,i,:], gt=y[i,:], alpha=alpha_update)
				comp, core_samples_mask, hdr_samples_mask = calib_dbscan_binsearch(X[:,i,:], gt=y[i,:], alpha=alpha_update)
			else:
				# Consideramos una posición de 2 dimensiones para una posicion de la trayectoria
				# Clusterizamos con dbscan
				#comp, core_samples_mask, hdr_samples_mask = calib_dbscan(predictions_calibration[:,i,-1,:], gt=groundtruth_calibration[i,-1,:], alpha=alpha_update)
				comp, core_samples_mask, hdr_samples_mask = calib_dbscan_binsearch(predictions_calibration[:,i,-1,:], gt=groundtruth_calibration[i,-1,:], alpha=alpha_update)
					
			# Guardamos el resultado de comparar el gt con el HDR
			pa.append(comp)
					
		# Estimamos la probabilidad empirica
		Pa_hat = np.mean(pa)
		error_cal.append(np.abs(Pa_hat-alpha))
		list_Pa_hat.append(Pa_hat)
		list_alpha_update.append(alpha_update)

		print("-------->")
		print(pa)
		print(it, alpha, Pa_hat, np.sum(pa), len(pa), alpha_step, alpha_update)
			
		# Verificamos el tipo de estimacion de la probabilidad empirica
		if it == 0:
			Pa_unc = Pa_hat
				
			if Pa_hat == alpha:
				print("*** Deteniendo calibracion ***")
				print('El alpha ya estaba calibrado: ', alpha_update)
				return Pa_unc, Pa_hat, alpha_update
				
			elif Pa_hat < alpha:
				band = 'under'
			else:
				band = 'over'

		# Caso subestimado			
		if band == 'under':
			# Validamos el criterio de paro
			if Pa_hat < alpha:
				# Actualizamos las variables
				alpha_update += alpha_step
				
				if alpha_update > 1.0:
					print("*** Truncando alpha a 1.0 ***")
					return Pa_unc, list_Pa_hat[np.argmin(error_cal)], list_alpha_update[np.argmin(error_cal)]
					#return Pa_unc, Pa_hat, 1.0
			else:
				# Si ya se cumplio detenemos la calibracion
				print("*** Deteniendo calibracion ***")
				print('Calibrando a alpha_update: ', alpha_update)
				return Pa_unc, Pa_hat, alpha_update
					
		# Caso sobrestimado
		else:
			# Validamos el criterio de paro
			if Pa_hat > alpha:
				# Actualizamos las variables
				alpha_update -= alpha_step
				
				if alpha_update < 0.0:
					print("*** Truncando alpha a 0.0 ***")
					#print('___')
					#print(error_cal)
					#print(list_alpha_update)
					#print(list_Pa_hat)
					#print(np.argmin(error_cal))
					return Pa_unc, list_Pa_hat[np.argmin(error_cal)], list_alpha_update[np.argmin(error_cal)]
					#return Pa_unc, Pa_hat, 0.0
			else:
				# Si ya se cumplio detenemos la calibracion
				print("*** Deteniendo calibracion ***")
				print('Calibrando a alpha_update: ', alpha_update)
				return Pa_unc, Pa_hat, alpha_update
			
		# --------------------------------------------------------------------------
		#break # Parar iteraciones de calibracion
	print("Se salio del for por numero maximo de iteraciones..")
	# Para el caso donde el gt no pertenezca al HDR
	#return Pa_unc, list_Pa_hat[np.argmin(error_cal)], list_alpha_update[np.argmin(error_cal)]
	return Pa_unc, 0.0, alpha_update

def get_alpha_calibration_bs(predictions_calibration, groundtruth_calibration, alpha, alpha_step=0.01, join_positions=False):

        # Inicializamos la variable iterativa
        alpha_update = alpha
        error_cal = []
        list_alpha_update = []
        list_Pa_hat = []
                
        # Iteramos el ciclo de calibracion
        for it in range(int(1/alpha_step)):
                
                # --------------------------------------------------------------------------
                pa = []
                # Recorremos todas las trayectorias del dataset
                for i in range(predictions_calibration.shape[1]):
                                
                        # Procesamos de acuerdo a la dimension de los datos a considerar
                        if join_positions:
                                # Consideramos una dimensión de num_positions*axis_coord
                                dim1, dim2, dim3, dim4 = predictions_calibration.shape
                                X = predictions_calibration.reshape(dim1,dim2,dim3*dim4)
                                y = groundtruth_calibration.reshape(dim2,dim3*dim4)
                                #print("New dim for dbscan: ", X.shape)
                                        
                                # Clusterizamos con dbscan
                                #comp, core_samples_mask, hdr_samples_mask = calib_dbscan(X[:,i,:], gt=y[i,:], alpha=alpha_update)
                                comp, core_samples_mask, hdr_samples_mask = calib_dbscan_binsearch(X[:,i,:], gt=y[i,:], alpha=alpha_update)
                        else:
                                # Consideramos una posición de 2 dimensiones para una posicion de la trayectoria
                                # Clusterizamos con dbscan
                                #comp, core_samples_mask, hdr_samples_mask = calib_dbscan(predictions_calibration[:,i,-1,:], gt=groundtruth_calibration[i,-1,:], alpha=alpha_update)
                                comp, core_samples_mask, hdr_samples_mask = calib_dbscan_binsearch(predictions_calibration[:,i,-1,:], gt=groundtruth_calibration[i,-1,:], alpha=alpha_update)
			
			# Guardamos el resultado de comparar el gt con el HDR
                        pa.append(comp)

		# Estimamos la probabilidad empirica
                Pa_hat = np.mean(pa)
                error_cal.append(np.abs(Pa_hat-alpha))
                list_Pa_hat.append(Pa_hat)
                list_alpha_update.append(alpha_update)

                print("-------->")
                print(pa)
                print(it, alpha, Pa_hat, np.sum(pa), len(pa), alpha_step, alpha_update)
                        
		# Verificamos el tipo de estimacion de la probabilidad empirica
                if it == 0:
                        Pa_unc = Pa_hat
                                
                        if Pa_hat == alpha:
                                print("*** Deteniendo calibracion ***")
                                print('El alpha ya estaba calibrado: ', alpha_update)
                                return Pa_unc, Pa_hat, alpha_update
                                
                        elif Pa_hat < alpha:
                                band = 'under'
                                a_min = alpha_update
                                a_max = 0.99
                                alpha_update = 0.5*(a_min+a_max)
                        else:
                                band = 'over'
                                a_min = 0.01
                                a_max = alpha_update
                                alpha_update = 0.5*(a_min+a_max)
                else:
                        print('error: ', np.abs(Pa_hat-alpha), a_min, a_max)
                        if np.abs(Pa_hat-alpha) < 0.001 or a_max-a_min< 0.001:
                                print("*** Deteniendo calibracion ***")
                                print('Calibrando a alpha_update: ', alpha_update)
                                return Pa_unc, Pa_hat, alpha_update
                        elif Pa_hat < alpha:
                                a_min = alpha_update
                        else:
                                a_max = alpha_update

                # Actualizamos la variable
                alpha_update = 0.5*(a_min+a_max)

        print("!!!Saliendo del ciclo....")



def load_data_synthetic():
	# Cargamos los datos sinteticos
	data = np.load('scripts/one_sample.npz')
	
	# Preprocesamos
	predictions_calibration = np.expand_dims(  data['preds']        , axis=1)
	observations_calibration = np.expand_dims( data['obsvs'].mean(0), axis=0)
	groundtruth_calibration = np.expand_dims(  data['preds'].mean(0), axis=0)
	
	# Regresamos las trayectorias
	return predictions_calibration, observations_calibration, groundtruth_calibration, predictions_calibration, observations_calibration, groundtruth_calibration

def get_curve_calibration(predictions_calibration, groundtruth_calibration, alpha_only=None, alpha_step=0.01, join_positions=False):

	if not alpha_only is None:
		conf_levels = [alpha_only]
	else:
		# Perform calibration for alpha values in the range [0,1]
		step        = 0.05
		conf_levels = np.arange(start=0.0, stop=1.0+step, step=step)
	
	unc_pcts = []
	cal_pcts = []
	alpha_cal = []
	# Recorremos cada nivel de confidencia alpha
	#for alpha in conf_levels:
	for alpha in conf_levels:
		print("------------------------------")
		print("Calibrando alpha: ", alpha)
		
		# Calibramos con el alpha inicial
		Pa_unc, Pa_cal, alpha_new = get_alpha_calibration_bs(predictions_calibration, groundtruth_calibration, alpha, alpha_step=alpha_step, join_positions=join_positions)
		
		cal_pcts.append(Pa_cal)
		unc_pcts.append(Pa_unc)
		alpha_cal.append(alpha_new)
	
	return conf_levels, unc_pcts, cal_pcts, alpha_cal

def get_curve_calibration_test(predictions_test, groundtruth_test, conf_levels, alpha_cal, join_positions=False):

	unc_pcts_test = []
	cal_pcts_test = []
	# Recorremos cada nivel de confidencia alpha
	for k in range(len(conf_levels)):
		print("------------------------------")
		print("TEST / Calibrando alpha: ", conf_levels[k])
		
		# Calibramos
		pa_unc = []
		pa_cal = []
		# Recorremos todas las trayectorias del dataset
		for i in range(predictions_test.shape[1]):
				
			#print("**- ", k, i, conf_levels[k], alpha_cal[k])
			# Procesamos de acuerdo a la dimension de los datos a considerar
			if join_positions:
				# Consideramos una dimensión de num_positions*axis_coord
				dim1, dim2, dim3, dim4 = predictions_test.shape
				X = predictions_test.reshape(dim1,dim2,dim3*dim4)
				y = groundtruth_test.reshape(dim2,dim3*dim4)
				#print("New dim for dbscan: ", X.shape)
					
				# Clusterizamos con dbscan
				#comp_unc, _, _ = calib_dbscan(X[:,i,:], gt=y[i,:], alpha=conf_levels[k])
				#comp_cal, _, _ = calib_dbscan(X[:,i,:], gt=y[i,:], alpha=alpha_cal[k])
				comp_unc, _, _ = calib_dbscan_binsearch(X[:,i,:], gt=y[i,:], alpha=conf_levels[k])
				comp_cal, _, _ = calib_dbscan_binsearch(X[:,i,:], gt=y[i,:], alpha=alpha_cal[k])
			else:
				# Consideramos una posición de 2 dimensiones para una posicion de la trayectoria
				# Clusterizamos con dbscan
				#comp_unc, _, _ = calib_dbscan(predictions_test[:,i,-1,:], gt=groundtruth_test[i,-1,:], alpha=conf_levels[k])
				#comp_cal, _, _ = calib_dbscan(predictions_test[:,i,-1,:], gt=groundtruth_test[i,-1,:], alpha=alpha_cal[k])
				comp_unc, _, _ = calib_dbscan_binsearch(predictions_test[:,i,-1,:], gt=groundtruth_test[i,-1,:], alpha=conf_levels[k])
				comp_cal, _, _ = calib_dbscan_binsearch(predictions_test[:,i,-1,:], gt=groundtruth_test[i,-1,:], alpha=alpha_cal[k])
					
			# Guardamos el resultado de comparar el gt con el HDR
			pa_unc.append(comp_unc)
			pa_cal.append(comp_cal)
					
		# Estimamos la probabilidad empirica
		cal_pcts_test.append(np.mean(pa_cal))
		unc_pcts_test.append(np.mean(pa_unc))

	return unc_pcts_test, cal_pcts_test
	
def compute_calibration_synthetic(join_positions=False, alpha_step=0.01):

	# Cargamos los datos sinteticos
	predictions_calibration, observations_calibration, groundtruth_calibration, predictions_test, observations_test, groundtruth_test = load_data_synthetic()
	
	print(predictions_calibration.shape)
	print(predictions_test.shape)
	print(observations_calibration.shape)
	print(observations_test.shape)
	print(groundtruth_calibration.shape)
	print(groundtruth_test.shape)

	# Obtenemos los vectores para generar la curva de calibracion
	#conf_levels, unc_pcts, cal_pcts, alpha_cal = get_curve_calibration(predictions_calibration, groundtruth_calibration, alpha_only=0.85, alpha_step=alpha_step, join_positions=join_positions)
	conf_levels, unc_pcts, cal_pcts, alpha_cal = get_curve_calibration(predictions_calibration, groundtruth_calibration, alpha_step=alpha_step, join_positions=join_positions)
	
	
	print(conf_levels)
	print(cal_pcts)
	print(unc_pcts)
	print(alpha_cal)
	
	# Save plot_calibration_curves
	output_dirs   = Output_directories()
	output_image_name = os.path.join(output_dirs.confidence, "confidence_level_method_bitrap.pdf")
	plot_calibration_curves2(conf_levels, unc_pcts, cal_pcts, output_image_name)
	
			
			
	

def compute_calibration_database(join_positions=False, alpha_step=0.01):
	"""
	Evaluation of calibration metrics and calibration methods
	"""
	test_name = get_test_name()
	logging.info('Uncertainty calibration with '+SUBDATASETS_NAMES[args.id_dataset][args.id_test]+' as test dataset')
	# Load data for calibration compute
	#predictions_calibration,predictions_test,observations_calibration,observations_test,groundtruth_calibration, groundtruth_test,__,__,sigmas_samples,sigmas_samples_full,id_test = get_data_for_calibration(test_name)
	predictions_calibration, predictions_test, observations_calibration, observations_test, groundtruth_calibration, groundtruth_test, sigmas_samples, sigmas_samples_full, id_test = get_data_for_calibration(test_name)

	print(predictions_calibration.shape)
	print(predictions_test.shape)
	print(observations_calibration.shape)
	print(observations_test.shape)
	print(groundtruth_calibration.shape)
	print(groundtruth_test.shape)
	#print(sigmas_samples.shape)
	#print(sigmas_samples_full.shape)
	print(id_test)

	#print(predictions_calibration[:,0,-1,:])
	
	#for i in range(predictions_test.shape[1]):
	#	#print(i, "------------------")
	#	calib_dbscan(predictions_test[:,0,-1,:])
	
	# Obtenemos los vectores para generar la curva de calibracion
	conf_levels, unc_pcts, cal_pcts, alpha_cal = get_curve_calibration(predictions_calibration, groundtruth_calibration, alpha_only=0.05, alpha_step=0.01, join_positions=join_positions)
	#conf_levels, unc_pcts, cal_pcts, alpha_cal = get_curve_calibration(predictions_calibration, groundtruth_calibration, alpha_step=0.01, join_positions=join_positions)
	 
	print(conf_levels)
	print(cal_pcts)
	print(unc_pcts)
	print(alpha_cal)

	# Calibramos el conjunto de test
	unc_pcts_test, cal_pcts_test = get_curve_calibration_test(predictions_test, groundtruth_test, conf_levels, alpha_cal, join_positions=join_positions)
	
	print(cal_pcts_test)
	print(unc_pcts_test)
	
	metrics_cal  = [["","MACE","RMSCE","MA"]]
	metrics_test = [["","MACE","RMSCE","MA"]]

	conf_levels = np.array(conf_levels)
	unc_pcts = np.array(unc_pcts)
	cal_pcts = np.array(cal_pcts)
	unc_pcts_test = np.array(unc_pcts_test)
	cal_pcts_test = np.array(cal_pcts_test)
	
	# Evaluate metrics before/after calibration
	compute_calibration_metrics(conf_levels, unc_pcts, metrics_cal, 11, "Before Recalibration")
	compute_calibration_metrics(conf_levels, cal_pcts, metrics_cal, 11, "After  Recalibration")
	compute_calibration_metrics(conf_levels, unc_pcts_test, metrics_test, 11, "Before Recalibration")
	compute_calibration_metrics(conf_levels, cal_pcts_test, metrics_test, 11, "After  Recalibration")
	
	# Save plot_calibration_curves
	output_dirs   = Output_directories()
	output_image_name_cal =  os.path.join(output_dirs.confidence, "confidence_level_method_bitrap_cal.pdf")
	output_image_name_test = os.path.join(output_dirs.confidence, "confidence_level_method_bitrap_test.pdf")
	plot_calibration_curves2(conf_levels, unc_pcts, cal_pcts, output_image_name_cal)
	plot_calibration_curves2(conf_levels, unc_pcts_test, cal_pcts_test, output_image_name_test)
	
	#--------------------- Guardamos las metricas de calibracion ---------------------------------
	save_metrics('dbscan', metrics_cal, metrics_test, 0, output_dirs)


if __name__ == "__main__":
	# Choose seed
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)
	np.random.seed(args.seed)
	random.seed(args.seed)
	# Loggin format
	logging.basicConfig(format='%(levelname)s: %(message)s',level=args.log_level)
	compute_calibration_database()
	#compute_calibration_synthetic(join_positions=False)
	#compute_calibration_synthetic(join_positions=True)



