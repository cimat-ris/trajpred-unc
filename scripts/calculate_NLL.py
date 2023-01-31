# flake8: noqa
#!/usr/bin/env python
# coding: utf-8
# Imports
import time
import sys,os,logging, argparse
''' TF_CPP_MIN_LOG_LEVEL
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printeds
3 = INFO, WARNING, and ERROR messages are not printed
'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append('../bayesian-torch')
sys.path.append('..')

import math,numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
from torchvision import transforms
import torch.optim as optim

# Local models
from models.bayesian_models_gaussian_loss import lstm_encdec_MCDropout, lstm_encdec, lstm_encdec_variational
from utils.datasets_utils import Experiment_Parameters, setup_loo_experiment, traj_dataset


# Parser arguments
parser = argparse.ArgumentParser(description='')
parser.add_argument('--batch-size', '--b',
					type=int, default=256, metavar='N',
					help='input batch size for training (default: 256)')
parser.add_argument('--epochs', '--e',
					type=int, default=100, metavar='N',
					help='number of epochs to train (default: 200)')
parser.add_argument('--id-test',
					type=int, default=2, metavar='N',
					help='id of the dataset to use as test in LOO (default: 2)')
parser.add_argument('--mc',
					type=int, default=100, metavar='N',
					help='number of elements in the ensemble (default: 100)')
parser.add_argument('--dropout-rate',
					type=int, default=0.5, metavar='N',
					help='dropout rate (default: 0.5)')
parser.add_argument('--learning-rate', '--lr',
					type=float, default=0.0004, metavar='N',
					help='learning rate of optimizer (default: 1E-3)')
parser.add_argument('--no-retrain',
					action='store_true',
					help='do not retrain the model')
parser.add_argument('--pickle',
					action='store_true',
					help='use previously made pickle files')
parser.add_argument('--show-plot', default=False,
                    action='store_true', help='show the test plots')
parser.add_argument('--plot-losses',
					action='store_true',
					help='plot losses curves after training')
parser.add_argument('--log-level',type=int, default=20,help='Log level (default: 20)')
parser.add_argument('--log-file',default='',help='Log file (default: standard output)')
args = parser.parse_args()

logging.basicConfig(format='%(levelname)s: %(message)s',level=20)
# GPU
if torch.cuda.is_available():
    logging.info(torch.cuda.get_device_name(torch.cuda.current_device()))


# Import data
batched_train_data,batched_val_data,batched_test_data,homography,reference_image = get_ethucy_dataset(args)


batch_size     = 64 #16
pickle        = False
num_ensembles = 5



# In[ ]:


# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[ ]:


# Instanciamos el modelo deterministico
model = lstm_encdec(2,128,256,2)
model.to(device)

# el modelo entrenado se carga durante las iteracciones


# In[ ]:


# Instanciamos el modelo dropout
dropout_rate = 0.6 #0.2

# Instanciamos el modelo
model_drop = lstm_encdec_MCDropout(2,128,256,2, dropout_rate = dropout_rate)
model_drop.to(device)

model_drop.load_state_dict(torch.load("../training_checkpoints/model_dropout_"+str(args.id_test)+".pth",map_location=torch.device('cpu')))
model_drop.eval()


# In[ ]:


# Instanciamos el modelo variational
prior_mu = 0.0
prior_sigma = 1.0
posterior_mu_init = 0.0
posterior_rho_init = -5 #-3.0 # 0.006715348489117967 # 0.01814992791780978 # 0.04858735157374196

# Model
model_var = lstm_encdec_variational(2,128,256,2,prior_mu,prior_sigma,posterior_mu_init,posterior_rho_init)
model_var.to(device)

model_var.load_state_dict(torch.load("../training_checkpoints/model_variational_"+str(args.id_test)+".pth",map_location=torch.device('cpu')))
#model_var.load_state_dict(torch.load("model_variational/model_variational_"+str(idTest)+".pth"))
model_var.eval()


# In[ ]:


def Gaussian2D(outputs, targets, sigmas):
    '''
    Computes the likelihood of predicted locations under a bivariate Gaussian distribution
    params:
    outputs: Torch variable containing tensor of shape [128, 12, 2]
    targets: Torch variable containing tensor of shape [128, 12, 2]
    sigmas:  Torch variable containing tensor of shape [128, 12, 3]
    '''
    #print(outputs.shape, targets.shape, sigmas.shape)

    # Extract mean, std devs and correlation
    mux, muy, sx, sy, corr = outputs[0], outputs[1], sigmas[0], sigmas[1], sigmas[2]

    # Exponential to get a positive value for std dev
    sx = np.exp(sx)
    sy = np.exp(sy)
    # tanh to get a value between [-1, 1] for correlation
    corr = np.tanh(corr)
    #mux, muy, sx, sy, corr = getCoef(outputs)s

    # Compute factors
    #normx = targets[:, :, 0] - mux
    #normy = targets[:, :, 1] - muy
    normx = targets[0] - mux
    normy = targets[1] - muy
    sxsy = sx * sy
    z = (normx/sx)**2 + (normy/sy)**2 - 2*((corr*normx*normy)/sxsy)
    negRho = 1 - corr**2

    # Numerator
    result = np.exp(-z/(2*negRho))
    # Normalization factor
    denom = 2 * np.pi * (sxsy * np.sqrt(negRho))

    # Final PDF calculation
    result = result / denom

    # Numerical stability
    epsilon = 1e-20
    return max(result, epsilon)

#    result = -torch.log(torch.clamp(result, min=epsilon))

    # Compute the loss across all frames and all nodes
#    loss = result.sum()/np.prod(result.shape)

#    return(loss)


# ### Ensembles

# In[ ]:


### Caso Train ###
import time
from path_prediction.obstacles import image_to_world_xy
from scipy.stats import multivariate_normal
import csv


nll_batch = 0
# Testing
for batch_idx, (datarel_test, targetrel_test, data_test, target_test) in enumerate(batched_train_data):

    #for ind_sample in range(data_test.shape[0]):

    list_pred = []
    list_sigmas = []
    # prediction
    for ind in range(num_ensembles):

        # Cargamos el Modelo
        model.load_state_dict(torch.load("../training_checkpoints/model_deterministic_"+str(ind)+"_"+str(idTest)+".pth",map_location=torch.device('cpu')))
        #model.load_state_dict(torch.load("model_deterministic/model_deterministic_"+str(ind)+"_"+str(idTest)+".pth"))
#        model.load_state_dict(torch.load("model_deterministic/model_deterministic_0_2.pth",map_location=torch.device('cpu')))
        model.eval()

        # Modelo a predecir
        #pred, sigmas  = model_drop.predict(datarel_test, dim_pred=12)
        #pred, kl, sigmas  = model_var.predict(datarel_test, dim_pred=12)
        pred, sigmas = model.predict(datarel_test, dim_pred=12)


        list_pred.append(pred)
        list_sigmas.append(sigmas)

    list_pred = np.array(list_pred)
    list_sigmas = np.array(list_sigmas)
    nll_i = 0
    for ind_sample in range(data_test.shape[0]):
        # Convertimos a coordenadas absolutas
        displacement = np.cumsum(list_pred[:,ind_sample,:,:], axis=0)
        sigmas_abs = np.cumsum(list_sigmas[:,ind_sample,:,:], axis=0)
        this_pred_out_abs = displacement + np.array([data_test[ind_sample,:,:][-1].numpy()])

        # Recorremos las posiciones
        for pos in range(data_test.shape[1]):
            g_pdf = []
            # Recorremos cada ensemble
            for ind in range(num_ensembles):

                g = Gaussian2D( this_pred_out_abs[ind,pos,:] , target_test[ind,pos,:].detach().numpy(), sigmas_abs[ind,pos,:])#.detach().numpy()
                g_pdf.append(g)
            nll_i += -np.log(np.mean(g_pdf))
    nll_batch += nll_i/np.prod(data_test.shape[:2])
nll = nll_batch/(batch_idx+1)

mean_nll = nll

print("kde_nll: ", mean_nll)
with open('NLL_repeticiones.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["ensemble", idTest, mean_nll])


# In[ ]:


### Caso Train ###
import time
from path_prediction.obstacles import image_to_world_xy
from scipy.stats import multivariate_normal
import csv

#from kde_nll import *

#ind_sample = 1
#bck = np.load('background.npy')
#bck = plt.imread(os.path.join(dataset_dir,dataset_names[idTest],'reference.png'))

#seed = 9
# Agregamos la semilla
#torch.manual_seed(seed)
#torch.cuda.manual_seed(seed)

nll_batch = 0
# Testing
for batch_idx, (datarel_test, targetrel_test, data_test, target_test) in enumerate(batched_train_data):

    #for ind_sample in range(data_test.shape[0]):

    list_pred = []
    list_sigmas = []
    # prediction
    for ind in range(num_ensembles):

        # Cargamos el Modelo
        model.load_state_dict(torch.load("../training_checkpoints/model_deterministic_"+str(ind)+"_"+str(idTest)+".pth",map_location=torch.device('cpu')))
        #model.load_state_dict(torch.load("model_deterministic/model_deterministic_"+str(ind)+"_"+str(idTest)+".pth"))
#        model.load_state_dict(torch.load("model_deterministic/model_deterministic_0_2.pth",map_location=torch.device('cpu')))
        model.eval()

        # Modelo a predecir
        #pred, sigmas  = model_drop.predict(datarel_test, dim_pred=12)
        #pred, kl, sigmas  = model_var.predict(datarel_test, dim_pred=12)
        pred, sigmas = model.predict(datarel_test, dim_pred=12)


        list_pred.append(pred)
        list_sigmas.append(sigmas)

    list_pred = np.array(list_pred)
    list_sigmas = np.array(list_sigmas)
    nll_i = 0
    for ind_sample in range(data_test.shape[0]):
        # Convertimos a coordenadas absolutas
        displacement = np.cumsum(list_pred[:,ind_sample,:,:], axis=0)
        sigmas_abs = np.cumsum(list_sigmas[:,ind_sample,:,:], axis=0)
        this_pred_out_abs = displacement + np.array([data_test[ind_sample,:,:][-1].numpy()])

        # Recorremos las posiciones
        for pos in range(data_test.shape[1]):
            g_pdf = []
            # Recorremos cada ensemble
            for ind in range(num_ensembles):

                g = Gaussian2D( this_pred_out_abs[ind,pos,:] , target_test[ind,pos,:].detach().numpy(), sigmas_abs[ind,pos,:])#.detach().numpy()
                g_pdf.append(g)
            nll_i += -np.log(np.mean(g_pdf))
    nll_batch += nll_i/np.prod(data_test.shape[:2])
nll = nll_batch/(batch_idx+1)

mean_nll = nll

print("kde_nll: ", mean_nll)
with open('NLL_repeticiones.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["ensemble", idTest, mean_nll])


# In[ ]:


### Caso Train ###
import time
from path_prediction.obstacles import image_to_world_xy
from scipy.stats import multivariate_normal
import csv

#from kde_nll import *

#ind_sample = 1
#bck = np.load('background.npy')
#bck = plt.imread(os.path.join(dataset_dir,dataset_names[idTest],'reference.png'))

#seed = 9
# Agregamos la semilla
#torch.manual_seed(seed)
#torch.cuda.manual_seed(seed)

nll_batch = 0
# Testing
for batch_idx, (datarel_test, targetrel_test, data_test, target_test) in enumerate(batched_train_data):

    #for ind_sample in range(data_test.shape[0]):

    list_pred = []
    list_sigmas = []
    # prediction
    for ind in range(num_ensembles):

        # Cargamos el Modelo
        model.load_state_dict(torch.load("../training_checkpoints/model_deterministic_"+str(ind)+"_"+str(idTest)+".pth",map_location=torch.device('cpu')))
        #model.load_state_dict(torch.load("model_deterministic/model_deterministic_"+str(ind)+"_"+str(idTest)+".pth"))
#        model.load_state_dict(torch.load("model_deterministic/model_deterministic_0_2.pth",map_location=torch.device('cpu')))
        model.eval()

        # Modelo a predecir
        #pred, sigmas  = model_drop.predict(datarel_test, dim_pred=12)
        #pred, kl, sigmas  = model_var.predict(datarel_test, dim_pred=12)
        pred, sigmas = model.predict(datarel_test, dim_pred=12)


        list_pred.append(pred)
        list_sigmas.append(sigmas)

    list_pred = np.array(list_pred)
    list_sigmas = np.array(list_sigmas)
    nll_i = 0
    for ind_sample in range(data_test.shape[0]):
        # Convertimos a coordenadas absolutas
        displacement = np.cumsum(list_pred[:,ind_sample,:,:], axis=0)
        sigmas_abs = np.cumsum(list_sigmas[:,ind_sample,:,:], axis=0)
        this_pred_out_abs = displacement + np.array([data_test[ind_sample,:,:][-1].numpy()])

        # Recorremos las posiciones
        for pos in range(data_test.shape[1]):
            g_pdf = []
            # Recorremos cada ensemble
            for ind in range(num_ensembles):

                g = Gaussian2D( this_pred_out_abs[ind,pos,:] , target_test[ind,pos,:].detach().numpy(), sigmas_abs[ind,pos,:])#.detach().numpy()
                g_pdf.append(g)
            nll_i += -np.log(np.mean(g_pdf))
    nll_batch += nll_i/np.prod(data_test.shape[:2])
nll = nll_batch/(batch_idx+1)

mean_nll = nll

print("kde_nll: ", mean_nll)
with open('NLL_repeticiones.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["ensemble", idTest, mean_nll])


# In[ ]:


### Caso Train ###
import time
from path_prediction.obstacles import image_to_world_xy
from scipy.stats import multivariate_normal
import csv

#from kde_nll import *

#ind_sample = 1
#bck = np.load('background.npy')
#bck = plt.imread(os.path.join(dataset_dir,dataset_names[idTest],'reference.png'))

#seed = 9
# Agregamos la semilla
#torch.manual_seed(seed)
#torch.cuda.manual_seed(seed)

nll_batch = 0
# Testing
for batch_idx, (datarel_test, targetrel_test, data_test, target_test) in enumerate(batched_train_data):

    #for ind_sample in range(data_test.shape[0]):

    list_pred = []
    list_sigmas = []
    # prediction
    for ind in range(num_ensembles):

        # Cargamos el Modelo
        model.load_state_dict(torch.load("../training_checkpoints/model_deterministic_"+str(ind)+"_"+str(idTest)+".pth",map_location=torch.device('cpu')))
        #model.load_state_dict(torch.load("model_deterministic/model_deterministic_"+str(ind)+"_"+str(idTest)+".pth"))
#        model.load_state_dict(torch.load("model_deterministic/model_deterministic_0_2.pth",map_location=torch.device('cpu')))
        model.eval()

        # Modelo a predecir
        #pred, sigmas  = model_drop.predict(datarel_test, dim_pred=12)
        #pred, kl, sigmas  = model_var.predict(datarel_test, dim_pred=12)
        pred, sigmas = model.predict(datarel_test, dim_pred=12)


        list_pred.append(pred)
        list_sigmas.append(sigmas)

    list_pred = np.array(list_pred)
    list_sigmas = np.array(list_sigmas)
    nll_i = 0
    for ind_sample in range(data_test.shape[0]):
        # Convertimos a coordenadas absolutas
        displacement = np.cumsum(list_pred[:,ind_sample,:,:], axis=0)
        sigmas_abs = np.cumsum(list_sigmas[:,ind_sample,:,:], axis=0)
        this_pred_out_abs = displacement + np.array([data_test[ind_sample,:,:][-1].numpy()])

        # Recorremos las posiciones
        for pos in range(data_test.shape[1]):
            g_pdf = []
            # Recorremos cada ensemble
            for ind in range(num_ensembles):

                g = Gaussian2D( this_pred_out_abs[ind,pos,:] , target_test[ind,pos,:].detach().numpy(), sigmas_abs[ind,pos,:])#.detach().numpy()
                g_pdf.append(g)
            nll_i += -np.log(np.mean(g_pdf))
    nll_batch += nll_i/np.prod(data_test.shape[:2])
nll = nll_batch/(batch_idx+1)

mean_nll = nll

print("kde_nll: ", mean_nll)
with open('NLL_repeticiones.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["ensemble", idTest, mean_nll])


# In[ ]:


### Caso Train ###
import time
from path_prediction.obstacles import image_to_world_xy
from scipy.stats import multivariate_normal
import csv

#from kde_nll import *

#ind_sample = 1
#bck = np.load('background.npy')
#bck = plt.imread(os.path.join(dataset_dir,dataset_names[idTest],'reference.png'))

#seed = 9
# Agregamos la semilla
#torch.manual_seed(seed)
#torch.cuda.manual_seed(seed)

nll_batch = 0
# Testing
for batch_idx, (datarel_test, targetrel_test, data_test, target_test) in enumerate(batched_train_data):

    #for ind_sample in range(data_test.shape[0]):

    list_pred = []
    list_sigmas = []
    # prediction
    for ind in range(num_ensembles):

        # Cargamos el Modelo
        model.load_state_dict(torch.load("../training_checkpoints/model_deterministic_"+str(ind)+"_"+str(idTest)+".pth",map_location=torch.device('cpu')))
        #model.load_state_dict(torch.load("model_deterministic/model_deterministic_"+str(ind)+"_"+str(idTest)+".pth"))
#        model.load_state_dict(torch.load("model_deterministic/model_deterministic_0_2.pth",map_location=torch.device('cpu')))
        model.eval()

        # Modelo a predecir
        #pred, sigmas  = model_drop.predict(datarel_test, dim_pred=12)
        #pred, kl, sigmas  = model_var.predict(datarel_test, dim_pred=12)
        pred, sigmas = model.predict(datarel_test, dim_pred=12)


        list_pred.append(pred)
        list_sigmas.append(sigmas)

    list_pred = np.array(list_pred)
    list_sigmas = np.array(list_sigmas)
    nll_i = 0
    for ind_sample in range(data_test.shape[0]):
        # Convertimos a coordenadas absolutas
        displacement = np.cumsum(list_pred[:,ind_sample,:,:], axis=0)
        sigmas_abs = np.cumsum(list_sigmas[:,ind_sample,:,:], axis=0)
        this_pred_out_abs = displacement + np.array([data_test[ind_sample,:,:][-1].numpy()])

        # Recorremos las posiciones
        for pos in range(data_test.shape[1]):
            g_pdf = []
            # Recorremos cada ensemble
            for ind in range(num_ensembles):

                g = Gaussian2D( this_pred_out_abs[ind,pos,:] , target_test[ind,pos,:].detach().numpy(), sigmas_abs[ind,pos,:])#.detach().numpy()
                g_pdf.append(g)
            nll_i += -np.log(np.mean(g_pdf))
    nll_batch += nll_i/np.prod(data_test.shape[:2])
nll = nll_batch/(batch_idx+1)

mean_nll = nll

print("kde_nll: ", mean_nll)
with open('NLL_repeticiones.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["ensemble", idTest, mean_nll])


# ### Dropout

# In[ ]:


### Caso Train ###
import time
from path_prediction.obstacles import image_to_world_xy
from scipy.stats import multivariate_normal
import csv

#from kde_nll import *

#ind_sample = 1
#bck = np.load('background.npy')
#bck = plt.imread(os.path.join(dataset_dir,dataset_names[idTest],'reference.png'))

#seed = 9
# Agregamos la semilla
#torch.manual_seed(seed)
#torch.cuda.manual_seed(seed)

nll_batch = 0
# Testing
for batch_idx, (datarel_test, targetrel_test, data_test, target_test) in enumerate(batched_train_data):

    #for ind_sample in range(data_test.shape[0]):

    list_pred = []
    list_sigmas = []
    # prediction
    for ind in range(num_ensembles):

        # Modelo a predecir
        pred, sigmas  = model_drop.predict(datarel_test, dim_pred=12)
        #pred, kl, sigmas  = model_var.predict(datarel_test, dim_pred=12)
        #pred, sigmas = model.predict(datarel_test, dim_pred=12)


        list_pred.append(pred)
        list_sigmas.append(sigmas)

    list_pred = np.array(list_pred)
    list_sigmas = np.array(list_sigmas)
    nll_i = 0
    for ind_sample in range(data_test.shape[0]):
        # Convertimos a coordenadas absolutas
        displacement = np.cumsum(list_pred[:,ind_sample,:,:], axis=0)
        sigmas_abs = np.cumsum(list_sigmas[:,ind_sample,:,:], axis=0)
        this_pred_out_abs = displacement + np.array([data_test[ind_sample,:,:][-1].numpy()])

        # Recorremos las posiciones
        for pos in range(data_test.shape[1]):
            g_pdf = []
            # Recorremos cada ensemble
            for ind in range(num_ensembles):

                g = Gaussian2D( this_pred_out_abs[ind,pos,:] , target_test[ind,pos,:].detach().numpy(), sigmas_abs[ind,pos,:])#.detach().numpy()
                g_pdf.append(g)
            nll_i += -np.log(np.mean(g_pdf))
    nll_batch += nll_i/np.prod(data_test.shape[:2])
nll = nll_batch/(batch_idx+1)

mean_nll = nll

print("kde_nll: ", mean_nll)
with open('NLL_repeticiones.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["dropout", idTest, mean_nll])


# In[ ]:


### Caso Train ###
import time
from path_prediction.obstacles import image_to_world_xy
from scipy.stats import multivariate_normal
import csv

#from kde_nll import *

#ind_sample = 1
#bck = np.load('background.npy')
#bck = plt.imread(os.path.join(dataset_dir,dataset_names[idTest],'reference.png'))

#seed = 9
# Agregamos la semilla
#torch.manual_seed(seed)
#torch.cuda.manual_seed(seed)

nll_batch = 0
# Testing
for batch_idx, (datarel_test, targetrel_test, data_test, target_test) in enumerate(batched_train_data):

    #for ind_sample in range(data_test.shape[0]):

    list_pred = []
    list_sigmas = []
    # prediction
    for ind in range(num_ensembles):

        # Modelo a predecir
        pred, sigmas  = model_drop.predict(datarel_test, dim_pred=12)
        #pred, kl, sigmas  = model_var.predict(datarel_test, dim_pred=12)
        #pred, sigmas = model.predict(datarel_test, dim_pred=12)


        list_pred.append(pred)
        list_sigmas.append(sigmas)

    list_pred = np.array(list_pred)
    list_sigmas = np.array(list_sigmas)
    nll_i = 0
    for ind_sample in range(data_test.shape[0]):
        # Convertimos a coordenadas absolutas
        displacement = np.cumsum(list_pred[:,ind_sample,:,:], axis=0)
        sigmas_abs = np.cumsum(list_sigmas[:,ind_sample,:,:], axis=0)
        this_pred_out_abs = displacement + np.array([data_test[ind_sample,:,:][-1].numpy()])

        # Recorremos las posiciones
        for pos in range(data_test.shape[1]):
            g_pdf = []
            # Recorremos cada ensemble
            for ind in range(num_ensembles):

                g = Gaussian2D( this_pred_out_abs[ind,pos,:] , target_test[ind,pos,:].detach().numpy(), sigmas_abs[ind,pos,:])#.detach().numpy()
                g_pdf.append(g)
            nll_i += -np.log(np.mean(g_pdf))
    nll_batch += nll_i/np.prod(data_test.shape[:2])
nll = nll_batch/(batch_idx+1)

mean_nll = nll

print("kde_nll: ", mean_nll)
with open('NLL_repeticiones.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["dropout", idTest, mean_nll])


# In[ ]:


### Caso Train ###
import time
from path_prediction.obstacles import image_to_world_xy
from scipy.stats import multivariate_normal
import csv

#from kde_nll import *

#ind_sample = 1
#bck = np.load('background.npy')
#bck = plt.imread(os.path.join(dataset_dir,dataset_names[idTest],'reference.png'))

#seed = 9
# Agregamos la semilla
#torch.manual_seed(seed)
#torch.cuda.manual_seed(seed)

nll_batch = 0
# Testing
for batch_idx, (datarel_test, targetrel_test, data_test, target_test) in enumerate(batched_train_data):

    #for ind_sample in range(data_test.shape[0]):

    list_pred = []
    list_sigmas = []
    # prediction
    for ind in range(num_ensembles):

        # Modelo a predecir
        pred, sigmas  = model_drop.predict(datarel_test, dim_pred=12)
        #pred, kl, sigmas  = model_var.predict(datarel_test, dim_pred=12)
        #pred, sigmas = model.predict(datarel_test, dim_pred=12)


        list_pred.append(pred)
        list_sigmas.append(sigmas)

    list_pred = np.array(list_pred)
    list_sigmas = np.array(list_sigmas)
    nll_i = 0
    for ind_sample in range(data_test.shape[0]):
        # Convertimos a coordenadas absolutas
        displacement = np.cumsum(list_pred[:,ind_sample,:,:], axis=0)
        sigmas_abs = np.cumsum(list_sigmas[:,ind_sample,:,:], axis=0)
        this_pred_out_abs = displacement + np.array([data_test[ind_sample,:,:][-1].numpy()])

        # Recorremos las posiciones
        for pos in range(data_test.shape[1]):
            g_pdf = []
            # Recorremos cada ensemble
            for ind in range(num_ensembles):

                g = Gaussian2D( this_pred_out_abs[ind,pos,:] , target_test[ind,pos,:].detach().numpy(), sigmas_abs[ind,pos,:])#.detach().numpy()
                g_pdf.append(g)
            nll_i += -np.log(np.mean(g_pdf))
    nll_batch += nll_i/np.prod(data_test.shape[:2])
nll = nll_batch/(batch_idx+1)

mean_nll = nll

print("kde_nll: ", mean_nll)
with open('NLL_repeticiones.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["dropout", idTest, mean_nll])


# In[ ]:


### Caso Train ###
import time
from path_prediction.obstacles import image_to_world_xy
from scipy.stats import multivariate_normal
import csv

#from kde_nll import *

#ind_sample = 1
#bck = np.load('background.npy')
#bck = plt.imread(os.path.join(dataset_dir,dataset_names[idTest],'reference.png'))

#seed = 9
# Agregamos la semilla
#torch.manual_seed(seed)
#torch.cuda.manual_seed(seed)

nll_batch = 0
# Testing
for batch_idx, (datarel_test, targetrel_test, data_test, target_test) in enumerate(batched_train_data):

    #for ind_sample in range(data_test.shape[0]):

    list_pred = []
    list_sigmas = []
    # prediction
    for ind in range(num_ensembles):

        # Modelo a predecir
        pred, sigmas  = model_drop.predict(datarel_test, dim_pred=12)
        #pred, kl, sigmas  = model_var.predict(datarel_test, dim_pred=12)
        #pred, sigmas = model.predict(datarel_test, dim_pred=12)


        list_pred.append(pred)
        list_sigmas.append(sigmas)

    list_pred = np.array(list_pred)
    list_sigmas = np.array(list_sigmas)
    nll_i = 0
    for ind_sample in range(data_test.shape[0]):
        # Convertimos a coordenadas absolutas
        displacement = np.cumsum(list_pred[:,ind_sample,:,:], axis=0)
        sigmas_abs = np.cumsum(list_sigmas[:,ind_sample,:,:], axis=0)
        this_pred_out_abs = displacement + np.array([data_test[ind_sample,:,:][-1].numpy()])

        # Recorremos las posiciones
        for pos in range(data_test.shape[1]):
            g_pdf = []
            # Recorremos cada ensemble
            for ind in range(num_ensembles):

                g = Gaussian2D( this_pred_out_abs[ind,pos,:] , target_test[ind,pos,:].detach().numpy(), sigmas_abs[ind,pos,:])#.detach().numpy()
                g_pdf.append(g)
            nll_i += -np.log(np.mean(g_pdf))
    nll_batch += nll_i/np.prod(data_test.shape[:2])
nll = nll_batch/(batch_idx+1)

mean_nll = nll

print("kde_nll: ", mean_nll)
with open('NLL_repeticiones.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["dropout", idTest, mean_nll])


# In[ ]:


### Caso Train ###
import time
from path_prediction.obstacles import image_to_world_xy
from scipy.stats import multivariate_normal
import csv

#from kde_nll import *

#ind_sample = 1
#bck = np.load('background.npy')
#bck = plt.imread(os.path.join(dataset_dir,dataset_names[idTest],'reference.png'))

#seed = 9
# Agregamos la semilla
#torch.manual_seed(seed)
#torch.cuda.manual_seed(seed)

nll_batch = 0
# Testing
for batch_idx, (datarel_test, targetrel_test, data_test, target_test) in enumerate(batched_train_data):

    #for ind_sample in range(data_test.shape[0]):

    list_pred = []
    list_sigmas = []
    # prediction
    for ind in range(num_ensembles):

        # Modelo a predecir
        pred, sigmas  = model_drop.predict(datarel_test, dim_pred=12)
        #pred, kl, sigmas  = model_var.predict(datarel_test, dim_pred=12)
        #pred, sigmas = model.predict(datarel_test, dim_pred=12)


        list_pred.append(pred)
        list_sigmas.append(sigmas)

    list_pred = np.array(list_pred)
    list_sigmas = np.array(list_sigmas)
    nll_i = 0
    for ind_sample in range(data_test.shape[0]):
        # Convertimos a coordenadas absolutas
        displacement = np.cumsum(list_pred[:,ind_sample,:,:], axis=0)
        sigmas_abs = np.cumsum(list_sigmas[:,ind_sample,:,:], axis=0)
        this_pred_out_abs = displacement + np.array([data_test[ind_sample,:,:][-1].numpy()])

        # Recorremos las posiciones
        for pos in range(data_test.shape[1]):
            g_pdf = []
            # Recorremos cada ensemble
            for ind in range(num_ensembles):

                g = Gaussian2D( this_pred_out_abs[ind,pos,:] , target_test[ind,pos,:].detach().numpy(), sigmas_abs[ind,pos,:])#.detach().numpy()
                g_pdf.append(g)
            nll_i += -np.log(np.mean(g_pdf))
    nll_batch += nll_i/np.prod(data_test.shape[:2])
nll = nll_batch/(batch_idx+1)

mean_nll = nll

print("kde_nll: ", mean_nll)
with open('NLL_repeticiones.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["dropout", idTest, mean_nll])


# In[ ]:





# ### Variational

# In[ ]:


### Caso Train ###
import time
from path_prediction.obstacles import image_to_world_xy
from scipy.stats import multivariate_normal
import csv

#from kde_nll import *

#ind_sample = 1
#bck = np.load('background.npy')
#bck = plt.imread(os.path.join(dataset_dir,dataset_names[idTest],'reference.png'))

#seed = 9
# Agregamos la semilla
#torch.manual_seed(seed)
#torch.cuda.manual_seed(seed)

nll_batch = 0
# Testing
for batch_idx, (datarel_test, targetrel_test, data_test, target_test) in enumerate(batched_train_data):

    #for ind_sample in range(data_test.shape[0]):

    list_pred = []
    list_sigmas = []
    # prediction
    for ind in range(num_ensembles):

        # Modelo a predecir
        #pred, sigmas  = model_drop.predict(datarel_test, dim_pred=12)
        pred, kl, sigmas  = model_var.predict(datarel_test, dim_pred=12)
        #pred, sigmas = model.predict(datarel_test, dim_pred=12)


        list_pred.append(pred)
        list_sigmas.append(sigmas)

    list_pred = np.array(list_pred)
    list_sigmas = np.array(list_sigmas)
    nll_i = 0
    for ind_sample in range(data_test.shape[0]):
        # Convertimos a coordenadas absolutas
        displacement = np.cumsum(list_pred[:,ind_sample,:,:], axis=0)
        sigmas_abs = np.cumsum(list_sigmas[:,ind_sample,:,:], axis=0)
        this_pred_out_abs = displacement + np.array([data_test[ind_sample,:,:][-1].numpy()])

        # Recorremos las posiciones
        for pos in range(data_test.shape[1]):
            g_pdf = []
            # Recorremos cada ensemble
            for ind in range(num_ensembles):

                g = Gaussian2D( this_pred_out_abs[ind,pos,:] , target_test[ind,pos,:].detach().numpy(), sigmas_abs[ind,pos,:])#.detach().numpy()
                g_pdf.append(g)
            nll_i += -np.log(np.mean(g_pdf))
    nll_batch += nll_i/np.prod(data_test.shape[:2])
nll = nll_batch/(batch_idx+1)

mean_nll = nll

print("kde_nll: ", mean_nll)
with open('NLL_repeticiones.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["variational", idTest, mean_nll])


# In[ ]:


### Caso Train ###
import time
from path_prediction.obstacles import image_to_world_xy
from scipy.stats import multivariate_normal
import csv

#from kde_nll import *

#ind_sample = 1
#bck = np.load('background.npy')
#bck = plt.imread(os.path.join(dataset_dir,dataset_names[idTest],'reference.png'))

#seed = 9
# Agregamos la semilla
#torch.manual_seed(seed)
#torch.cuda.manual_seed(seed)

nll_batch = 0
# Testing
for batch_idx, (datarel_test, targetrel_test, data_test, target_test) in enumerate(batched_train_data):

    #for ind_sample in range(data_test.shape[0]):

    list_pred = []
    list_sigmas = []
    # prediction
    for ind in range(num_ensembles):

        # Modelo a predecir
        #pred, sigmas  = model_drop.predict(datarel_test, dim_pred=12)
        pred, kl, sigmas  = model_var.predict(datarel_test, dim_pred=12)
        #pred, sigmas = model.predict(datarel_test, dim_pred=12)


        list_pred.append(pred)
        list_sigmas.append(sigmas)

    list_pred = np.array(list_pred)
    list_sigmas = np.array(list_sigmas)
    nll_i = 0
    for ind_sample in range(data_test.shape[0]):
        # Convertimos a coordenadas absolutas
        displacement = np.cumsum(list_pred[:,ind_sample,:,:], axis=0)
        sigmas_abs = np.cumsum(list_sigmas[:,ind_sample,:,:], axis=0)
        this_pred_out_abs = displacement + np.array([data_test[ind_sample,:,:][-1].numpy()])

        # Recorremos las posiciones
        for pos in range(data_test.shape[1]):
            g_pdf = []
            # Recorremos cada ensemble
            for ind in range(num_ensembles):

                g = Gaussian2D( this_pred_out_abs[ind,pos,:] , target_test[ind,pos,:].detach().numpy(), sigmas_abs[ind,pos,:])#.detach().numpy()
                g_pdf.append(g)
            nll_i += -np.log(np.mean(g_pdf))
    nll_batch += nll_i/np.prod(data_test.shape[:2])
nll = nll_batch/(batch_idx+1)

mean_nll = nll

print("kde_nll: ", mean_nll)
with open('NLL_repeticiones.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["variational", idTest, mean_nll])


# In[ ]:


### Caso Train ###
import time
from path_prediction.obstacles import image_to_world_xy
from scipy.stats import multivariate_normal
import csv

#from kde_nll import *

#ind_sample = 1
#bck = np.load('background.npy')
#bck = plt.imread(os.path.join(dataset_dir,dataset_names[idTest],'reference.png'))

#seed = 9
# Agregamos la semilla
#torch.manual_seed(seed)
#torch.cuda.manual_seed(seed)

nll_batch = 0
# Testing
for batch_idx, (datarel_test, targetrel_test, data_test, target_test) in enumerate(batched_train_data):

    #for ind_sample in range(data_test.shape[0]):

    list_pred = []
    list_sigmas = []
    # prediction
    for ind in range(num_ensembles):

        # Modelo a predecir
        #pred, sigmas  = model_drop.predict(datarel_test, dim_pred=12)
        pred, kl, sigmas  = model_var.predict(datarel_test, dim_pred=12)
        #pred, sigmas = model.predict(datarel_test, dim_pred=12)


        list_pred.append(pred)
        list_sigmas.append(sigmas)

    list_pred = np.array(list_pred)
    list_sigmas = np.array(list_sigmas)
    nll_i = 0
    for ind_sample in range(data_test.shape[0]):
        # Convertimos a coordenadas absolutas
        displacement = np.cumsum(list_pred[:,ind_sample,:,:], axis=0)
        sigmas_abs = np.cumsum(list_sigmas[:,ind_sample,:,:], axis=0)
        this_pred_out_abs = displacement + np.array([data_test[ind_sample,:,:][-1].numpy()])

        # Recorremos las posiciones
        for pos in range(data_test.shape[1]):
            g_pdf = []
            # Recorremos cada ensemble
            for ind in range(num_ensembles):

                g = Gaussian2D( this_pred_out_abs[ind,pos,:] , target_test[ind,pos,:].detach().numpy(), sigmas_abs[ind,pos,:])#.detach().numpy()
                g_pdf.append(g)
            nll_i += -np.log(np.mean(g_pdf))
    nll_batch += nll_i/np.prod(data_test.shape[:2])
nll = nll_batch/(batch_idx+1)

mean_nll = nll

print("kde_nll: ", mean_nll)
with open('NLL_repeticiones.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["variational", idTest, mean_nll])


# In[ ]:


### Caso Train ###
import time
from path_prediction.obstacles import image_to_world_xy
from scipy.stats import multivariate_normal
import csv

#from kde_nll import *

#ind_sample = 1
#bck = np.load('background.npy')
#bck = plt.imread(os.path.join(dataset_dir,dataset_names[idTest],'reference.png'))

#seed = 9
# Agregamos la semilla
#torch.manual_seed(seed)
#torch.cuda.manual_seed(seed)

nll_batch = 0
# Testing
for batch_idx, (datarel_test, targetrel_test, data_test, target_test) in enumerate(batched_train_data):

    #for ind_sample in range(data_test.shape[0]):

    list_pred = []
    list_sigmas = []
    # prediction
    for ind in range(num_ensembles):

        # Modelo a predecir
        #pred, sigmas  = model_drop.predict(datarel_test, dim_pred=12)
        pred, kl, sigmas  = model_var.predict(datarel_test, dim_pred=12)
        #pred, sigmas = model.predict(datarel_test, dim_pred=12)


        list_pred.append(pred)
        list_sigmas.append(sigmas)

    list_pred = np.array(list_pred)
    list_sigmas = np.array(list_sigmas)
    nll_i = 0
    for ind_sample in range(data_test.shape[0]):
        # Convertimos a coordenadas absolutas
        displacement = np.cumsum(list_pred[:,ind_sample,:,:], axis=0)
        sigmas_abs = np.cumsum(list_sigmas[:,ind_sample,:,:], axis=0)
        this_pred_out_abs = displacement + np.array([data_test[ind_sample,:,:][-1].numpy()])

        # Recorremos las posiciones
        for pos in range(data_test.shape[1]):
            g_pdf = []
            # Recorremos cada ensemble
            for ind in range(num_ensembles):

                g = Gaussian2D( this_pred_out_abs[ind,pos,:] , target_test[ind,pos,:].detach().numpy(), sigmas_abs[ind,pos,:])#.detach().numpy()
                g_pdf.append(g)
            nll_i += -np.log(np.mean(g_pdf))
    nll_batch += nll_i/np.prod(data_test.shape[:2])
nll = nll_batch/(batch_idx+1)

mean_nll = nll

print("kde_nll: ", mean_nll)
with open('NLL_repeticiones.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["variational", idTest, mean_nll])


# In[ ]:


### Caso Train ###
import time
from path_prediction.obstacles import image_to_world_xy
from scipy.stats import multivariate_normal
import csv

#from kde_nll import *

#ind_sample = 1
#bck = np.load('background.npy')
#bck = plt.imread(os.path.join(dataset_dir,dataset_names[idTest],'reference.png'))

#seed = 9
# Agregamos la semilla
#torch.manual_seed(seed)
#torch.cuda.manual_seed(seed)

nll_batch = 0
# Testing
for batch_idx, (datarel_test, targetrel_test, data_test, target_test) in enumerate(batched_train_data):

    #for ind_sample in range(data_test.shape[0]):

    list_pred = []
    list_sigmas = []
    # prediction
    for ind in range(num_ensembles):

        # Modelo a predecir
        #pred, sigmas  = model_drop.predict(datarel_test, dim_pred=12)
        pred, kl, sigmas  = model_var.predict(datarel_test, dim_pred=12)
        #pred, sigmas = model.predict(datarel_test, dim_pred=12)


        list_pred.append(pred)
        list_sigmas.append(sigmas)

    list_pred = np.array(list_pred)
    list_sigmas = np.array(list_sigmas)
    nll_i = 0
    for ind_sample in range(data_test.shape[0]):
        # Convertimos a coordenadas absolutas
        displacement = np.cumsum(list_pred[:,ind_sample,:,:], axis=0)
        sigmas_abs = np.cumsum(list_sigmas[:,ind_sample,:,:], axis=0)
        this_pred_out_abs = displacement + np.array([data_test[ind_sample,:,:][-1].numpy()])

        # Recorremos las posiciones
        for pos in range(data_test.shape[1]):
            g_pdf = []
            # Recorremos cada ensemble
            for ind in range(num_ensembles):

                g = Gaussian2D( this_pred_out_abs[ind,pos,:] , target_test[ind,pos,:].detach().numpy(), sigmas_abs[ind,pos,:])#.detach().numpy()
                g_pdf.append(g)
            nll_i += -np.log(np.mean(g_pdf))
    nll_batch += nll_i/np.prod(data_test.shape[:2])
nll = nll_batch/(batch_idx+1)

mean_nll = nll

print("kde_nll: ", mean_nll)
with open('NLL_repeticiones.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["variational", idTest, mean_nll])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:
