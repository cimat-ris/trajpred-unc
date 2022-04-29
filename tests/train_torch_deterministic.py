#!/usr/bin/env python
# coding: utf-8

# # Inicio de Código

# In[1]:


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
from models.lstm_encdec import lstm_encdec
from utils.datasets_utils import Experiment_Parameters, setup_loo_experiment, traj_dataset
from utils.plot_utils import plot_traj_img

# Local constants
from utils.constants import OBS_TRAJ_REL, PRED_TRAJ_REL, OBS_TRAJ, PRED_TRAJ, TRAINING_CKPT_DIR


# In[2]:


logging.basicConfig(format='%(levelname)s: %(message)s',level=20)
# GPU
if torch.cuda.is_available():
    logging.info(torch.cuda.get_device_name(torch.cuda.current_device()))


# In[3]:


# Load the default parameters
experiment_parameters = Experiment_Parameters(add_kp=False,obstacles=False)

dataset_dir   = "../datasets/"
dataset_names = ['eth-hotel','eth-univ','ucy-zara01','ucy-zara02','ucy-univ']
idTest        = 2
pickle        = True

# parameters models
num_epochs     = 10
initial_lr     = 0.0003
batch_size     = 64

band_train = True
model_name = 'model_deterministic'


# In[4]:


# Load the dataset and perform the split
training_data, validation_data, test_data, test_homography = setup_loo_experiment('ETH_UCY',dataset_dir,dataset_names,idTest,experiment_parameters,pickle_dir='../pickle',use_pickled_data=pickle)


# In[5]:


# Creamos el dataset para torch
train_data = traj_dataset(training_data[OBS_TRAJ_REL ], training_data[PRED_TRAJ_REL],training_data[OBS_TRAJ], training_data[PRED_TRAJ])
val_data = traj_dataset(validation_data[OBS_TRAJ_REL ], validation_data[PRED_TRAJ_REL],validation_data[OBS_TRAJ], validation_data[PRED_TRAJ])
test_data = traj_dataset(test_data[OBS_TRAJ_REL ], test_data[PRED_TRAJ_REL], test_data[OBS_TRAJ], test_data[PRED_TRAJ])


# In[6]:


# Form batches
batched_train_data = torch.utils.data.DataLoader( train_data, batch_size = batch_size, shuffle=False)
batched_val_data =  torch.utils.data.DataLoader( val_data, batch_size = batch_size, shuffle=False)
batched_test_data =  torch.utils.data.DataLoader( test_data, batch_size = batch_size, shuffle=False)


# ## Entrenamos el modelo

# In[7]:


import torch.optim as optim

# Función para entrenar los modelos
def train(model):
    # Training the Model
    optimizer = optim.SGD(model.parameters(), lr=initial_lr)

    list_loss_train = []
    list_loss_val = []
    for epoch in range(num_epochs):
        # Training
        print("----- ")
        print("epoch: ", epoch)
        error = 0
        total = 0
        # Recorremos cada batch
        for batch_idx, (data, target, _ , _) in enumerate(batched_train_data):
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()
            if torch.cuda.is_available():
              data  = data.to(device)
              target=target.to(device)

            # Step 2. Run our forward pass and compute the loss
            pred, loss = model(data, target, training=True)
            error += loss
            total += len(target)

            # Step 3. Compute the gradients, and update the parameters by
            loss.backward()
            optimizer.step()
        print("training loss: ", error/total)
        list_loss_train.append(error.detach().cpu().numpy()/total)

        # Validation
        error = 0
        total = 0
        for batch_idx, (data_val, target_val, _ , _) in enumerate(batched_val_data):

            if torch.cuda.is_available():
              data_val  = data_val.to(device)
              target_val = target_val.to(device)

            pred_val, loss_val = model(data_val, target_val)
            error += loss_val
            total += len(target_val)

        print("Validation loss: ", error/total)
        list_loss_val.append(error.detach().cpu().numpy()/total)

    # Visualizamos los errores
    plt.figure(figsize=(12,12))
    plt.plot(list_loss_train, label="loss train")
    plt.plot(list_loss_val, label="loss val")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    


# In[8]:


# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Seleccionamos una semilla
seed = 1


# In[9]:


if band_train:
    # Agregamos la semilla
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Instanciamos el modelo
    model = lstm_encdec(2,128,256,2)
    model.to(device)

    # Entremamos el modelo
    print("\n*** Entrenando para seed: ", seed)
    train(model)

    plt.savefig("images/loss_"+str(idTest)+".pdf")
    plt.show()

    # Guardamos el Modelo
    torch.save(model.state_dict(), os.path.join(TRAINING_CKPT_DIR, model_name+"_"+str(idTest)+".pth"))
        


# ## Visualizamos las predicciones

# In[10]:


# Instanciamos el modelo
model = lstm_encdec(2,128,256,2)
model.load_state_dict(torch.load(os.path.join(TRAINING_CKPT_DIR, model_name+"_"+str(idTest)+".pth")))
model.to(device)
model.eval()


# In[11]:


ind_sample = 1
bck = plt.imread(os.path.join(dataset_dir,dataset_names[idTest],'reference.png'))

# Testing
for batch_idx, (datarel_test, targetrel_test, data_test, target_test) in enumerate(batched_test_data):
    
    if torch.cuda.is_available():
        datarel_test  = datarel_test.to(device)

    # prediction
    pred = model.predict(datarel_test, dim_pred=12)

    # ploting
    plt.figure(figsize=(12,12))
    plt.imshow(bck)
    plot_traj_img(pred[ind_sample,:,:], data_test[ind_sample,:,:], target_test[ind_sample,:,:], test_homography, bck)
    plt.legend()
    plt.title('Trajectory samples')
    plt.show()
    # Solo aplicamos a un elemento del batch
    break


# In[ ]:




