#!/usr/bin/env python
# coding: utf-8

# # Para ejecutar en Google Colab en Drive
# # Inicio de Código

# In[1]:


# Imports
import sys,os,logging
''' TF_CPP_MIN_LOG_LEVEL
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printeds
3 = INFO, WARNING, and ERROR messages are not printed
'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append('bayesian-torch')
sys.path.append('OF-PathPred')
import math,numpy as np
# Important imports
import matplotlib.pyplot as plt
from path_prediction.datasets_utils import setup_loo_experiment
from path_prediction.testing_utils import evaluation_minadefde,evaluation_qualitative,evaluation_attention,plot_comparisons_minadefde, get_testing_batch
from path_prediction.training_utils import Experiment_Parameters

import torch
torch.manual_seed(1)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from bayesian_torch.layers import LinearReparameterization
from bayesian_torch.layers import LSTMReparameterization


# In[12]:



class Model_Parameters(object):
    """Model parameters.
    """
    def __init__(self, add_attention=True, add_kp=False, add_social=False, output_representation='dxdy'):
        # -----------------
        # Observation/prediction lengths
        self.obs_len        = 8
        self.pred_len       = 12
        self.seq_len        = self.obs_len + self.pred_len
        self.add_kp         = add_kp
        self.add_social     = add_social
        self.add_attention  = add_attention
        self.stack_rnn_size = 2
        self.output_representation = output_representation
        self.output_var_dirs= 0
        # Key points
        self.kp_size        = 18
        # Optical flow
        self.flow_size      = 64
        # For training
        self.num_epochs     = 35
        self.batch_size     = 256  # batch size 512
        self.use_validation = True
        # Network architecture
        self.P              =   2 # Dimensions of the position vectors
        self.enc_hidden_size= 256                  # Default value in NextP
        self.dec_hidden_size= self.enc_hidden_size # Default value in NextP
        self.emb_size       = 128  # Default value in NextP
        self.dropout_rate   = 0.3 # Default value in NextP

        #self.activation_func= tf.nn.tanh
        self.multi_decoder  = False
        self.modelname      = 'gphuctl'
        self.optimizer      = 'adam'
        self.initial_lr     = 0.01
        # MC dropout
        self.is_mc_dropout         = False
        self.mc_samples            = 20


# In[5]:
logging.basicConfig(format='%(levelname)s: %(message)s',level=20)
# GPU
if torch.cuda.is_available():
    logging.info(torch.cuda.get_device_name(torch.cuda.current_device()))

# Load the default parameters
experiment_parameters = Experiment_Parameters(add_kp=False,obstacles=False)

dataset_dir   = "OF-PathPred/datasets/"
dataset_names = ['eth-hotel','eth-univ','ucy-zara01','ucy-zara02','ucy-univ']
#dataset_names = ['eth-hotel','eth-univ','ucy-zara01']


# In[6]:


# Load the dataset and perform the split
training_data, validation_data, test_data, test_homography = setup_loo_experiment('ETH_UCY',dataset_dir,dataset_names,2,experiment_parameters,use_pickled_data=False)

print('obs_traj: ',training_data['obs_traj'].shape)
print('obs_traj_rel: ',training_data['obs_traj_rel'].shape)
print('obs_traj_theta: ',training_data['obs_traj_theta'].shape)
print('pred_traj: ',training_data['pred_traj'].shape)
print('pred_traj_rel: ',training_data['pred_traj_rel'].shape)


# In[7]:


#############################################################
# Model parameters
model_parameters = Model_Parameters(add_attention=True,add_kp=experiment_parameters.add_kp,add_social=True,output_representation=experiment_parameters.output_representation)

model_parameters.num_epochs     = 2
model_parameters.output_var_dirs= 0
model_parameters.is_mc_dropout  = False
model_parameters.initial_lr     = 0.03
model_parameters.dropout_rate   = 0



# Creamos la clase para el dataset
class traj_dataset(Dataset):

    def __init__(self, Xrel_Train, Yrel_Train, X_Train, Y_Train, transform=None):
        self.Xrel_Train = Xrel_Train
        self.Yrel_Train = Yrel_Train
        self.X_Train = X_Train
        self.Y_Train = Y_Train
        self.transform = transform

    def __len__(self):
        return len(self.X_Train)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        xrel = self.Xrel_Train[idx]
        yrel = self.Yrel_Train[idx]
        x = self.X_Train[idx]
        y = self.Y_Train[idx]

        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
            xrel = self.transform(xrel)
            yrel = self.transform(yrel)

        return xrel, yrel, x, y


# In[9]:


# Creamos el dataset para torch
train_data = traj_dataset(training_data['obs_traj_rel'], training_data['pred_traj_rel'],training_data['obs_traj'], training_data['pred_traj'])
val_data = traj_dataset(validation_data['obs_traj_rel'], validation_data['pred_traj_rel'],validation_data['obs_traj'], validation_data['pred_traj'])
test_data = traj_dataset(test_data['obs_traj_rel'], test_data['pred_traj_rel'], test_data['obs_traj'], test_data['pred_traj'])


# In[10]:


# Form batches
batched_train_data = torch.utils.data.DataLoader( train_data, batch_size = model_parameters.batch_size, shuffle=True)
batched_val_data =  torch.utils.data.DataLoader( val_data, batch_size = model_parameters.batch_size, shuffle=True)
batched_test_data =  torch.utils.data.DataLoader( test_data, batch_size = model_parameters.batch_size, shuffle=True)


# In[11]:




prior_mu = 0.0
prior_sigma = 1.0
posterior_mu_init = 0.0
posterior_rho_init = -4.0
posterior_rho_init = -3.0 # 0.006715348489117967 # 0.01814992791780978 # 0.04858735157374196

len_trainset = 15481
len_valset = 1720


class LSTM_variational(nn.Module):
    def __init__(self, in_size,  embedding_dim, hidden_dim, output_size):
        super(LSTM_variational, self).__init__()

        # Linear layer
        self.embedding = LinearReparameterization(
            in_features = in_size,
            out_features = embedding_dim, # 128
            prior_mean = prior_mu,
            prior_variance = prior_sigma,
            posterior_mu_init = posterior_mu_init,
            posterior_rho_init = posterior_rho_init,
        )

        # LSTM layer encoder
        self.lstm1 = LSTMReparameterization(
            in_features = embedding_dim,
            out_features = hidden_dim, # 256
            prior_mean = prior_mu,
            prior_variance = prior_sigma,
            posterior_mu_init = posterior_mu_init,
            posterior_rho_init = posterior_rho_init,
        )

        # LSTM layer decoder
        self.lstm2 = LSTMReparameterization(
            in_features = embedding_dim,
            out_features = hidden_dim, # 256
            prior_mean = prior_mu,
            prior_variance = prior_sigma,
            posterior_mu_init = posterior_mu_init,
            posterior_rho_init = posterior_rho_init,
        )
        # Linear layer
        self.decoder = LinearReparameterization(
            in_features = hidden_dim,
            out_features = output_size, # 2
            prior_mean = prior_mu,
            prior_variance = prior_sigma,
            posterior_mu_init = posterior_mu_init,
            posterior_rho_init = posterior_rho_init,
        )
        self.loss_fun = nn.MSELoss()
    #
    def forward(self, X, y, training=False, num_mc=1):

        output_ = []
        kl_     = []
        #
        nbatches = len(X)
        # Last position in the trajectory
        x_last = X[:,-1,:].view(nbatches, 1, -1)

        # Monte Carlo iterations
        for mc_run in range(num_mc):
            kl_sum = 0
            # Layers
            emb, kl = self.embedding(X) # encoder for batch
            kl_sum += kl
            lstm_out, (hn1, cn1), kl = self.lstm1(emb)
            kl_sum += kl

            # Iterate for each time step
            pred = []
            for i, target in enumerate(y.permute(1,0,2)):
                emb_last, kl = self.embedding(x_last) # encoder for last position
                kl_sum += kl
                lstm_out, (hn2, cn2), kl = self.lstm2(emb_last, (hn1[:,-1,:],cn1[:,-1,:]))
                kl_sum += kl

                # Decoder and Prediction
                dec, kl = self.decoder(hn2)
                kl_sum += kl
                t_pred = dec + x_last
                pred.append(t_pred)

                # Update the last position
                if training:
                    x_last = target.view(len(target), 1, -1)
                    len_evaldataset = len_trainset
                else:
                    x_last = t_pred
                    len_evaldataset = len_valset
                hn1 = hn2
                cn1 = cn2

            # Concatenate the trajectories preds
            pred = torch.cat(pred, dim=1)

            # save to list
            output_.append(pred)
            kl_.append(kl_sum)

        pred    = torch.mean(torch.stack(output_), dim=0)
        kl_loss = torch.mean(torch.stack(kl_), dim=0)

        # Calculate of nl loss
        nll_loss = self.loss_fun(pred, y)
        # Concatenate the predictions and return
        return pred, nll_loss, kl_loss

    def predict(self, X, dim_pred= 1):

      # Copy data
      x = X
      # Last position traj
      x_last = X[:,-1,:].view(len(x), 1, -1)

      kl_sum = 0
      # Layers
      emb, kl = self.embedding(X) # encoder for batch
      kl_sum += kl
      lstm_out, (hn1, cn1), kl = self.lstm1(emb)
      kl_sum += kl

      # Iterate for each time step
      pred = []
      for i in range(dim_pred):
          emb_last, kl = self.embedding(x_last) # encoder for last position
          kl_sum += kl
          lstm_out, (hn2, cn2), kl = self.lstm2(emb_last, (hn1[:,-1,:],cn1[:,-1,:]))
          kl_sum += kl

          # Decoder and Prediction
          dec, kl = self.decoder(hn2)
          kl_sum += kl
          t_pred = dec + x_last
          pred.append(t_pred)

          # Update the last position
          x_last = t_pred
          hn1 = hn2
          cn1 = cn2

      # Concatenate the predictions and return
      return torch.cat(pred, dim=1).detach().cpu().numpy(), kl_sum



# In[13]:


# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTM_variational(2,128,256,2)
model.to(device)


# In[14]:


import torch.optim as optim

# Training the Model
optimizer = optim.SGD(model.parameters(), lr=model_parameters.initial_lr)

#optimizer = optim.SGD(model.parameters(), lr=0.015)
#optimizer = optim.SGD(model.parameters(), lr=0.03)
#optimizer = optim.SGD(model.parameters(), lr=0.05)
#optimizer = optim.SGD(model.parameters(), lr=0.08) #nan
#optimizer = optim.SGD(model.parameters(), lr=0.07) #nan
#optimizer = optim.SGD(model.parameters(), lr=0.06) #nan

epochs = model_parameters.num_epochs
num_mc = 10

nl_loss_ = []
kl_loss_ = []
for epoch in range(epochs):
    # Training
    print("----- ")
    print("epoch: ", epoch)
    error = 0
    total = 0
    M     = len(batched_train_data)
    for batch_idx, (data, target, _a , _b) in enumerate(batched_train_data):
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        if torch.cuda.is_available():
            data  = data.to(device)
            target=target.to(device)

        # Step 2. Run our forward pass and compute the losses
        pred, nl_loss, kl_loss = model(data, target, training=True, num_mc=num_mc)

        nl_loss_.append(nl_loss.detach().item())
        kl_loss_.append(kl_loss.detach().item())

        # TODO: Divide by the batch size
        #pi     = (2.0**(M-batch_idx))/(2.0**M-1) #  Blundell?
        loss   = nl_loss+ kl_loss
        error += loss.detach().item()
        total += len(target)

        # Step 3. Compute the gradients, and update the parameters by
        loss.backward()
        optimizer.step()
    print("Average training loss: {:.3e}".format(error/total))

    # Validation
    error = 0
    total = 0
    M     = len(batched_val_data)
    for batch_idx, (data_val, target_val, _ , _) in enumerate(batched_val_data):
        if torch.cuda.is_available():
            data_val  = data_val.to(device)
            target_val=target_val.to(device)
        pred_val, nl_loss, kl_loss = model(data_val, target_val)
        pi     = (2.0**(M-batch_idx))/(2.0**M-1) # From Blundell
        loss   = nl_loss+ pi*kl_loss
        error += loss.detach().item()
        total += len(target_val)

    print("Average validation loss: {:.3e}".format(error/total))


# In[ ]:


plt.plot(nl_loss_,"--b", label="nl_loss")
plt.plot(kl_loss_,"--r", label="kl_loss")
plt.title("Loss training")
plt.xlabel("Iteration")
plt.ylabel("loss")
plt.legend()
plt.show()


# In[ ]:


plt.plot(nl_loss_,"--b", label="nl_loss")
plt.title("Loss training")
plt.xlabel("Iteration")
plt.ylabel("loss")
plt.legend()
plt.show()


# In[ ]:


plt.plot(kl_loss_,"--r", label="kl_loss")
plt.title("Loss training")
plt.xlabel("Iteration")
plt.ylabel("loss")
plt.legend()
plt.show()


# In[ ]:


# Guardamos el Modelo
torch.save(model.state_dict(), "training_checkpoints/model_variational.pth")


# In[ ]:





# In[ ]:


def plot_traj(pred_traj, obs_traj_gt, pred_traj_gt, test_homography, background):
    print("-----")
    homography = np.linalg.inv(test_homography)

    # Convert it to absolute (starting from the last observed position)
    displacement = np.cumsum(pred_traj, axis=0)
    this_pred_out_abs = displacement + np.array([obs_traj_gt[-1].numpy()])

    obs   = image_to_world_xy(obs_traj_gt, homography, flip=False)
    gt    = image_to_world_xy(pred_traj_gt, homography, flip=False)
    gt = np.concatenate([obs[-1,:].reshape((1,2)), gt],axis=0)
    tpred   = image_to_world_xy(this_pred_out_abs, homography, flip=False)
    tpred = np.concatenate([obs[-1,:].reshape((1,2)), tpred],axis=0)

    plt.figure(figsize=(12,12))
    plt.imshow(background)
    plt.plot(obs[:,0],obs[:,1],"-b", linewidth=2, label="Observations")
    plt.plot(gt[:,0], gt[:,1],"-r", linewidth=2, label="Ground truth")
    plt.plot(tpred[:,0],tpred[:,1],"-g", linewidth=2, label="Prediction")
    plt.legend()
    plt.title('Trajectory samples')
    plt.show()


# In[ ]:


from path_prediction.obstacles import image_to_world_xy

num_samples = 30
num_monte_carlo = 20
i = 1 # sample of batch
# TODO
# bck = np.load('background.npy')

# Testing
cont = 0
for batch_idx, (datarel_test, targetrel_test, data_test, target_test) in enumerate(batched_test_data):
    print("-----")
    print(cont)
    if torch.cuda.is_available():
        datarel_test  = datarel_test.to(device)
        targetrel_test= targetrel_test.to(device)
        data_test     = data_test.to(device)
        target_test   = target_test.to(device)
    homography = np.linalg.inv(test_homography)

    obs_traj_gt  = data_test[i,:,:]
    pred_traj_gt = target_test[i,:,:]
    obs   = image_to_world_xy(obs_traj_gt.cpu(), homography, flip=False)
    gt    = image_to_world_xy(pred_traj_gt.cpu(), homography, flip=False)
    gt = np.concatenate([obs[-1,:].reshape((1,2)), gt], axis=0)

    plt.figure(figsize=(12,12))
    # plt.imshow(bck)
    plt.plot(obs[:,0],obs[:,1],"-b", linewidth=2, label="Observations")
    plt.plot(gt[:,0], gt[:,1],"-r", linewidth=2, label="Ground truth")

    # prediction
    for mc_run in range(num_monte_carlo):
        pred, kl = model.predict(datarel_test, dim_pred=12)
        # ploting
        #plot_traj(pred[i,:,:], data_test[i,:,:], target_test[i,:,:], test_homography, bck)

        pred_traj = pred[i,:,:]

        # Convert it to absolute (starting from the last observed position)
        displacement      = np.cumsum(pred_traj, axis=0)
        this_pred_out_abs = displacement + np.array([obs_traj_gt[-1].cpu().numpy()])

        tpred = image_to_world_xy(this_pred_out_abs, homography, flip=False)
        tpred = np.concatenate([obs[-1,:].reshape((1,2)), tpred], axis=0)

        if mc_run == 0:
            plt.plot(tpred[:,0],tpred[:,1],"-g", linewidth=2, label="Prediction")
        else:
            plt.plot(tpred[:,0],tpred[:,1],"-g", linewidth=2)

    plt.legend()
    plt.title('Trajectory samples')
    plt.savefig("traj_variational_1_4"+str(cont)+".pdf")
    plt.show()

    cont += 1

    if cont == num_samples:
        break


# In[ ]:





# In[ ]:





# In[ ]:
