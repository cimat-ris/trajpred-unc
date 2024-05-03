import torch
import torch.nn as nn
import torch.nn.functional as F

from bayesian_torch.layers import LinearReparameterization
from bayesian_torch.layers import LSTMReparameterization
from trajpred_unc.models.losses import Gaussian2DLikelihood
import numpy as np



class lstm_encdec_MCDropout(nn.Module):
    def __init__(self, config):
        super(lstm_encdec_MCDropout, self).__init__()

        self.dropout_rate = config["dropout_rate"]

        # Layers
        self.embedding  = nn.Linear(config["input_dim"], config["embedding_dim"])
        self.lstm_past  = nn.LSTM(config["embedding_dim"],config["hidden_dim"],num_layers=config["num_layers"])
        self.lstm_future= nn.LSTM(config["embedding_dim"],config["hidden_dim"],num_layers=config["num_layers"])
        # Added outputs for  sigmaxx, sigmayy, sigma xy
        self.decoder    = nn.Linear(config["hidden_dim"],config["output_dim"]+3)
        self.dt         = 0.4

    # Encoding of the past trajectry
    def encode(self, X):
        # Last position traj
        x_last = X[:,-1:,:]
        # Embedding positions [batch, seq_len, input_size]
        emb = self.embedding(X)
        # Add dropout
        emb = F.dropout(emb, p=self.dropout_rate, training=True)
        # LSTM for batch [seq_len, batch, input_size]
        __, hidden_state = self.lstm_past(emb.permute(1,0,2))
        return x_last,hidden_state

    # Decoding the next future position
    def decode(self, last_pos, hidden_state):
        # Embedding last position
        emb_last = self.embedding(last_pos)
        # Add dropout
        emb_last = F.dropout(emb_last, p=self.dropout_rate, training=True)
        # lstm for last position with hidden states from batch
        __, hidden_state = self.lstm_future(emb_last.permute(1,0,2), hidden_state)
        # Decoder and Prediction
        dec      = self.decoder(hidden_state[0][-1:].permute(1,0,2))
        pred_pos = dec[:,:,:2] + last_pos
        sigma_pos= dec[:,:,2:]
        return pred_pos,sigma_pos,hidden_state

    def forward(self, X, y, data_abs , target_abs, training=False, teacher_forcing=False):
        # Encode the past trajectory
        last_pos,hidden_state = self.encode(X)

        loss       = 0
        pred_traj  = []
        sigma_traj = []

        # Decode de future trajectories
        for i, target_pos in enumerate(y.permute(1,0,2)):
            # Decode last position and hidden state into new position
            pred_pos,sigma_pos,hidden_state = self.decode(last_pos,hidden_state)
            # Keep new position and variance
            pred_traj.append(pred_pos)
            sigma_traj.append(sigma_pos)
            # Update the last position
            if training:
                last_pos = target_pos.view(len(target_pos), 1, -1)
            else:
                last_pos = pred_pos
            means_traj = data_abs[:,-1,:] + torch.cat(pred_traj, dim=1).sum(1)
            loss += Gaussian2DLikelihood(target_abs[:,i,:], means_traj, torch.cat(sigma_traj, dim=1), self.dt)
        # Return total loss
        return loss

    def predict(self, obs_vel, obs_pos, prediction_horizon= 12):
        # Encode the past trajectory
        last_pos,hidden_state = self.encode(obs_vel)

        pred_traj  = []
        sigma_traj = []

        for i in range(prediction_horizon):
            # Decode last position and hidden state into new position
            pred_pos,sigma_pos,hidden_state = self.decode(last_pos,hidden_state)
            # Keep new position and variance
            pred_traj.append(pred_pos)
            # Convert sigma_pos into real variances
            sigma_pos[:,:,0]   = torch.exp(sigma_pos[:,:,0])+1e-2
            sigma_pos[:,:,1]   = torch.exp(sigma_pos[:,:,1])+1e-2
            sigma_traj.append(sigma_pos)
            # Update the last position
            last_pos = pred_pos

        # Concatenate the predictions and return
        pred_traj = torch.cumsum(torch.cat(pred_traj, dim=1), dim=1).detach().cpu().numpy()+obs_pos[:,-1:,:].cpu().numpy()
        sigma_traj= torch.cumsum(torch.cat(sigma_traj, dim=1), dim=1).detach().cpu().numpy()
        return pred_traj,sigma_traj

# Bayes By Backprop (BBB)
class lstm_encdec_variational(nn.Module):
    def __init__(self, in_size,  embedding_dim, hidden_dim, output_size, prior_mu, prior_sigma, posterior_mu_init, posterior_rho_init):
        super(lstm_encdec_variational, self).__init__()

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
            # Added outputs for  sigmaxx, sigmayy, sigma xy
            out_features = output_size + 3, # 5
            prior_mean = prior_mu,
            prior_variance = prior_sigma,
            posterior_mu_init = posterior_mu_init,
            posterior_rho_init = posterior_rho_init,
        )
        #self.loss_fun = nn.MSELoss()
        self.dt = 0.4

    # Encoding of the past trajectry
    def encode(self, X):
        kl_sum = 0
        obs_length = X.shape[1]
        # Last position traj
        x_last = X[:,-1,:].view(len(X), 1, -1)
        # Embedding positions [batch, seq_len, input_size]
        emb, kl = self.embedding(X)
        kl_sum += kl
        # LSTM for batch [seq_len, batch, input_size]
        lstm_out, hidden_state, kl = self.lstm1(emb)
        kl_sum += kl/obs_length
        return x_last, hidden_state, kl_sum

    # Decoding the next future position
    def decode(self, last_pos, hidden_state):
        kl_sum = 0
        # Embedding last position
        emb_last, kl = self.embedding(last_pos)
        kl_sum += kl
        # lstm for last position with hidden states from batch
        lstm_out, hidden_state, kl = self.lstm2(emb_last, (hidden_state[0][:,-1,:],hidden_state[1][:,-1,:]))
        kl_sum += kl
        # Decoder and Prediction
        dec, kl  = self.decoder(hidden_state[0])
        kl_sum += kl
        pred_pos = dec[:,:,:2] + last_pos
        sigma_pos= dec[:,:,2:]
        return pred_pos, sigma_pos, hidden_state, kl_sum

    def forward(self,obs_vels,target_vels,obs_abs,target_abs,teacher_forcing=False, num_mc=1):

        nll_loss = 0
        output_ = []
        kl_     = []

        # Monte Carlo iterations
        for mc_run in range(num_mc):
            kl_sum = 0
            # Encode the past trajectory
            last_vel, hidden_state, kl = self.encode(obs_vels)
            kl_sum += kl

            # Iterate for each time step
            loss       = 0
            pred_vels  = []
            sigma_vels = []

            for i, target_vel in enumerate(target_vels.permute(1,0,2)):
                # Decode last position and hidden state into new position
                pred_vel, sigma_vel, hidden_state, kl = self.decode(last_vel,hidden_state)
                if i==0:
                    kl_sum += kl
                # Keep new position and variance
                pred_vels.append(pred_vel)
                sigma_vels.append(sigma_vel)
                # Update the last position
                if teacher_forcing:
                    last_vel = target_vel.view(len(target_vel), 1, -1)
                else:
                    last_vel = pred_vel

                # Utilizamos la nueva funcion loss
                pred_abs = obs_abs[:,-1,:] + torch.mul(torch.cat(pred_vels, dim=1).sum(1),self.dt)
                loss += Gaussian2DLikelihood(target_abs[:,i,:], pred_abs, torch.cat(sigma_vels, dim=1),self.dt)

            # Concatenate the trajectories preds
            pred_vels = torch.cat(pred_vels, dim=1)
            nll_loss += loss/num_mc

            # save to list
            output_.append(pred_vels)
            kl_.append(kl_sum)
        pred    = torch.mean(torch.stack(output_), dim=0)
        kl_loss = torch.mean(torch.stack(kl_), dim=0)
        # Concatenate the predictions and return
        return pred, nll_loss, kl_loss

    def predict(self, obs_vels, obs_pos, dim_pred= 12):
        kl_sum = 0
        # Encode the past trajectory
        last_pos, hidden_state, kl = self.encode(obs_vels)
        kl_sum += kl

        pred_traj  = []
        sigma_traj = []

        for i in range(dim_pred):
            # Decode last position and hidden state into new position
            pred_pos, sigma_pos, hidden_state, kl = self.decode(last_pos,hidden_state)
            kl_sum += kl
            # Keep new position and variance
            pred_traj.append(pred_pos)
            # Convert sigma_pos into real variances
            sigma_pos[:,:,0]   = torch.exp(sigma_pos[:,:,0])+1e-2
            sigma_pos[:,:,1]   = torch.exp(sigma_pos[:,:,1])+1e-2
            sigma_traj.append(sigma_pos)
            # Update the last position
            last_pos = pred_pos

        # Concatenate the predictions and return
        pred_traj = self.dt*torch.cumsum(torch.cat(pred_traj, dim=1), dim=1).detach().cpu().numpy()+obs_pos[:,-1:,:].cpu().numpy()
        sigma_traj= self.dt*self.dt*torch.cumsum(torch.cat(sigma_traj, dim=1), dim=1).detach().cpu().numpy()
        # Concatenate the predictions and return
        return pred_traj, kl_sum, sigma_traj
