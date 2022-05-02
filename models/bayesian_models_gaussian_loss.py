import torch
import torch.nn as nn
import torch.nn.functional as F

from bayesian_torch.layers import LinearReparameterization
from bayesian_torch.layers import LSTMReparameterization

import numpy as np

class lstm_encdec_MCDropout(nn.Module):
    def __init__(self, in_size, embedding_dim, hidden_dim, output_size, dropout_rate=0.0):
        super(lstm_encdec_MCDropout, self).__init__()

        self.dropout_rate = dropout_rate

        # Layers
        self.embedding = nn.Linear(in_size, embedding_dim)
        self.lstm1     = nn.LSTM(embedding_dim, hidden_dim)
        self.lstm2     = nn.LSTM(embedding_dim, hidden_dim)
        # Added outputs for  sigmaxx, sigmayy, sigma xy
        self.decoder   = nn.Linear(hidden_dim, output_size + 3)

    # Encoding of the past trajectry
    def encode(self, X):
        # Last position traj
        x_last = X[:,-1,:].view(len(X), 1, -1)
        # Embedding positions [batch, seq_len, input_size]
        emb = self.embedding(X)
        # Add dropout
        emb = F.dropout(emb, p=self.dropout_rate, training=True)
        # LSTM for batch [seq_len, batch, input_size]
        lstm_out, hidden_state = self.lstm1(emb.permute(1,0,2))
        return x_last,hidden_state

    # Decoding the next future position
    def decode(self, last_pos, hidden_state):
        # Embedding last position
        emb_last = self.embedding(last_pos)
        # Add dropout
        emb_last = F.dropout(emb_last, p=self.dropout_rate, training=True)
        # lstm for last position with hidden states from batch
        lstm_out, hidden_state = self.lstm2(emb_last.permute(1,0,2), hidden_state)
        # Decoder and Prediction
        dec      = self.decoder(hidden_state[0].permute(1,0,2))
        pred_pos = dec[:,:,:2] + last_pos
        sigma_pos= dec[:,:,2:]
        return pred_pos,sigma_pos,hidden_state

    def forward(self, X, y, data_abs , target_abs, training=False):
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
                last_pos = target.view(len(target_pos), 1, -1)
            else:
                last_pos = pred_pos
            means_traj = data_abs[:,-1,:] + torch.cat(pred_traj, dim=1).sum(1)
            loss += Gaussian2DLikelihood(target_abs[:,i,:], means_traj, torch.cat(sigma_traj, dim=1))
        # Return total loss
        return loss

    def predict(self, X, dim_pred= 1):
        # Encode the past trajectory
        last_pos,hidden_state = self.encode(X)

        pred_traj  = []
        sigma_traj = []

        for i in range(dim_pred):
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
        pred_traj = torch.cumsum(torch.cat(pred_traj, dim=1), dim=1).detach().cpu().numpy()
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
        lstm_out, hidden_state, kl = self.lstm1(emb.permute(1,0,2))
        kl_sum += kl/obs_length
        return x_last, hidden_state, kl_sum

    # Decoding the next future position
    def decode(self, last_pos, hidden_state):
        kl_sum = 0
        # Embedding last position
        emb_last, kl = self.embedding(last_pos)
        kl_sum += kl
        # lstm for last position with hidden states from batch
        lstm_out, hidden_state, kl = self.lstm2(emb_last.permute(1,0,2), hidden_state)
        kl_sum += kl
        # Decoder and Prediction
        dec, kl  = self.decoder(hidden_state[0].permute(1,0,2))
        kl_sum += kl
        pred_pos = dec[:,:,:2] + last_pos
        sigma_pos= dec[:,:,2:]
        return pred_pos, sigma_pos, hidden_state, kl_sum

    def forward(self, X, y, data_abs , target_abs, training=False, num_mc=1):

        nll_loss = 0
        output_ = []
        kl_     = []

        # Monte Carlo iterations
        for mc_run in range(num_mc):
            kl_sum = 0
            # Encode the past trajectory
            last_pos, hidden_state, kl = self.encode(X)
            kl_sum += kl

            # Iterate for each time step
            loss       = 0
            pred_traj  = []
            sigma_traj = []

            for i, target in enumerate(y.permute(1,0,2)):
                # Decode last position and hidden state into new position
                pred_pos, sigma_pos, hidden_state, kl = self.decode(last_pos,hidden_state)
                if i==0:
                    kl_sum += kl
                # Keep new position and variance
                pred_traj.append(pred_pos)
                sigma_traj.append(sigma_pos)
                # Update the last position
                if training:
                    last_pos = target.view(len(target), 1, -1)
                else:
                    last_pos = pred_pos

                # Utilizamos la nueva funcion loss
                means_traj = data_abs[:,-1,:] + torch.cat(pred_traj, dim=1).sum(1)
                loss += Gaussian2DLikelihood(target_abs[:,i,:], means_traj, torch.cat(sigma_traj, dim=1))

            # Concatenate the trajectories preds
            pred_traj = torch.cat(pred_traj, dim=1)
            nll_loss += loss/num_mc

            # save to list
            output_.append(pred_traj)
            kl_.append(kl_sum)
        pred    = torch.mean(torch.stack(output_), dim=0)
        kl_loss = torch.mean(torch.stack(kl_), dim=0)
        # Calculate of nl loss
        #nll_loss = self.loss_fun(pred, y)
        # Concatenate the predictions and return
        return pred, nll_loss, kl_loss

    def predict(self, X, dim_pred= 1):
        kl_sum = 0
        kl_sum += kl
        # Encode the past trajectory
        last_pos, hidden_state, kl = self.encode(X)
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
        pred_traj = torch.cumsum(torch.cat(pred_traj, dim=1), dim=1).detach().cpu().numpy()
        sigma_traj= torch.cumsum(torch.cat(sigma_traj, dim=1), dim=1).detach().cpu().numpy()

        # Concatenate the predictions and return
        return pred_traj, kl_sum, sigma_traj
        #return torch.cat(pred, dim=1).detach().cpu().numpy(), kl_sum, torch.cat(sigma, dim=1).detach().cpu().numpy()
