import torch
import torch.nn as nn
import torch.nn.functional as F

from bayesian_torch.layers import LinearReparameterization
from bayesian_torch.layers import LSTMReparameterization

import numpy as np

def Gaussian2DLikelihood(means, targets, sigmas):
    '''
    Computes the likelihood of predicted locations under a bivariate Gaussian distribution
    params:
    outputs: Torch variable containing tensor of shape [128, 12, 2]
    targets: Torch variable containing tensor of shape [128, 12, 2]
    sigmas:  Torch variable containing tensor of shape [128, 12, 3]
    '''

    # Extract mean, std devs and correlation
    mux, muy, sx, sy, corr = means[:, 0], means[:, 1], sigmas[:, 0], sigmas[:, 1], sigmas[:, 2]

    # Exponential to get a positive value for std dev
    sx = torch.exp(sx)
    sy = torch.exp(sy)
    # tanh to get a value between [-1, 1] for correlation
    corr = torch.tanh(corr)

    # Compute factors
    normx = targets[:, 0] - mux
    normy = targets[:, 1] - muy
    sxsy = sx * sy
    z = torch.pow((normx/sx), 2) + torch.pow((normy/sy), 2) - 2*((corr*normx*normy)/sxsy)
    negRho = 1 - torch.pow(corr, 2)

    # Numerator
    result = torch.exp(-z/(2*negRho))
    # Normalization factor
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

    # Final PDF calculation
    result = result / denom

    # Numerical stability
    epsilon = 1e-20
    result = -torch.log(torch.clamp(result, min=epsilon))

    # Compute the loss across all frames and all nodes
    loss = result.sum()/np.prod(result.shape)

    return(loss)

# A simple encoder-decoder network for HTP
class lstm_encdec(nn.Module):
    def __init__(self, in_size, embedding_dim, hidden_dim, output_size):
        super(lstm_encdec, self).__init__()

        # Layers
        self.embedding = nn.Linear(in_size, embedding_dim)
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim)
        self.lstm2 = nn.LSTM(embedding_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, output_size + 3) # Agregamos salidas para sigmaxx, sigmayy, sigma xy

    # Encoding of the past trajectry
    def encode(self, X):
        # Last position traj
        x_last = X[:,-1,:].view(len(X), 1, -1)
        # Embedding positions
        emb = self.embedding(X)
        # LSTM for batch [seq_len, batch, input_size]
        lstm_out, hidden_state = self.lstm1(emb.permute(1,0,2))
        return x_last,hidden_state

    # Decoding the next future position
    def decode(self, last_pos, hidden_state):
        # Embedding last position
        emb_last = self.embedding(last_pos)
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
            loss += Gaussian2DLikelihood(means_traj, target_abs[:,i,:], torch.cat(sigma_traj, dim=1).sum(1))

        # Concatenate the predictions and return
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
            sigma_traj.append(sigma_pos)
            # Update the last position
            last_pos = pred_pos
        # Concatenate the predictions and return
        return torch.cat(pred_traj, dim=1).detach().cpu().numpy(), torch.cat(sigma_traj, dim=1).detach().cpu().numpy()

class lstm_encdec_MCDropout(nn.Module):
    def __init__(self, in_size, embedding_dim, hidden_dim, output_size, dropout_rate=0.0):
        super(lstm_encdec_MCDropout, self).__init__()

        self.dropout_rate = dropout_rate

        # Layers
        self.embedding = nn.Linear(in_size, embedding_dim)
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim, dropout=self.dropout_rate)
        self.lstm2 = nn.LSTM(embedding_dim, hidden_dim, dropout=self.dropout_rate)
        self.decoder = nn.Linear(hidden_dim, output_size+3)

        #self.loss_fun = nn.CrossEntropyLoss()
        #self.loss_fun = nn.MSELoss()

    def forward(self, X, y, data_abs , target_abs, training=False):

        # Copy data
        x = X
        # Last position traj
        x_last = X[:,-1,:].view(len(x), 1, -1)

        # Layers
        emb = self.embedding(X) # encoder for batch
        # Add dropout
        emb = F.dropout(emb, p=self.dropout_rate, training=True)
        lstm_out, (hn1, cn1) = self.lstm1(emb.permute(1,0,2)) # LSTM for batch [seq_len, batch, input_size]

        loss = 0
        pred = []
        sigma = []
        for i, target in enumerate(y.permute(1,0,2)):
            emb_last = self.embedding(x_last) # encoder for last position
            emb_last = F.dropout(emb_last, p=self.dropout_rate, training=True)
            lstm_out, (hn2, cn2) = self.lstm2(emb_last.permute(1,0,2), (hn1,cn1)) # lstm for last position with hidden states from batch

            # Decoder and Prediction
            dec = self.decoder(hn2.permute(1,0,2))
            t_pred = dec[:,:,:2] + x_last
            pred.append(t_pred)
            sigma.append(dec[:,:,2:])

            # Calculate of loss
            #loss += self.loss_fun(t_pred, target.view(len(target), 1, -1))

            # Update the last position
            if training:
                x_last = target.view(len(target), 1, -1)
            else:
                x_last = t_pred
            hn1 = hn2
            cn1 = cn2

            means = data_abs[:,-1,:] + torch.cat(pred, dim=1).sum(1)
            loss += Gaussian2DLikelihood( means, target_abs[:,i,:], torch.cat(sigma, dim=1).sum(1))

        # Concatenate the predictions and return
        return loss

    def predict(self, X, dim_pred= 1):

        # Copy data
        x = X
        # Last position traj
        x_last = X[:,-1,:].view(len(x), 1, -1)

        # Layers
        emb = self.embedding(X) # encoder for batch
        # Add dropout
        emb = F.dropout(emb, p=self.dropout_rate, training=True)
        lstm_out, (hn1, cn1) = self.lstm1(emb.permute(1,0,2)) # LSTM for batch [seq_len, batch, input_size]

        loss = 0
        pred = []
        sigma = []
        for i in range(dim_pred):
            emb_last = self.embedding(x_last) # encoder for last position
            # Add dropout
            emb_last = F.dropout(emb_last, p=self.dropout_rate, training=True)
            lstm_out, (hn2, cn2) = self.lstm2(emb_last.permute(1,0,2), (hn1,cn1)) # lstm for last position with hidden states from batch

            # Decoder and Prediction
            dec = self.decoder(hn2.permute(1,0,2))
            t_pred = dec[:,:,:2] + x_last
            pred.append(t_pred)
            sigma.append(dec[:,:,2:])

            # Update the last position
            x_last = t_pred
            hn1 = hn2
            cn1 = cn2

        # Concatenate the predictions and return
        return torch.cat(pred, dim=1).detach().cpu().numpy(), torch.cat(sigma, dim=1).detach().cpu().numpy()

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
            out_features = output_size + 3, # 5
            prior_mean = prior_mu,
            prior_variance = prior_sigma,
            posterior_mu_init = posterior_mu_init,
            posterior_rho_init = posterior_rho_init,
        )
        #self.loss_fun = nn.MSELoss()
    #
    def forward(self, X, y, data_abs , target_abs, training=False, num_mc=1):

        nll_loss = 0
        output_ = []
        kl_     = []
        #
        nbatches = len(X)
        # Last position in the trajectory
        x_last     = X[:,-1,:].view(nbatches, 1, -1)
        obs_length = X.shape[1]
        # Monte Carlo iterations
        for mc_run in range(num_mc):
            kl_sum = 0
            # Layers
            emb, kl = self.embedding(X) # encoder for batch
            kl_sum += kl
            lstm_out, (hn1, cn1), kl = self.lstm1(emb)
            kl_sum += kl/obs_length

            # Iterate for each time step
            loss = 0
            pred = []
            sigma = []
            gt = []
            for i, target in enumerate(y.permute(1,0,2)):
                emb_last, kl = self.embedding(x_last) # encoder for last position
                if i==0:
                    kl_sum += kl
                lstm_out, (hn2, cn2), kl = self.lstm2(emb_last, (hn1[:,-1,:],cn1[:,-1,:]))
                if i==0:
                    kl_sum += kl

                # Decoder and Prediction
                dec, kl = self.decoder(hn2)
                if i==0:
                    kl_sum += kl
                t_pred = dec[:,:,:2] + x_last
                pred.append(t_pred)
                sigma.append(dec[:,:,2:])
                gt.append(target.view(len(target), 1, -1))

                # Update the last position
                if training:
                    x_last = target.view(len(target), 1, -1)
                else:
                    x_last = t_pred
                hn1 = hn2
                cn1 = cn2

                # Utilizamos la nueva funcion loss
                means = data_abs[:,-1,:] + torch.cat(pred, dim=1).sum(1)
                loss += Gaussian2DLikelihood( means, target_abs[:,i,:], torch.cat(sigma, dim=1).sum(1))

            # Concatenate the trajectories preds
            pred = torch.cat(pred, dim=1)
            nll_loss += loss/num_mc

            # save to list
            output_.append(pred)
            kl_.append(kl_sum)
        pred    = torch.mean(torch.stack(output_), dim=0)
        kl_loss = torch.mean(torch.stack(kl_), dim=0)
        # Calculate of nl loss
        #nll_loss = self.loss_fun(pred, y)
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
      sigma = []
      for i in range(dim_pred):
          emb_last, kl = self.embedding(x_last) # encoder for last position
          kl_sum += kl
          lstm_out, (hn2, cn2), kl = self.lstm2(emb_last, (hn1[:,-1,:],cn1[:,-1,:]))
          kl_sum += kl

          # Decoder and Prediction
          dec, kl = self.decoder(hn2)
          kl_sum += kl
          t_pred = dec[:,:,:2] + x_last
          pred.append(t_pred)
          sigma.append(dec[:,:,2:])

          # Update the last position
          x_last = t_pred
          hn1 = hn2
          cn1 = cn2

      # Concatenate the predictions and return
      return torch.cat(pred, dim=1).detach().cpu().numpy(), kl_sum, torch.cat(sigma, dim=1).detach().cpu().numpy()
