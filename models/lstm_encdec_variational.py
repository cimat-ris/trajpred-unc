import torch
import torch.nn as nn
import torch.nn.functional as F
from bayesian_torch.layers import LinearReparameterization
from bayesian_torch.layers import LSTMReparameterization


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
                else:
                    x_last = t_pred
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
