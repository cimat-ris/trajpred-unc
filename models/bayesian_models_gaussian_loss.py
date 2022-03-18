import torch
import torch.nn as nn
import torch.nn.functional as F

def Gaussian2DLikelihood(outputs, targets, sigmas):
    '''
    Computes the likelihood of predicted locations under a bivariate Gaussian distribution
    params:
    outputs: Torch variable containing tensor of shape [128, 12, 2]
    targets: Torch variable containing tensor of shape [128, 12, 2]
    sigmas:  Torch variable containing tensor of shape [128, 12, 3]
    '''

    # Extract mean, std devs and correlation
    mux, muy, sx, sy, corr = outputs[:, 0], outputs[:, 1], sigmas[:, 0], sigmas[:, 1], sigmas[:, 2]

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
    
    
class lstm_encdec(nn.Module):
    def __init__(self, in_size, embedding_dim, hidden_dim, output_size):
        super(lstm_encdec, self).__init__()

        # Layers
        self.embedding = nn.Linear(in_size, embedding_dim)
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim)
        self.lstm2 = nn.LSTM(embedding_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, output_size + 3) # Agregamos salidas para sigmaxx, sigmayy, sigma xy

    def forward(self, X, y, data_abs , target_abs, training=False):

        # Copy data
        x = X
        # Last position traj
        x_last = X[:,-1,:].view(len(x), 1, -1)

        # Layers
        emb = self.embedding(X) # encoder for batch
        lstm_out, (hn1, cn1) = self.lstm1(emb.permute(1,0,2)) # LSTM for batch [seq_len, batch, input_size]

        loss = 0
        pred = []
        sigma = []
        gt = []
        # Recorremos el numero de trayectorias predichas
        for i, target in enumerate(y.permute(1,0,2)):
            emb_last = self.embedding(x_last) # encoder for last position
            lstm_out, (hn2, cn2) = self.lstm2(emb_last.permute(1,0,2), (hn1,cn1)) # lstm for last position with hidden states from batch

            # Decoder and Prediction
            dec = self.decoder(hn2.permute(1,0,2))
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
        lstm_out, (hn1, cn1) = self.lstm1(emb.permute(1,0,2)) # LSTM for batch [seq_len, batch, input_size]

        loss = 0
        pred = []
        sigma = []
        for i in range(dim_pred):
            emb_last = self.embedding(x_last) # encoder for last position
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
