import torch
import torch.nn as nn
import torch.nn.functional as F

# A simple encoder-decoder network for HTP
class lstm_encdec(nn.Module):
    def __init__(self, in_size, embedding_dim, hidden_dim, output_size):
        super(lstm_encdec, self).__init__()
        # Layers
        self.embedding = nn.Linear(in_size, embedding_dim)
        self.lstm1     = nn.LSTM(embedding_dim, hidden_dim)
        self.lstm2     = nn.LSTM(embedding_dim, hidden_dim)
        self.decoder   = nn.Linear(hidden_dim, output_size)
        # Loss function
        self.loss_fun  = nn.MSELoss()

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
    def decode(self, x_last, hidden_state):
        # Embedding last position
        emb_last = self.embedding(x_last)
        # lstm for last position with hidden states from batch
        lstm_out, hidden_state = self.lstm2(emb_last.permute(1,0,2), hidden_state)
        # Decoder and Prediction
        dec = self.decoder(hidden_state[0].permute(1,0,2))
        t_pred = dec + x_last
        return t_pred,hidden_state

    def forward(self,obs_displs,target_displs,obs_abs,target_abs,teacher_forcing=False):
        # Encode the past trajectory (sequence of displacements)
        last_displ,hidden_state = self.encode(obs_displs)

        loss        = 0
        pred_displs = []

        # Decode the future trajectories
        for i, target_displ in enumerate(target_displs.permute(1,0,2)):
            # Decode last displacement and hidden state into new displacement
            pred_displ,hidden_state = self.decode(last_displ,hidden_state)
            # Keep displacement and variance on displacement
            pred_displs.append(pred_displ)
            # Update the last position
            if teacher_forcing:
                # With teacher forcing, use the GT displacement
                last_displ = target_displ.view(len(target_displ), 1, -1)
            else:
                # Otherwise, use the predicted displacement we just did
                last_displ = pred_displ
            # Deduce absolute position by summing all our predicted displacements to
            # the last absolute position
            pred_abs = obs_abs[:,-1,:] + torch.cat(pred_displs, dim=1).sum(1)
            # Evaluate likelihood
            loss += self.loss_fun(target_abs[:,i,:],pred_abs)
        # Return total loss
        return loss

    def predict(self, obs_displs, dim_pred= 1):
        # Encode the past trajectory
        last_displ,hidden_state = self.encode(obs_displs)

        pred_displs  = []

        for i in range(dim_pred):
            # Decode last position and hidden state into new position
            pred_displ,hidden_state = self.decode(last_displ,hidden_state)
            # Keep new displacement and the corresponding variance
            pred_displs.append(pred_displ)
            # Update the last displacement
            last_displ = pred_displ

        # Sum the displacements and the variances to get the relative trajectory
        pred_traj = torch.cumsum(torch.cat(pred_displs, dim=1), dim=1).detach().cpu().numpy()
        return pred_traj
