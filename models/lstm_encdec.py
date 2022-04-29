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

    def forward(self, X, y, training=False):
        # Encode the past trajectory
        last_pos,hidden_state = self.encode(X)

        loss = 0
        pred_traj = []
        # Decode the future trajectory
        for i, target_pos in enumerate(y.permute(1,0,2)):
            # Decode last position and hidden state into new position
            pred_pos, hidden_state = self.decode(last_pos,hidden_state)
            # Keep new position
            pred_traj.append(pred_pos)
            # Calculate of loss
            loss += self.loss_fun(pred_pos, target_pos.view(len(target_pos), 1, -1))
            # Update the last position
            if training:
                # Use teacher forcing
                last_pos = target_pos.view(len(target_pos), 1, -1)
            else:
                # In testing, we do not have the target
                last_pos = pred_pos
        # Concatenate the predictions and return
        return torch.cat(pred_traj, dim=1), loss

    def predict(self, X, dim_pred= 1):
        # Encode the past trajectory
        x_last,hidden_state = self.encode(X)
        pred = []
        # Decode the future trajectory
        for i in range(dim_pred):
            # Decode last position and hidden state into new position
            t_pred, hidden_state = self.decode(x_last,hidden_state)
            # Keep new position
            pred.append(t_pred)
            # Update the last position
            x_last = t_pred
        # Concatenate the predictions and return
        return torch.cat(pred, dim=1).detach().cpu().numpy()
