import torch
import torch.nn as nn
import torch.nn.functional as F

class lstm_encdec(nn.Module):
    def __init__(self, in_size, embedding_dim, hidden_dim, output_size):
        super(lstm_encdec, self).__init__()

        # Layers
        self.embedding = nn.Linear(in_size, embedding_dim)
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim)
        self.lstm2 = nn.LSTM(embedding_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, output_size)

        #self.loss_fun = nn.CrossEntropyLoss()
        self.loss_fun = nn.MSELoss()

    def forward(self, X, y, training=False):

        # Copy data
        x = X
        # Last position traj
        x_last = X[:,-1,:].view(len(x), 1, -1)

        # Layers
        emb = self.embedding(X) # encoder for batch
        lstm_out, (hn1, cn1) = self.lstm1(emb.permute(1,0,2)) # LSTM for batch [seq_len, batch, input_size]

        loss = 0
        pred = []
        for i, target in enumerate(y.permute(1,0,2)):
            emb_last = self.embedding(x_last) # encoder for last position
            lstm_out, (hn2, cn2) = self.lstm2(emb_last.permute(1,0,2), (hn1,cn1)) # lstm for last position with hidden states from batch

            # Decoder and Prediction
            dec = self.decoder(hn2.permute(1,0,2))
            t_pred = dec + x_last
            pred.append(t_pred)

            # Calculate of loss
            loss += self.loss_fun(t_pred, target.view(len(target), 1, -1))

            # Update the last position
            if training:
                x_last = target.view(len(target), 1, -1)
            else:
                x_last = t_pred
            hn1 = hn2
            cn1 = cn2

        # Concatenate the predictions and return
        return torch.cat(pred, dim=1), loss

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
        for i in range(dim_pred):
            emb_last = self.embedding(x_last) # encoder for last position
            lstm_out, (hn2, cn2) = self.lstm2(emb_last.permute(1,0,2), (hn1,cn1)) # lstm for last position with hidden states from batch

            # Decoder and Prediction
            dec = self.decoder(hn2.permute(1,0,2))
            t_pred = dec + x_last
            pred.append(t_pred)

            # Update the last position
            x_last = t_pred
            hn1 = hn2
            cn1 = cn2

        # Concatenate the predictions and return
        return torch.cat(pred, dim=1).detach().numpy()
