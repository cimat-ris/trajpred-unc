import torch
import torch.nn as nn
from trajpred_unc.models.losses import Gaussian2DLikelihood,convertToCov

# A simple LSTM-based encoder-decoder network for HTP
class lstm_encdec(nn.Module):
    def __init__(self,config):
        super(lstm_encdec, self).__init__()
        # Layers
        self.embedding  = nn.Linear(config["input_dim"], config["embedding_dim"])
        self.drop_past  = nn.Dropout(p=config["dropout_rate"])
        self.lstm_past  = nn.LSTM(config["input_dim"],config["hidden_dim"],num_layers=config["num_layers"])
        self.lstm_future= nn.LSTM(config["embedding_dim"],config["hidden_dim"],num_layers=config["num_layers"])
        self.drop_future= nn.Dropout(p=config["dropout_rate"])
        self.decoder    = nn.Linear(config["hidden_dim"],config["output_dim"])
        self.nl         = nn.SiLU()
        # Loss function
        self.loss_fun  = nn.MSELoss()
        self.dt        = 0.4

    # Encoding of the past trajectory data
    def encode(self, V):
        # Last spatial data
        v_last = V[:,-1:,:]
        # LSTM for batch [seq_len, batch, input_size]
        __, hidden_state = self.lstm_past(V.permute(1,0,2))
        return v_last,hidden_state

    # Decoding the next future displacements
    def decode(self, v_last, hidden_state):
        # Embedding last spatial data
        emb_last = self.nl(self.embedding(v_last))
        # lstm for last spatial data with hidden states from batch
        __, hidden_state = self.lstm_future(emb_last.permute(1,0,2), hidden_state)
        # Decoder and Prediction
        dec = self.decoder(self.drop_future(self.nl(hidden_state[0][-1:].permute(1,0,2))))
        # Model evaluates deviation to the linear model
        t_pred = dec + v_last
        return t_pred,hidden_state

    def forward(self,obs_vels,target_vels,obs_abs,target_abs,teacher_forcing=False):
        # Encode the past trajectory (sequence of velocities)
        last_vel,hidden_state = self.encode(obs_vels)
        loss      = 0
        pred_vels = []
        # Decode the future trajectories
        for i, target_vel in enumerate(target_vels.permute(1,0,2)):
            # Decode last displacement and hidden state into new displacement
            pred_vel,hidden_state = self.decode(last_vel,hidden_state)
            # Keep displacement and variance on displacement
            pred_vels.append(pred_vel)
            # Update the last position
            if teacher_forcing:
                # With teacher forcing, use the GT displacement
                last_vel = target_vel.view(len(target_vel), 1, -1)
            else:
                # Otherwise, use the predicted displacement we just did
                last_vel = pred_vel
            # Deduce absolute position by summing all our predicted displacements to
            # the last absolute position
            pred_abs = obs_abs[:,-1,:] + torch.mul(torch.cat(pred_vels, dim=1).sum(1),self.dt)
            # Evaluate likelihood
            loss += self.loss_fun(target_abs[:,i,:],pred_abs)
        # Return total loss
        return loss

    def predict(self, obs_vels, obs_pos, prediction_horizon=12):
        # Encode the past trajectory (sequence of velocities)
        last_vel,hidden_state = self.encode(obs_vels)
        pred_vels  = []
        for i in range(prediction_horizon):
            # Decode last velocity and hidden state into new position
            pred_vel,hidden_state = self.decode(last_vel,hidden_state)
            # Keep new velocity
            pred_vels.append(pred_vel)
            # Update the last velocity
            last_vel = pred_vel
        # Sum the displacements to get the relative trajectory
        return self.dt*torch.cumsum(torch.cat(pred_vels, dim=1), dim=1).detach().cpu().numpy()+obs_pos[:,-1:,:].cpu().numpy()

# A simple encoder-decoder network for HTP
class lstm_encdec_gaussian(nn.Module):
    def __init__(self,config):
        super(lstm_encdec_gaussian, self).__init__()

        # Layers
        self.embedding  = nn.Linear(config["input_dim"], config["embedding_dim"])
        self.drop_past  = nn.Dropout(p=config["dropout_rate"])
        self.lstm_past  = nn.LSTM(config["input_dim"],config["hidden_dim"],num_layers=config["num_layers"])
        self.lstm_future= nn.LSTM(config["embedding_dim"],config["hidden_dim"],num_layers=config["num_layers"])
        self.drop_future= nn.Dropout(p=config["dropout_rate"])
        self.decoder    = nn.Linear(config["hidden_dim"],config["output_dim"])
        self.nl         = nn.SiLU()

        # Added outputs for  sigmaxx, sigmayy, sigma xy
        self.decoder   = nn.Linear(config["hidden_dim"], config["output_dim"] + 3)
        self.dt        = 0.4

    # Encoding of the past trajectory
    def encode(self, V):
        # Last spatial data
        v_last = V[:,-1:,:]
        # LSTM for batch [seq_len, batch, input_size]
        __, hidden_state = self.lstm_past(V.permute(1,0,2))
        return v_last,hidden_state

    # Decoding the next spatial data
    def decode(self, v_last, hidden_state):
        # Embedding last spatial data
        emb_last = self.nl(self.embedding(v_last))
        # lstm for last spatial data with hidden states from batch
        __, hidden_state = self.lstm_future(emb_last.permute(1,0,2), hidden_state)
        # Decoder and Prediction
        dec = self.decoder(self.drop_future(self.nl(hidden_state[0][-1:].permute(1,0,2))))
        # Model evaluates deviation to the linear model
        v_pred = dec[:,:,:2] + v_last
        sigma_v= dec[:,:,2:]
        return v_pred,sigma_v,hidden_state

    def forward(self,obs_vels,target_vels,obs_abs,target_abs,teacher_forcing=False):
        # Encode the past trajectory (sequence of displacements)
        last_vel,hidden_state = self.encode(obs_vels)

        loss        = 0
        pred_vels   = []
        sigma_vels  = []

        # Decode the future trajectories
        for i, target_vel in enumerate(target_vels.permute(1,0,2)):
            # Decode last displacement and hidden state into new velocity
            pred_vel,sigma_vel,hidden_state = self.decode(last_vel,hidden_state)
            # Keep displacement and variance on velocity
            pred_vels.append(pred_vel)
            sigma_vels.append(sigma_vel)
            # Update the last position
            if teacher_forcing:
                # With teacher forcing, use the GT displacement
                last_vel = target_vel.view(len(target_vel), 1, -1)
            else:
                # Otherwise, use the predicted displacement we just did
                last_vel = pred_vel
            # Deduce absolute position by summing all our predicted displacements to
            # the last absolute position
            pred_abs = obs_abs[:,-1,:] + torch.mul(torch.cat(pred_vels, dim=1).sum(1),self.dt)
            # Evaluate likelihood
            loss += Gaussian2DLikelihood(target_abs[:,i,:], pred_abs, torch.cat(sigma_vels, dim=1),self.dt)
        # Return total loss
        return loss

    def predict(self, obs_vels, obs_pos, prediction_horizon=12):
        # Encode the past trajectory
        last_vel,hidden_state = self.encode(obs_vels)

        pred_vels  = []
        sigma_vels = []

        for i in range(prediction_horizon):
            # Decode last position and hidden state into new position
            pred_vel,sigma_vel,hidden_state = self.decode(last_vel,hidden_state)
            # Keep new displacement and the corresponding variance
            pred_vels.append(pred_vel)
            # Convert sigma_displ into real variances
            sigma_vel[:,:,0],sigma_vel[:,:,1],sigma_vel[:,:,2] = convertToCov(sigma_vel[:,:,0], sigma_vel[:,:,1], sigma_vel[:,:,2])
            sigma_vels.append(sigma_vel)
            # Update the last displacement
            last_vel = pred_vel

        # Sum the displacements and the variances to get the relative trajectory
        pred_traj = self.dt*torch.cumsum(torch.cat(pred_vels, dim=1), dim=1).detach().cpu().numpy()+obs_pos[:,-1:,:].cpu().numpy()
        sigma_traj= self.dt*self.dt*torch.cumsum(torch.cat(sigma_vels, dim=1), dim=1).detach().cpu().numpy()
        return pred_traj,sigma_traj

