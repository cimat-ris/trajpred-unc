import os, sys, time
import importlib
import torch
import numpy as np
import logging
from torch.utils.tensorboard import SummaryWriter
from trajpred_unc.utils.constants import SUBDATASETS_NAMES
from trajpred_unc.uncertainties.calibration_utils import save_data_for_uncertainty_calibration

import sys
sys.path.append('../')
sys.path.append('../SocialVAE')

from SocialVAE.social_vae import SocialVAE
from SocialVAE.data import Dataloader
from SocialVAE.utils import ADE_FDE, FPC, seed, get_rng_state, set_rng_state

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--frameskip", type=int, default=1)
parser.add_argument("--config", type=str, default=None)
parser.add_argument("--ckpt", type=str, default=None)
parser.add_argument("--device", type=str, default=None)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--no-fpc", action="store_true", default=False)
parser.add_argument("--fpc-finetune", action="store_true", default=False)
parser.add_argument('--log-level',type=int, default=20,help='Log level (default: 20)')
parser.add_argument('--id-test',
						type=str, default="zara01", metavar='N',
						help='id of the dataset to use as test in LOO (default: 0)')

def test(model, test_data, config, fpc=1):
    all_X_globals  = []
    all_pred_trajs = []
    all_gt_trajs   = []

    sys.stdout.write("\r\033[K Evaluating...{}/{}".format(0, len(test_data)))
    tic = time.time()
    model.eval()
    ADE, FDE = [], []
    batch = 0
    fpc = int(fpc) if fpc else 1
    fpc_config = "FPC: {}".format(fpc) if fpc > 1 else "w/o FPC"
    with torch.no_grad():
        # Cycling over the batches of the test dataset
        for x, y, neighbor in test_data:
            batch += 1
            sys.stdout.write("\r\033[K Evaluating...{}/{} ({}) -- time: {}s".format(
                        batch, len(test_data), fpc_config, int(time.time()-tic)))
                    
            if config.PRED_SAMPLES > 0 and fpc > 1:
                y_ = []
                for _ in range(fpc):
                    y_.append(model(x, neighbor, n_predictions=config.PRED_SAMPLES))
                y_ = torch.cat(y_, 0)
                cand = []
                for i in range(y_.size(-2)):
                    cand.append(FPC(y_[..., i, :].cpu().numpy(), n_samples=config.PRED_SAMPLES))
                    # n_samples x PRED_HORIZON x N x 2
                y_ = torch.stack([y_[_,:,i] for i, _ in enumerate(cand)], 2)
            else:
                # n_samples x PRED_HORIZON x N x 2
                # Predict
                y_ = model(x, neighbor[:config.OB_HORIZON,:], n_predictions=config.PRED_SAMPLES)
            all_X_globals.append(x.cpu().numpy())
            all_pred_trajs.append(y_.cpu().numpy())
            all_gt_trajs.append(y.cpu().numpy())
            ade, fde = ADE_FDE(y_, y)
            if config.PRED_SAMPLES > 0:
                ade = torch.min(ade, dim=0)[0]
                fde = torch.min(fde, dim=0)[0]
            ADE.append(ade)
            FDE.append(fde)
        ADE = torch.cat(ADE)
        FDE = torch.cat(FDE)
        if torch.is_tensor(config.WORLD_SCALE) or config.WORLD_SCALE != 1:
            if not torch.is_tensor(config.WORLD_SCALE):
                config.WORLD_SCALE = torch.as_tensor(config.WORLD_SCALE, device=ADE.device, dtype=ADE.dtype)
            ADE *= config.WORLD_SCALE
            FDE *= config.WORLD_SCALE
        ade = ADE.mean()
        fde = FDE.mean()
        sys.stdout.write("\r\033[minADE: {:.4f}; minFDE: {:.4f} ({}) -- time: {}s".format(
                ade, fde, fpc_config, 
                int(time.time()-tic)))
        all_X_globals  = np.concatenate(all_X_globals, axis=1)
        all_pred_trajs = np.concatenate(all_pred_trajs, axis=2)
        all_gt_trajs   = np.concatenate(all_gt_trajs, axis=1)
        return all_X_globals,all_pred_trajs,all_gt_trajs
    
def main():
    # Loggin format
    logging.basicConfig(format='%(levelname)s: %(message)s',level=20)
    args = parser.parse_args()
    args.config = "../SocialVAE/config/"+args.id_test+".py"
    args.test = ["../SocialVAE/data/zara01/test"]
    args.ckpt = "../SocialVAE/models/zara01/"
    spec     = importlib.util.spec_from_file_location("config", args.config)
    config   = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = torch.device(args.device)
    
    seed(args.seed)
    init_rng_state = get_rng_state(args.device)
    rng_state      = init_rng_state

    ###############################################################################
    #####                                                                    ######
    ##### prepare datasets                                                   ######
    #####                                                                    ######
    ###############################################################################
    kwargs = dict(
            batch_first=False, frameskip=args.frameskip,
            ob_horizon=config.OB_HORIZON, pred_horizon=config.PRED_HORIZON,
            device=args.device, seed=args.seed)
    train_data, test_data = None, None

    if config.INCLUSIVE_GROUPS is not None:
        inclusive = [config.INCLUSIVE_GROUPS for _ in range(len(args.test))]
    else:
        inclusive = None
    test_dataset = Dataloader(
            args.test, **kwargs, inclusive_groups=inclusive,
            batch_size=config.BATCH_SIZE, shuffle=False,
    )
    test_data = torch.utils.data.DataLoader(test_dataset, 
            collate_fn=test_dataset.collate_fn,
            batch_sampler=test_dataset.batch_sampler
    )

    ###############################################################################
    #####                                                                    ######
    ##### load model                                                         ######
    #####                                                                    ######
    ###############################################################################
    logging.info("Loading model")
    model = SocialVAE(horizon=config.PRED_HORIZON, ob_radius=config.OB_RADIUS, hidden_dim=config.RNN_HIDDEN_DIM)
    model.to(args.device)
    if args.ckpt:
        ckpt = os.path.join(args.ckpt, "ckpt-best")
        logging.info("Load from ckpt: {}".format(ckpt))
        if os.path.exists(ckpt):
            state_dict = torch.load(ckpt, map_location=args.device)
            model.load_state_dict(state_dict["model"])

    ##############################################################################
    #####                                                                    ######
    ##### test                                                               ######
    #####                                                                    ######
    ###############################################################################
    fpc = False
    set_rng_state(init_rng_state, args.device)
    Xs, Ypreds, Ys = test(model, test_data, config, fpc)
    Xs     = np.swapaxes(Xs, 0, 1)
    Ys     = np.swapaxes(Ys, 0, 1)
    Ypreds = np.swapaxes(Ypreds, 0, 2)
    Ypreds = np.swapaxes(Ypreds, 1, 2)

    pickle_filename = "socialvae_"+args.id_test
    save_data_for_uncertainty_calibration(pickle_filename,Ypreds,Xs,Ys,None,args.id_test)

if __name__ == '__main__':
	main()
