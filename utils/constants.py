# Dataset keys
OBS_TRAJ       = 'obs_traj'
OBS_TRAJ_VEL   = 'obs_traj_vel'
OBS_TRAJ_ACC   = 'obs_traj_acc'
OBS_TRAJ_THETA = 'obs_traj_theta'
OBS_NEIGHBORS  = 'obs_neighbors'
PRED_TRAJ      = 'pred_traj'
PRED_TRAJ_VEL  = 'pred_traj_vel'
PRED_TRAJ_ACC  = 'pred_traj_acc'
FRAMES_IDS     = 'frames_ids'
KEY_IDX        = 'key_idx'

# Strings used in dataset filenames
DATASETS_DIR        = ['datasets/','datasets/sdd/sdd_data']
SUBDATASETS_NAMES   = [['eth-hotel','eth-univ','ucy-zara01','ucy-zara02','ucy-univ'],
 					   ['bookstore', 'coupa', 'deathCircle', 'gates', 'hyang', 'little', 'nexus', 'quad']]
TRAIN_DATA_STR = '/training_data_'
TEST_DATA_STR  = '/test_data_'
VAL_DATA_STR   = '/validation_data_'

# test names strings used for ETH and SDD calibration
DETERMINISTIC_GAUSSIAN = 'deterministic_gaussian'
ENSEMBLES = 'ensembles'
DROPOUT = 'dropout'
BITRAP_BT = 'bitrap_BT'
VARIATIONAL = 'variational'

DETERMINISTIC_GAUSSIAN_SDD = 'deterministic_gaussian_sdd'
ENSEMBLES_SDD = 'ensembles_sdd'
DROPOUT_SDD = 'dropout_sdd'
BITRAP_BT_SDD = 'bitrap_BT_sdd'
VARIATIONAL_SDD = 'variational_sdd'

# Training checkpoints dir
TRAINING_CKPT_DIR = 'training_checkpoints'
# Pickle dir
PICKLE_DIR = 'pickle'
# Images dir
IMAGES_DIR = 'images'
# Images and data helpers
REFERENCE_IMG = 'reference.png'
MUN_POS_CSV   = 'mundo/mun_pos.csv'


CALIBRATION_CONFORMAL_FVAL = 2
CALIBRATION_CONFORMAL_FREL = 3
