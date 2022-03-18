import numpy as np
from matplotlib import pyplot as plt

# Image-to-world mapping
def image_to_world_xy(image_xy, H,flip=False):
    """Convert image (x, y) position to world (x, y) position.
    This function use the homography for do the transform.

    :param image_xy: polygon image (x, y) positions
    :param H: homography matrix
    :return: world (x, y) positions
    """
    image_xy  = np.array(image_xy)
    image_xy1 = np.concatenate([image_xy, np.ones((len(image_xy), 1))],axis=1)
    world_xy1 = H.dot(image_xy1.T).T
    if flip:
        world_xy1 = world_xy1[:,::-1]
        return world_xy1[:,1:] / np.expand_dims(world_xy1[:, 0], axis=1)
    else:
        return world_xy1[:, :2] / np.expand_dims(world_xy1[:, 2], axis=1)
        
def plot_traj2(pred_traj, obs_traj_gt, pred_traj_gt):

    # Convert it to absolute (starting from the last observed position)
    displacement = np.cumsum(pred_traj, axis=0)
    this_pred_out_abs = displacement + np.array([obs_traj_gt[-1].numpy()])

    obs   = obs_traj_gt
    gt    = pred_traj_gt

    gt = np.concatenate([obs[-1,:].reshape((1,2)), gt],axis=0)
    tpred   = this_pred_out_abs

    tpred = np.concatenate([obs[-1,:].reshape((1,2)), tpred],axis=0)

    label1, = plt.plot(obs[:,0],obs[:,1],"-b", linewidth=2, label="Observations")
    label2, = plt.plot(gt[:,0], gt[:,1],"-r", linewidth=2, label="Ground truth")
    label3, = plt.plot(tpred[:,0],tpred[:,1],"-g", linewidth=2, label="Prediction")

    return label1, label2, label3
    
    
def plot_traj(pred_traj, obs_traj_gt, pred_traj_gt, test_homography, background):
    homography = np.linalg.inv(test_homography)

    # Convert it to absolute (starting from the last observed position)
    displacement = np.cumsum(pred_traj, axis=0)
    this_pred_out_abs = displacement + np.array([obs_traj_gt[-1].numpy()])

    obs   = image_to_world_xy(obs_traj_gt, homography, flip=False)
    gt    = image_to_world_xy(pred_traj_gt, homography, flip=False)
    gt = np.concatenate([obs[-1,:].reshape((1,2)), gt],axis=0)
    tpred   = image_to_world_xy(this_pred_out_abs, homography, flip=False)
    tpred = np.concatenate([obs[-1,:].reshape((1,2)), tpred],axis=0)

    plt.plot(obs[:,0],obs[:,1],"-b", linewidth=2, label="Observations")
    plt.plot(gt[:,0], gt[:,1],"-r", linewidth=2, label="Ground truth")
    plt.plot(tpred[:,0],tpred[:,1],"-g", linewidth=2, label="Prediction")

