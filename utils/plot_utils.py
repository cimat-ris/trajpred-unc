import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse

def draw_covariance_ellipse(mean_pos, cov_pos, ax=None, color=None, n_sigmas=2):
    """
    Visualize 2D covariance matrix
    """
    ax = ax or plt.gca()
    covariance     = np.zeros((2,2))
    covariance[0,0]= cov_pos[0]
    covariance[1,1]= cov_pos[1]
    covariance[0,1]= cov_pos[2]
    covariance[1,0]= cov_pos[2]
    # Convert covariance to principal axes
    U, S, Vt = np.linalg.svd(covariance)
    angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
    semi_w, semi_h = 2 * np.sqrt(S)
    # Draw the ellipse at different sigmas
    e = Ellipse(mean_pos,n_sigmas*semi_w,n_sigmas*semi_h,angle,color=color)
    e.set_alpha(0.2)
    ax.add_patch(e)

# Image-to-world mapping
def world_to_image_xy(world_xy,homography_to_img,flip=False):
    """
    Convert image (x, y) position to world (x, y) position.
    This function use the homography for do the transform.

    :param image_xy: polygon image (x, y) positions
    :param H: homography matrix
    :return: world (x, y) positions
    """
    world_xy  = np.array(world_xy)
    world_xy1 = np.concatenate([world_xy, np.ones((len(world_xy), 1))],axis=1)
    image_xy1 = homography_to_img.dot(world_xy1.T).T
    if flip:
        image_xy1 = image_xy1[:,::-1]
        return image_xy1[:,1:] / np.expand_dims(image_xy1[:, 0], axis=1)
    else:
        return image_xy1[:, :2] / np.expand_dims(image_xy1[:, 2], axis=1)

def plot_traj_world(pred_traj, obs_traj_gt, pred_traj_gt, ax=None):
    """
    Plot a predicted trajectory in the world frame.
    First input is the trajectory expressed with respect to the last observed position.
    """
    ax = ax or plt.gca()
    ax.axis('equal')
    # Convert it to absolute (starting from the last observed position)
    this_pred_out_abs = pred_traj + np.array([obs_traj_gt[-1].numpy()])

    obs   = obs_traj_gt
    gt    = pred_traj_gt

    gt = np.concatenate([obs[-1,:].reshape((1,2)), gt],axis=0)
    tpred   = this_pred_out_abs

    tpred = np.concatenate([obs[-1,:].reshape((1,2)), tpred],axis=0)

    label1, = ax.plot(obs[:,0],obs[:,1],"-b", linewidth=2, label="Observations")
    label2, = ax.plot(gt[:,0], gt[:,1],"-r", linewidth=2, label="Ground truth")
    label3, = ax.plot(tpred[:,0],tpred[:,1],"-g", linewidth=2, label="Prediction")

    return label1, label2, label3

def plot_cov_world(pred_traj,cov_traj,obs_traj_gt,ax=None):
    """"""
    ax = ax or plt.gca()
    ax.axis('equal')
    # Convert it to absolute (starting from the last observed position)
    this_pred_out_abs = pred_traj + np.array([obs_traj_gt[-1].numpy()])
    length            = this_pred_out_abs.shape[0]
    for pos in range(0,length,2):
        draw_covariance_ellipse(this_pred_out_abs[pos],cov_traj[pos], ax, "Green")

# Plot trajectories in the image frame
def plot_traj_img(pred_traj, obs_traj_gt, pred_traj_gt, homography_to_world, background, ax=None):
    """"""
    ax = ax or plt.gca()

    homography_to_img = np.linalg.inv(homography_to_world)

    # Convert it to absolute (starting from the last observed position)
    this_pred_out_abs = pred_traj + np.array([obs_traj_gt[-1].numpy()])

    obs   = world_to_image_xy(obs_traj_gt, homography_to_img, flip=False)
    gt    = world_to_image_xy(pred_traj_gt, homography_to_img, flip=False)
    gt    = np.concatenate([obs[-1,:].reshape((1,2)), gt],axis=0)
    tpred = world_to_image_xy(this_pred_out_abs, homography_to_img, flip=False)
    tpred = np.concatenate([obs[-1,:].reshape((1,2)), tpred],axis=0)

    ax.plot(obs[:,0],obs[:,1],"-b", linewidth=2, label="Observations")
    ax.plot(gt[:,0], gt[:,1],"-r", linewidth=2, label="Ground truth")
    ax.plot(tpred[:,0],tpred[:,1],"-g", linewidth=2, label="Prediction")
