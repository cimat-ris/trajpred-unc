import torch

def convertToCov(sx,sy,corr):
    # Exponential to get a positive value for variances
    sx   = torch.exp(sx)+1e-2
    sy   = torch.exp(sy)+1e-2
    sxsy = torch.sqrt(sx*sy)
    # tanh to get a value between [-1, 1] for correlation
    corr = torch.tanh(corr)
    # Covariance
    cov  = sxsy*corr
    return sx,sy,cov

def Gaussian2DLikelihood(targets, means, sigmas, dt=0.4):
    '''
    Computes the likelihood of predicted locations under a bivariate Gaussian distribution
    params:
    targets: Torch variable containing tensor of shape [nbatch, 12, 2]
    means: Torch variable containing tensor of shape [nbatch, 12, 2]
    sigmas:  Torch variable containing tensor of shape [nbatch, 12, 3]
    '''
    # Extract mean, std devs and correlation
    mux, muy, sx, sy, corr = means[:, 0], means[:, 1], sigmas[:, :, 0], sigmas[:,:,1], sigmas[:,:,2]
    sx,sy,cov = convertToCov(sx, sy, corr)
    # Variances and covariances are summed along time.
    # They are also scaled to fit displacements instead of velocities.
    sx   = dt*dt*sx.sum(1)
    sy   = dt*dt*sy.sum(1)
    cov  = dt*dt*cov.sum(1)
    # Compute factors
    normx= targets[:, 0] - mux
    normy= targets[:, 1] - muy
    det  = sx*sy-cov*cov
    z    = torch.pow(normx,2)*sy/det + torch.pow(normy,2)*sx/det - 2*cov*normx*normy/det
    result = 0.5*(z+torch.log(det))
    # Compute the loss across all frames and all nodes
    loss = result.sum()
    return(loss)
