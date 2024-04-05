import random
from scipy.stats import gaussian_kde
from scipy.stats import multivariate_normal,multinomial
import numpy as np
import matplotlib.pyplot as plt 
from trajpred_unc.utils.plot_utils import world_to_image_xy
from trajpred_unc.uncertainties.hdr_kde import get_falpha

def gaussian_kde_from_gaussianmixture(prediction, sigmas_prediction, kde_size=1000, resample_size=100):
	"""
	Builds a KDE representation from a Gaussian mixture (output of one of the prediction algorithms)
	Args:
	  - prediction: set of position predictions
	  - sigmas_prediction: covariances of the predictions
	  - resample_size: number of samples to produce from the KDE
	Returns:
	  - kde: PDF estimate through KDE
	  - sample_kde: Sampled points (x,y) from the PDF
	"""
	# This array will hold the parameters of each element of the mixture
	gaussian_mixture = []
	# Form the Gaussian mixture
	for idx_ensemble in range(sigmas_prediction.shape[0]):
		# Get means and standard deviations
		sigmas_samples_ensemble = sigmas_prediction[idx_ensemble,:]
		sx,sy,cov               = sigmas_samples_ensemble[0],sigmas_samples_ensemble[1],sigmas_samples_ensemble[2]
        # Predictions arrive here in **absolute coordinates**
		mean                = prediction[idx_ensemble, :]
		# TODO: use the correlations too?
		covariance          = np.array([[sx, cov],[cov, sy]])
		gaussian_mixture.append(multivariate_normal(mean,covariance))
	# Performs sampling on the Gaussian mixture
	pi                 = np.ones((len(gaussian_mixture),))/len(gaussian_mixture)
	partition          = multinomial(n=kde_size,p=pi).rvs(size=1)
	sample_pdf         = np.zeros((kde_size,2))
	sum                = 0
	for gaussian_id,gaussian in enumerate(gaussian_mixture):
		#sample_pdf.append(gaussian.rvs(size=partition[0][gaussian_id]))
		sample_pdf[sum:sum+partition[0][gaussian_id]]=gaussian.rvs(size=partition[0][gaussian_id])
		sum = sum +partition[0][gaussian_id]
	# Use the samples to generate a KDE
	f_density = gaussian_kde(sample_pdf.T)
	rows_id   = random.sample(range(0,sample_pdf.shape[0]),resample_size)
	return f_density,sample_pdf[rows_id, :]

# Sort samples with respect to their density value
def sort_sample(sample_pdf):
    # Samples from the pdf are sorted in decreasing order
    sample_pdf_zip = zip(sample_pdf, sample_pdf/np.sum(sample_pdf))
    return sorted(sample_pdf_zip, key=lambda x: x[1], reverse=True)

# Given a set of pdf values from samples on the pdf, and a pdf value, deduce alpha
def samples_to_alphas(kde,samples):
	# Evaluate our samples on the kde
	fs_samples      = kde.evaluate(samples)
	sorted_samples  = sort_sample(fs_samples)
	# Alphas corresponding to each sample
	observed_alphas = []
	for fk in fs_samples:
		observed_alpha = 0.0
		for p in sorted_samples:
			if fk>p[0]:
				break
			# Accumulate density here
			observed_alpha += p[1]
		observed_alphas.append(observed_alpha)
	observed_alphas = np.array(observed_alphas)
	return observed_alphas,fs_samples,sorted_samples

def evaluate_kde(prediction, sigmas_prediction, ground_truth, kde_size=1000, resample_size=100):
	"""
	Builds a KDE representation for the prediction and evaluate the ground truth on it
	Args:
	  - prediction: set of predicted positions
	  - sigmas_prediction: set of covariances on the predicted position (may be None)
	  - ground_truth: set of ground truth positions
	  - resample_size: number of samples to produce from the KDE
	Returns:
	  - f_ground_truth: PDF values at the ground truth points
	  - f_samples: PDF values at the samples
	"""
	if sigmas_prediction is not None:
		# In this case, we use a Gaussian output and create a KDE representation from it
		f_density, samples = gaussian_kde_from_gaussianmixture(prediction,sigmas_prediction,kde_size=kde_size,resample_size=resample_size)
		f_samples      = f_density.pdf(samples.T)
	else:
		# In this case, we just have samples and create the KDE from them
		f_density = gaussian_kde(prediction.T)
		# Then we sample (again) from the obtained representation
		samples   = f_density.resample(resample_size,0)
		f_samples = f_density.pdf(samples)
	# Evaluate the GT and the samples on the obtained KDE
	f_ground_truth = f_density.pdf(ground_truth)
	return f_density, f_ground_truth, f_samples, samples

def plot_kde_img(observed,target,xs,ys,img_bckd,homography,ax,iso_inv=None):
    xmin = 0
    xmax = img_bckd.shape[1]
    ymin = 0
    ymax = img_bckd.shape[0]
    xx, yy = np.mgrid[xmin:xmax:100j,ymin:ymax:100j]
	
    # Testing/visualization uncalibrated KDE
    image_grid       = np.vstack([xx.ravel(), yy.ravel()])
    world_grid       = world_to_image_xy(np.transpose(image_grid),homography,flip=False)
    # Prediction samples
    world_samples    = np.vstack([xs, ys])
    homography_to_img= np.linalg.inv(homography)
    image_samples    = world_to_image_xy(np.transpose(world_samples),homography_to_img,flip=False)
			
    # Build a Kernel Density Estimator with these samples
    kde             = gaussian_kde(world_samples)
    # Evaluate our samples on it
    alphas_samples,fs_samples,sorted_samples = samples_to_alphas(kde,world_samples)    
    if iso_inv is not None:
        modified_alphas = iso_inv.transform(alphas_samples)
        # New values for f
        fs_samples_new  = []
        for alpha in modified_alphas:
            fs_samples_new.append(get_falpha(sorted_samples,alpha))
        fs_samples_new    = np.array(fs_samples_new)
        sorted_samples    = sort_sample(fs_samples_new)
        importance_weights= fs_samples_new/fs_samples
		# Recompute KDE with importance weights
        kde               = gaussian_kde(world_samples,weights=importance_weights)
        alphas_samples, fs_samples, sorted_samples = samples_to_alphas(kde,world_samples)

    # Visualization of the uncalibrated KDE with its level curves
    alphas = np.linspace(1.0,0.0,num=5,endpoint=False)
    levels = []
    for alpha in alphas:
        level = get_falpha(sorted_samples,alpha)
        levels.append(level)
    # Apply the KDE on the points of the world grid
    f_values     = np.reshape(kde(np.transpose(world_grid)).T, xx.shape)
    transparency = np.minimum(np.rot90(f_values)/np.max(f_values),1.0)

    ## Or kernel density estimate plot instead of the contourf plot
    ax.legend_ = None
    ax.imshow(img_bckd)
    observations = world_to_image_xy(observed[:,:], homography_to_img, flip=False)
    groundtruth  = world_to_image_xy(target[:,:], homography_to_img, flip=False)
    # Contour plot
    cset = ax.contour(xx, yy, f_values, colors='darkgreen',levels=levels[1:],linewidths=0.75)
    cset.levels = np.array(alphas[1:])
    ax.clabel(cset, cset.levels,fontsize=8)
    ax.plot(observations[:,0],observations[:,1],color='blue')
    ax.plot([observations[-1,0],groundtruth[0,0]],[observations[-1,1],groundtruth[0,1]],color='red')
    ax.plot(groundtruth[:,0],groundtruth[:,1],color='red')
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymax,ymin)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    # Plot the pdf
    ax.imshow(transparency,alpha=np.sqrt(transparency),cmap=plt.cm.Greens_r,extent=[xmin, xmax, ymin, ymax])
