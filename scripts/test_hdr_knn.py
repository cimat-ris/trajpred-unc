import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

from trajpred_unc.uncertainties.hdr_knn import hdr_knn

# Generated data
centers = [[1, 1], [-1, -1], [1, -1]]
n_samples = 1000
k         = int(np.sqrt(n_samples//2))

X, labels_true = make_blobs(
	n_samples=n_samples, centers=centers, cluster_std=0.4, random_state=0
)

# Compute the HDR
hdr     = hdr_knn(X)
alpha   = 0.85
X_inside= hdr.classify(X, alpha)
print(sum(X_inside)/len(X_inside))
n_test= 1000
Y     = np.random.uniform(low=-2, high=2, size=(n_test,2))
Y_inside= hdr.classify(Y, alpha)



plt.plot(X[X_inside==True, 0], X[X_inside==True, 1], "o",color="b",markersize=3)
plt.plot(X[X_inside==False,0], X[X_inside==False, 1], "o",color="r",markersize=3)
plt.plot(Y[Y_inside==True, 0],Y[Y_inside==True, 1],"o",color="c",markersize=3)
plt.plot(Y[Y_inside==False, 0],Y[Y_inside==False, 1],"o",color="orange",markersize=3)
# Set the aspect of the plot to be equal
plt.axis("equal")
plt.show()
