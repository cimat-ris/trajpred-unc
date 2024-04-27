from sklearn.neighbors import BallTree
import numpy as np

# Class to approximate the HDR of a distribution given through samples with KNN
class hdr_knn:
    def __init__(self, X):
        # As a rule of thumb, the number of neighbors should be the square root of the number of samples
        self.X         = X
        self.n_samples = X.shape[0]
        self.k         = int(np.sqrt(self.n_samples//2))
        self.tree      = BallTree(X, leaf_size=2)
        self.X_sparsity= self.compute_sparsity(self.X)

    # Compute and sort the values of the sum of distances to the k nearest neighbors 
    def compute_sparsity(self, X):
        # Compute the sum of distances to the closest neighbors
        dist, __  = self.tree.query(X, k=self.k) 
        M         = np.sum(dist, axis=1)  
        M         = np.sort(M, axis=0)
        return M
    
    # For a given alpha, classify data points as inside or outside the HDR
    def classify(self, Y, alpha):
        threshold = self.X_sparsity[int(alpha*self.n_samples)]
        # Compute the sum of distances to the closest neighbors
        dist, __  = self.tree.query(Y, k=self.k) 
        MY        = np.sum(dist, axis=1)
        return MY<threshold
        
