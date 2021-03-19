import numpy as np


class PCA:
    def __init__(self, data, n_components):
        self.mean = np.mean(data, axis=0)
        new_data = data - self.mean
        cov_mat = np.cov(new_data, rowvar=False)
        eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)
        sorted_index = np.argsort(eigen_values)[::-1]
        sorted_eigenvalue = eigen_values[sorted_index]
        sorted_eigenvectors = eigen_vectors[:, sorted_index]
        self.principle_components = sorted_eigenvectors[:, 0:n_components]

    def fit(self, data):
        return np.dot(eigenvector_subset.transpose(), new_data.transpose()).transpose()
