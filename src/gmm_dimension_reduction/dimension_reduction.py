import numpy as np

from gmm_dimension_reduction import validate


def gmm_pca(pca_mean, pca_eigenvectors, weights, means, covs):
    validate.is_gaussian_mixture_model(weights, means, covs)
    dr_means = []
    dr_covs = []
    for mean, cov in zip(means, covs):
        validate.is_gaussian(mean, cov)
        dr_means.append(__get_mean(pca_eigenvectors, mean - pca_mean))
        dr_covs.append(__get_cov(pca_eigenvectors, cov))
    return weights, dr_means, dr_covs


def __get_mean(pca_eigenvectors, mean):
    return np.dot(pca_eigenvectors.transpose(), mean.transpose()).transpose()


def __get_cov(pca_eigenvectors, covariance_matrix):
    return np.dot(pca_eigenvectors.transpose(), np.dot(covariance_matrix, pca_eigenvectors))
