import numpy as np


def is_gaussian_mixture_model(weights, means, covs, rtol=1e-05, atol=1e-08):
    if not np.isclose(np.sum(weights), 1.0, rtol, atol):
        raise ValueError(f"The weights are not summing up to 1[{np.sum(weights)}")
    if len(weights) != len(means) != len(covs):
        raise AttributeError(
            f"The weights[{len(weights)}], means[{len(means)}] and covs[{len(covs)}] have to be the same size")


def is_gaussian(mean, cov, rtol=1e-05, atol=1e-08):
    cov = np.asarray(cov)
    if len(mean) != len(cov[0]):
        raise AttributeError(
            f"No gaussian input | mean: {mean} | cov: {cov}")
    # todo:: add check if cov-matrix is psd
    if not np.allclose(cov, cov.T, rtol=rtol, atol=atol):
        raise ValueError(
            f"Covariance matrix has to be symmetric"
        )
