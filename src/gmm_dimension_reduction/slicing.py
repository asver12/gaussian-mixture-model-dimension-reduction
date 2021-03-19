import numpy as np
import itertools

from . import validate

def slice_gmm(variables, weights, means, covs):
    """
    Generates the slice of a gaussian mixture model through marginalization for given dimensions, which is again a slice

    :param variables: The variable indices you want to end up with
    :param weights: [pi] the weights each component holds. The sum of all pies has to be 1.0
    :param means: [mu] the mean of each component
    :param covs: [sigma] the covariance matrices of each component as diag
    :return:
    """
    validate.is_gaussian_mixture_model(weights, means, covs)
    variables = np.asarray(variables) - 1
    return weights, *list(zip(*[__marginalize_gaussian(variables, mean, covs) for mean, covs in zip(means, covs)]))


def __marginalize_gaussian(variables, mean, cov):
    validate.is_gaussian(mean, cov)
    cov = np.asarray(cov)
    if len(cov.shape) == 1:
        marginalized_cov = cov[variables]
    elif len(cov.shape) == 2:
        if cov.shape[0] != cov.shape[1]:
            raise ValueError(f"Given matrix is not a cov-matrix")
        marginalized_cov = cov[tuple(np.array(list(itertools.product(variables, repeat=2))).T.tolist())].reshape(
            len(variables), len(variables))
    else:
        raise ValueError(f"Given matrix is not a cov-matrix")
    return np.asarray(mean)[variables], marginalized_cov


if __name__ == "__main__":
    N = 4
    b = np.random.randint(-2000, 2000, size=(N, N))
    cov = (b + b.T) / 2
    print(cov)
    mean = [5, 6, 7, 8]
    weights, means, covs = slice_gmm([1, 2, 3], [1, 0], [mean, mean], [cov, cov])
    print(weights)
    print(means)
    print(covs)
