import numpy as np
from scipy.stats import multivariate_normal

class GaussianMixtureModel:
    def __init__(self, weights, means, covariances):
        self.weights = weights
        self.means = means
        self.covariances = covariances
        self.n_components = len(weights)

    def __pdf(self, x):
        return np.sum(np.dot(self.weights, [multivariate_normal(mean, covariance).pdf(x) for mean, covariance in
                                            zip(self.means, self.covariances)]))

    def predict(self, x, prediction_function="selfmade"):
        pred_function = self.__predict
        if prediction_function == "scipy":
            # A lot of unnecessary calls with this function
            pred_function = self.__pdf
        if len(np.asarray(x).shape) == 1:
            return pred_function(x)
        return [self.__pdf(x_1) for x_1 in x]

    def __predict(self, x):
        return np.sum(np.dot(self.weights, [self.__multivariate_normal(x, mean, covariance) for mean, covariance in
                                            zip(self.means, self.covariances)]))

    def __multivariate_normal(self, X, mean_vector, covariance_matrix):
        return (2 * np.pi) ** (-len(X) / 2) * np.linalg.det(covariance_matrix) ** (-1 / 2) * np.exp(
            -np.dot(np.dot((X - mean_vector).T, np.linalg.inv(covariance_matrix)), (X - mean_vector)) / 2)
