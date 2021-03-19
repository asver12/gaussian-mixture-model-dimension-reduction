import unittest
import numpy as np
import gmm_dimension_reduction


class TestSlicing(unittest.TestCase):

    def test_default(self):
        cov = [[1950., 47., 860.5, -879.],
               [47., 444., 212., 350.],
               [860.5, 212., 1535., -19.],
               [-879., 350., -19., 1290.]]
        expected_covs = [[1950., 47., 860.5], [47., 444., 212.], [860.5, 212., 1535.]]
        expected_covs = [expected_covs, expected_covs]

        mean = [1, 2, 3, 4]
        expected_means = ([1, 2, 3], [1, 2, 3])

        weights = [0.5, 0.5]

        result_weights, result_means, result_covs = gmm_dimension_reduction.slicing.slice_gmm([1, 2, 3], weights, [mean, mean], [cov, cov])
        print(result_covs)
        self.assertSequenceEqual(result_weights, weights)
        np.testing.assert_array_equal(result_means, expected_means)
        np.testing.assert_array_equal(result_covs, expected_covs)
