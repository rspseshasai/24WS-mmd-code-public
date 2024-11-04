import unittest
import numpy as np
from scipy.sparse import csr_matrix

from rec_sys.cf_algorithms import rate_all_items


class TestSparseRateAllItems(unittest.TestCase):
    def test_sparse_vs_dense_results(self):
        # Create a sample utility matrix (UM) with known ratings
        dense_UM = np.array([
            [5, np.nan, 3],
            [np.nan, 4, np.nan],
            [2, np.nan, np.nan]
        ])
        sparse_UM = csr_matrix(dense_UM)

        # Define user index and neighborhood size
        user_index = 1
        neighborhood_size = 2

        # Get ratings from dense and sparse versions
        dense_ratings = rate_all_items(dense_UM, user_index, neighborhood_size)
        sparse_ratings = rate_all_items(sparse_UM, user_index, neighborhood_size)

        # Assert results are close
        np.testing.assert_allclose(dense_ratings, sparse_ratings, rtol=1e-5)

if __name__ == '__main__':
    unittest.main()
