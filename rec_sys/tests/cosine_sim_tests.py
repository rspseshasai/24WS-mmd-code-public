import unittest

import numpy as np
from scipy.sparse import csr_matrix

from rec_sys.cf_algorithms import centered_cosine_sim, cosine_sim, center_and_nan_to_zero


# Assuming the previous functions are defined in the same module
# from your_module import centered_cosine_sim

class TestCenteredCosineSim(unittest.TestCase):

    def test_centered_cosine_sim_b1(self):
        """Test centered cosine similarity with k = 100, xi = i + 1."""
        k = 100
        vector_x = np.arange(1, k + 1)  # [1, 2, ..., 100]
        vector_y = np.arange(1, k + 1)[::-1]  # [100, 99, ..., 1]

        # Convert to sparse matrices
        sparse_vector_x = csr_matrix(vector_x)
        sparse_vector_y = csr_matrix(vector_y)

        result = centered_cosine_sim(sparse_vector_x, sparse_vector_y)
        expected_result = -1.0  # They are perfectly negatively correlated
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_centered_cosine_sim_b2(self):
        """Test centered cosine similarity with specific NaN patterns."""

        k = 100
        vector_x = np.zeros(k)  # Start with a zero vector for x
        vector_y = np.zeros(k)  # Start with a zero vector for y

        # Fill vector_x according to the specified rules
        for i in range(k):
            if i % 10 in [2, 3, 4, 5, 6]:  # Set NaN for specific indices
                vector_x[i] = np.nan
            else:
                vector_x[i] = i + 1  # Assign xi = i + 1

        # Populate vector_y based on the relationship with vector_x
        for i in range(k):
            if not np.isnan(vector_x[i]):
                vector_y[k - 1 - i] = vector_x[i]  # Assign corresponding yi

        # Convert to sparse matrices
        sparse_vector_x = csr_matrix(vector_x)
        sparse_vector_y = csr_matrix(vector_y)

        # Compute centered cosine similarity
        result = centered_cosine_sim(sparse_vector_x, sparse_vector_y)

        # Calculate expected_result based on valid indices
        valid_indices = [i for i in range(k) if not np.isnan(vector_x[i])]
        centered_x = center_and_nan_to_zero(sparse_vector_x, axis=1).toarray().flatten()
        centered_y = center_and_nan_to_zero(sparse_vector_y, axis=1).toarray().flatten()
        # Calculate expected cosine similarity
        expected_result = cosine_sim(centered_x, centered_y)

        # # Print values for debugging
        # print("Vector X:", vector_x)
        # print("Vector Y:", vector_y)
        # print("Centered X:", centered_x)
        # print("Centered Y:", centered_y)
        # print("Result:", result)
        # print("Expected Result:", expected_result)

        # Perform the assertion
        self.assertAlmostEqual(result, expected_result, places=5)


# Run the tests
if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
