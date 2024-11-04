# Pavan, October 2024
# Algorithms for collaborative filtering
# Implement the CF from the lecture 1
import numpy as np


def center_and_nan_to_zero(matrix, axis=0):
    """ Center the matrix and replace NaN values with zeros. Handles both dense and sparse matrices. """
    if issparse(matrix):
        # Convert sparse matrix to dense for mean calculation
        dense_matrix = matrix.toarray()
    else:
        dense_matrix = matrix

    # Compute mean of non-NaN values along the specified axis
    means = np.nanmean(dense_matrix, axis=axis)

    # Center the matrix by subtracting the means
    if axis == 0:  # Mean across rows
        centered_matrix = dense_matrix - means[np.newaxis, :]  # Broadcasting for rows
    else:  # Mean across columns
        centered_matrix = dense_matrix - means[:, np.newaxis]  # Broadcasting for columns

    # Replace NaNs with zeros
    centered_matrix = np.nan_to_num(centered_matrix)

    # Convert back to sparse matrix if the input was sparse
    if issparse(matrix):
        return csr_matrix(centered_matrix)
    return centered_matrix


def cosine_sim(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


def fast_cosine_sim(utility_matrix, vector, axis=0):
    """ Compute the cosine similarity between the matrix and the vector"""
    # Compute the norms of each column
    norms = np.linalg.norm(utility_matrix, axis=axis)
    um_normalized = utility_matrix / norms
    # Compute the dot product of transposed normalized matrix and the vector
    dot = np.dot(np.transpose(um_normalized), vector)
    # Scale by the vector norm
    scaled = dot / np.linalg.norm(vector)
    return scaled


def centered_cosine_sim(vector_a, vector_b):
    # Convert vectors to 2D sparse matrices for compatibility
    matrix_a = csr_matrix(vector_a)
    matrix_b = csr_matrix(vector_b)

    # Center the vectors using center_and_nan_to_zero
    centered_a = center_and_nan_to_zero(matrix_a, axis=1).toarray().flatten()
    centered_b = center_and_nan_to_zero(matrix_b, axis=1).toarray().flatten()

    # Compute cosine similarity
    numerator = np.dot(centered_a, centered_b)
    denominator = np.linalg.norm(centered_a) * np.linalg.norm(centered_b)

    return numerator / denominator if denominator != 0 else 0


def fast_centered_cosine_sim(matrix, vector):
    # Ensure vector is 2D for compatibility
    if vector.ndim == 1:
        vector = vector[np.newaxis, :]  # Convert to 2D if it’s 1D

    # Center the vector (convert to dense array if sparse)
    if hasattr(vector, "toarray"):
        centered_vector = center_and_nan_to_zero(vector, axis=1).toarray().flatten()
    else:
        centered_vector = center_and_nan_to_zero(vector, axis=1).flatten()

    similarities = []
    for i in range(matrix.shape[0]):
        # Check if matrix is sparse and access rows accordingly
        row = matrix.getrow(i) if hasattr(matrix, "getrow") else matrix[i, :]

        # Ensure row is 2D before centering and convert to dense if sparse
        if row.ndim == 1:
            row = row[np.newaxis, :]  # Convert to 2D if it’s 1D
        if hasattr(row, "toarray"):
            centered_row = center_and_nan_to_zero(row, axis=1).toarray().flatten()
        else:
            centered_row = center_and_nan_to_zero(row, axis=1).flatten()

        # Calculate cosine similarity
        numerator = np.dot(centered_row, centered_vector)
        denominator = np.linalg.norm(centered_row) * np.linalg.norm(centered_vector)

        similarity = numerator / denominator if denominator != 0 else 0
        similarities.append(similarity)

    return np.array(similarities)


from scipy.sparse import csr_matrix, issparse


def rate_all_items(utility_matrix, user_index, neighborhood_size):
    # Determine if the utility matrix is sparse or dense
    is_sparse = hasattr(utility_matrix, 'nnz')  # Check if it's a sparse matrix

    if is_sparse:
        clean_utility_matrix = utility_matrix.tocsr()
    else:
        clean_utility_matrix = utility_matrix  # Already a dense matrix

    # Get the user's ratings
    user_ratings = clean_utility_matrix[user_index]

    # Convert to 1D array if the utility matrix is dense
    if not is_sparse:
        user_ratings = user_ratings.flatten()
    else:
        # If sparse, convert to dense for further processing
        user_ratings = user_ratings.toarray().flatten()  # Use toarray() instead of .A

    # Identify which items the user has rated
    rated_items = np.where(user_ratings > 0)[0]
    num_rated_items = len(rated_items)

    if num_rated_items == 0:
        return np.full(clean_utility_matrix.shape[1], np.nan)

    # Calculate similarities
    similarities = fast_centered_cosine_sim(clean_utility_matrix, user_ratings)

    # Get indices of the best neighbors based on similarity
    best_indices = np.argsort(similarities)[-neighborhood_size:][::-1]  # Highest similarities first
    best_among_who_rated = np.intersect1d(best_indices, rated_items)

    if best_among_who_rated.size == 0:
        return np.full(clean_utility_matrix.shape[1], np.nan)

    # Calculate the predicted rating for each item
    predicted_ratings = np.zeros(clean_utility_matrix.shape[1])

    for item_index in range(clean_utility_matrix.shape[1]):
        # Only compute ratings for items not already rated by the user
        if item_index not in rated_items:
            # Compute the rating based on neighbors
            if is_sparse:
                # For sparse matrix, use toarray() to convert to dense
                rating_of_item = np.sum(similarities[best_among_who_rated] * clean_utility_matrix[
                    best_among_who_rated, item_index].toarray().flatten())
            else:
                # For dense matrix, simply use indexing
                rating_of_item = np.sum(
                    similarities[best_among_who_rated] * clean_utility_matrix[best_among_who_rated, item_index])

            denominator = np.sum(similarities[best_among_who_rated])

            if denominator == 0:
                predicted_ratings[item_index] = np.nan
            else:
                predicted_ratings[item_index] = rating_of_item / denominator

            if np.isinf(predicted_ratings[item_index]):
                predicted_ratings[item_index] = np.nan

    return predicted_ratings
