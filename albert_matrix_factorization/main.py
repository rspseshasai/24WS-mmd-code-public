#!/usr/bin/python
#
# Created by Albert Au Yeung (2010)
#
# An implementation of matrix factorization

try:
    import numpy
except:
    print("This implementation requires the numpy module.")
    exit(0)

###############################################################################

"""
@INPUT:
    R     : a matrix to be factorized, dimension N x M
    P     : an initial matrix of dimension N x K
    Q     : an initial matrix of dimension M x K
    K     : the number of latent features
    steps : the maximum number of steps to perform the optimisation
    alpha : the learning rate
    beta  : the regularization parameter
@OUTPUT:
    the final matrices P and Q
"""


def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    Q = Q.T
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:  # Only consider non-zero ratings (i.e., not NaN or missing)
                    eij = R[i][j] - numpy.dot(P[i, :], Q[:, j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = numpy.dot(P, Q)
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - numpy.dot(P[i, :], Q[:, j]), 2)
                    for k in range(K):
                        e = e + (beta / 2) * (pow(P[i][k], 2) + pow(Q[k][j], 2))
        if e < 0.001:
            break
    return P, Q.T


###############################################################################

if __name__ == "__main__":
    # Utility matrix R
    R = [
        [1, 3, 0, 5, 0, 5, 0, 0, 5, 0, 4, 0],
        [0, 5, 4, 0, 3, 0, 4, 0, 0, 2, 1, 3],
        [2, 4, 0, 1, 2, 0, 3, 0, 4, 3, 5, 0],
        [0, 2, 4, 0, 5, 0, 0, 4, 0, 0, 2, 0],
        [0, 0, 4, 3, 4, 2, 0, 0, 0, 0, 2, 5],
        [1, 3, 3, 0, 2, 0, 0, 2, 0, 0, 4, 0]
    ]

    # Convert to numpy array
    R = numpy.array(R)

    N = len(R)  # Number of items (rows)
    M = len(R[0])  # Number of users (columns)
    K = 3  # Number of latent features (you can adjust this)

    # Random initialization of matrices P and Q
    P = numpy.random.rand(N, K)
    Q = numpy.random.rand(M, K)

    # Perform matrix factorization
    nP, nQ = matrix_factorization(R, P, Q, K)

    # Display results
    print("Matrix P (Items x Latent Factors):\n", nP)
    print("\nMatrix Q (Latent Factors x Users):\n", nQ)

    # Reconstructed matrix
    R_pred = numpy.dot(nP, nQ.T)  # Corrected dot product with transposed Q
    print("\nReconstructed Matrix R_pred (Items x Users):\n", R_pred)
