import numpy as np
import matplotlib.pyplot as plt


def matrix_factorization1(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    Q = Q.T
    error_list = []  # To store the error at each step
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i, :], Q[:, j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = np.dot(P, Q)
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - np.dot(P[i, :], Q[:, j]), 2)
                    for k in range(K):
                        e = e + (beta / 2) * (pow(P[i][k], 2) + pow(Q[k][j], 2))
        error_list.append(e)  # Append error for this iteration
        if e < 0.001:
            break
    return P, Q.T, error_list  # Return error list as well


# Define your 6x12 utility matrix
R = [
    [1, 0, 3, 0, 0, 5, 0, 0, 5, 0, 4, 0],
    [0, 0, 5, 4, 0, 0, 4, 0, 0, 2, 1, 3],
    [2, 4, 0, 1, 2, 0, 3, 0, 4, 3, 5, 0],
    [0, 2, 4, 0, 5, 0, 0, 4, 0, 0, 2, 0],
    [0, 0, 4, 3, 4, 2, 0, 0, 0, 0, 2, 5],
    [1, 0, 3, 0, 3, 0, 0, 2, 0, 0, 4, 0],
]

R = np.array(R)

N = len(R)
M = len(R[0])
K = 3

P = np.random.rand(N, K)
Q = np.random.rand(M, K)

nP, nQ, error_list = matrix_factorization1(R, P, Q, K)

print("Matrix P:")
print(np.round(nQ, 1))

print("\nMatrix Q:")
print(np.round(nP, 1))

nR = np.dot(nP, nQ.T)

print("\nPredicted Matrix R (dot product of Q and P.T):")
print(np.round(nR, 1))

# Plot the error over iterations
plt.figure(figsize=(10, 5))
plt.plot(error_list, label='Error')
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.title('Error over Iterations')
plt.legend()
plt.show()

# Plot the differences in the error
error_diff = [error_list[i+1] - error_list[i] for i in range(len(error_list)-1)]
plt.figure(figsize=(10, 5))
plt.plot(error_diff, label='Error Difference')
plt.xlabel('Iterations')
plt.ylabel('Error Difference')
plt.title('Differences in Error between Iterations')
plt.legend()
plt.show()
