# Description: Introduction to JAX via a simple example of SGD
# Artur Andrzejak, October 2024

import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt


# %% Define a simple function and sample some data
def f(x, param_w):
    return (x - param_w) * (x - 2 * param_w)

# Generate data
n_points, noise_frac, rnd_seed = 1000, 0.25, 42
x = jnp.linspace(-3, 6, n_points)
y_pure = f(x, 2.0)
# Add some noise to data
rnd_key = jax.random.PRNGKey(rnd_seed)
y_with_noise = y_pure + y_pure * noise_frac * jax.random.normal(rnd_key, (n_points,))
# Stack x and y_with_noise into a single array
train_ds = jnp.stack((x, y_with_noise), axis=1)
print(f"Training data (first 5):\n {train_ds[:5]}")

# %% Plot the data
plt.plot(x, y_pure, label='True function')
plt.plot(x, y_with_noise, 'o', label='Data with noise')
plt.legend()
plt.show()


# %% Define a simple loss function and its gradient
def loss(param_w, data):
    # return  jnp.sum((data[:,1] - f(data[:,0], param_w))**2)
    return jnp.log(jnp.sum((data[:, 1] - f(data[:, 0], param_w)) ** 2))


# Using JAX automatic differentiation - autograd
grad_loss = jax.grad(loss)


# %% Note that grad_loss is a function!
param_w = 1.0
print(f"\n\nLoss value for param_w = {param_w}: {loss(param_w, train_ds)}\n\n")
print(jax.make_jaxpr(grad_loss)(param_w, train_ds))


# %% Plot the loss function and its gradient
def compute_loss_and_grad(param_w, data, start, stop, num_points=100):
    param_w_values = jnp.linspace(start, stop, num_points)
    loss_values = jnp.array([loss(w, data) for w in param_w_values])
    grad_values = jnp.array([grad_loss(w, data) for w in param_w_values])
    return param_w_values, loss_values, grad_values

param_w_values, loss_values, grad_values = (
    compute_loss_and_grad(0.0, train_ds, -3, 10))

plt.plot(param_w_values, loss_values, label='Loss')
plt.plot(param_w_values, grad_values, label='Gradient')
plt.legend()
plt.show()


# %% Run the SG loop
num_epochs = 300
learning_rate = 0.005
param_w = 0.0  # Initial guess for the parameter

print("\n===== Running Gradient Descent =====")
for epoch in range(num_epochs):
    grad = grad_loss(param_w, train_ds)
    param_w = param_w - learning_rate * grad
    if epoch % 2 == 0:
        print(f"Epoch {epoch}: param_w={param_w}, grad={grad}, loss={loss(param_w, train_ds)}")


# %% Plot the results
plt.plot(x, y_pure, label='True function')
plt.plot(x, y_with_noise, 'o', label='Data with noise')
plt.plot(x, f(x, param_w), label='Fitted function')
plt.legend()
plt.show()


# %% Run stochastic gradient descent
num_epochs = 50
learning_rate = 0.01
param_w = 0.0
num_points_per_batch = n_points // 5
print("\n===== Running Stochastic Gradient Descent =====")
for epoch in range(num_epochs):
    # Get points for the current batch
    for i in range(0, n_points, num_points_per_batch):
        batch = train_ds[i:i + num_points_per_batch]
        grad = grad_loss(param_w, batch)
        param_w = param_w - learning_rate * grad

    print(f"Epoch {epoch}: param_w={param_w}, grad={grad}, loss={loss(param_w, train_ds)}")
