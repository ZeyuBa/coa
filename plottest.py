import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm
EPS=1e-5
# Example function to optimize
def objective_function(x):
    return np.sin(x) * np.cos(x/2)

# Sample data
# X_sample = np.array([-1.5,  2.5, 3.8, ]).reshape(-1, 1)
X_sample = np.array([-1.5, 0.1, 3.8, 6, 7.3]).reshape(-1, 1)
y_sample = objective_function(X_sample).ravel()

# Generate true function values for plotting
X = np.linspace(-2, 10, 1000).reshape(-1, 1)
y_true = objective_function(X).ravel()

# Gaussian Process with Matern kernel
kernel = Matern(length_scale=1.0, nu=2.5)
gp = GaussianProcessRegressor(kernel=kernel, alpha=0.1)
gp.fit(X_sample, y_sample)

# Make the prediction on the meshed x-axis (ask for MSE as well)
y_pred, sigma = gp.predict(X, return_std=True)
def expected_improvement(X, X_sample, gp, xi=0.01):
    mu, sigma = gp.predict(X, return_std=True)
    mu_sample = gp.predict(X_sample)

    # Flatten sigma to ensure it's a one-dimensional array
    sigma = sigma.flatten()

    # Needed for noise-based model,
    # otherwise use np.max(Y_sample).
    mu_sample_opt = np.max(mu_sample)

    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        print(ei.shape)

        # Apply a mask where sigma is effectively zero
        mask = (abs(sigma) < EPS)
        ei[mask] = 0.0  # This line assumes ei and mask are 1D and of the same length

    return ei


# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 6))
fig.suptitle('Gaussian Process and Acquisition Function After 5 Steps')




# Plot true function
# plt.subplot(2, 1, 1)
ax1.plot(X, y_true, 'g--', linewidth=2, label='Objective function')
ax1.fill(np.concatenate([X, X[::-1]]),
         np.concatenate([y_pred - 1.9600 * sigma, (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.5, fc='c', ec='None', label='95% confidence interval')

ax1.plot(X, y_pred, 'k--', linewidth=2, label='Predictive mean')
ax1.scatter(X_sample, y_sample, c='r', s=50, zorder=10, edgecolors=(0, 0, 0), label='Observations')
ax1.scatter(X_sample[np.argmax(y_sample)], np.max(y_sample), c='b', s=100, zorder=10, edgecolors=(0, 0, 0), label='Current best observation')
ax1.set_xlabel('x')
ax1.set_ylabel('GP(x)') 
ax1.legend(loc='upper left')
ax1.set_xlim((-2,8))
ax1.set_xticks(list(range(-1,8,1)))


# Plot acquisition function
acquisition = expected_improvement(X, X_sample, gp)
ax2.plot(X, acquisition, 'r-', lw=2, label='Acquisition function',)

ax2.fill_between(X.ravel(), 0, acquisition.ravel(), color='purple', alpha=0.3)

ax2.scatter(X[np.argmax(acquisition)], np.max(acquisition), c='y', s=100, label='Next candidate')

ax2.set_xlabel('x')
ax2.set_ylabel('EI')


ax2.set_xlim((-2,8))
ax2.legend(loc='upper left')
# Show plot
plt.tight_layout()
plt.savefig('./BO.png')
