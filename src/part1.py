import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# Grab data from source
data_training = pd.read_csv('https://personal.utdallas.edu/~cwk200000/files/auto-mpg-training.csv')
data_test = pd.read_csv('https://personal.utdallas.edu/~cwk200000/files/auto-mpg-test.csv')

# Format the data into x and y axis
X_train = data_training.drop('mpg', axis=1).values
y_train = data_training['mpg'].values
X_test = data_test.drop('mpg', axis=1).values
y_test = data_test['mpg'].values

# Standardize the features so we can converge
X_train_standardized = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
X_test_standardized = (X_test - X_train.mean(axis=0)) / X_train.std(axis=0)

# Add bias term to X_train and X_test
X_train_bias = np.hstack([np.ones((X_train_standardized.shape[0], 1)), X_train_standardized])
X_test_bias = np.hstack([np.ones((X_test_standardized.shape[0], 1)), X_test_standardized])

def mse_cost(w, X, y):
    predictions = X.dot(w)
    errors = y - predictions
    return (errors ** 2).mean()

def mse_gradient(w, X, y):
    N = len(y)
    predictions = X.dot(w)
    gradient = -2/N * X.T.dot(y - predictions)
    return gradient

# Gradient Descent function
def gradient_descent(gradient, start, learn_rate, n_iter=100, tolerance=1e-3):
    vector = start
    for _ in range(n_iter):
        diff = -learn_rate * gradient(vector)
        if np.all(np.abs(diff) <= tolerance):
            break
        vector += diff
    return vector

log_file = open('grad_descent_part_1.log', 'w')

# Parameters for gradient descent
learn_rate = 0.01
n_iter = 1000
tolerance = 1e-5

# Use gradient descent to find optimal weights
optimal_weights_training = gradient_descent(
    gradient=lambda w: mse_gradient(w, X_train_bias, y_train),
    start=np.random.randn(X_train_bias.shape[1]),
    learn_rate=learn_rate,
    n_iter=n_iter,
    tolerance=tolerance
)

log_entry = ("--- Final Model ---\n"
             f"Weights: {optimal_weights_training}\n"
             f"Training MSE: {mse_cost(optimal_weights_training, X_train_bias, y_train)}\n"
             f"Test MSE: {mse_cost(optimal_weights_training, X_test_bias, y_test)}\n")
log_file.write(log_entry)

log_file.close()