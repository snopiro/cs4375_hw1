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

# Add Bias Term
X_train = np.column_stack([np.ones(X_train.shape[0]), X_train])
X_test = np.column_stack([np.ones(X_test.shape[0]), X_test])

# Cost Function
def get_cost(X, y, w):
    m = len(y)
    y_pred = X.dot(w)
    return (1 / (2 * m)) * np.sum(np.square(y_pred - y))

# Gradient Descent
# Cost history can be used for debugging and visualizing the convergence
def gradient_descent(X, y, w, alpha, num_iters):
    m = len(y)
    cost_history = []

    for i in range(num_iters):
        gradient = (1/m) * X.T.dot(X.dot(w) - y)
        w -= alpha * gradient
        cost_history.append(get_cost(X, y, w))

    return w, cost_history

# Tuning Parameters & Logging
alphas = [0.01, 0.001, 0.0001]  # List of learning rates to try
iterations = [5, 10, 20, 50]  # List of number of iterations to try

log_file = open('grad_descent_part_1.log', 'w')

# Log Initial Model
# initial_w = np.zeros(X_train.shape[1])
# initial_w = np.random.rand(X_train.shape[1])
initial_w = np.array([1.0, 0.2, 0.3, 0.01, -0.001, .2])
log_entry = f"--- Initial Model ---\nWeights: {initial_w}\n\n--- Final Models ---\n"
log_file.write(log_entry)

for alpha in alphas:
    for num_iters in iterations:
        # Gradient Descent
        final_w, cost_history = gradient_descent(X_train, y_train, initial_w.copy(), alpha, num_iters)
        final_mse_train = get_cost(X_train, y_train, final_w)
        final_mse_test = get_cost(X_test, y_test, final_w)

        # Log Each Final Model
        log_entry = (f"Weights: {final_w}\n"
                     f"Alpha: {alpha}, Iterations: {num_iters}, "
                     f"Training MSE: {final_mse_train}, Test MSE: {final_mse_test}\n\n")
        log_file.write(log_entry)

        # plt.plot(cost_history)
        # plt.title(f"Convergence for alpha={alpha}, iterations={num_iters}")
        # plt.xlabel("Iterations")
        # plt.ylabel("Cost")
        # plt.show()

log_file.close()