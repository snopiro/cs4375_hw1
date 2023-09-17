import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

##################################################
# Initial variable setup
##################################################
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

##################################################
# Function definitions
##################################################

def mse_cost(w, X, y):
    predictions = X.dot(w)
    errors = y - predictions
    return (errors ** 2).mean()

def r_squared(w, X, y):
    predictions = X.dot(w)
    ss_res = np.sum((y - predictions) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - (ss_res / ss_tot)

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

def plot_results(results):
    # Visualization for MSE against Learning Rate
    grouped = results.groupby('learn_rate').mean()
    plt.figure(figsize=(10, 6))
    plt.plot(grouped.index, grouped['mse_train'], marker='o', label='MSE Train')
    plt.plot(grouped.index, grouped['mse_test'], marker='o', label='MSE Test')
    plt.title('MSE against Learning Rate')
    plt.xlabel('Learning Rate')
    plt.ylabel('MSE')
    plt.xscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="--", c='0.7')
    plt.savefig('files/mse_lr.png')

    # Visualization for MSE against Number of Iterations
    grouped_iter = results.groupby('n_iter').mean()
    plt.figure(figsize=(10, 6))
    plt.plot(grouped_iter.index, grouped_iter['mse_train'], marker='o', label='MSE Train')
    plt.plot(grouped_iter.index, grouped_iter['mse_test'], marker='o', label='MSE Test')
    plt.title('MSE against Number of Iterations')
    plt.xlabel('Number of Iterations')
    plt.ylabel('MSE')
    plt.xscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="--", c='0.7')
    plt.savefig('files/mse_iter.png')
    
    # Visualization for MSE against Tolerance
    grouped_tol = results.groupby('tolerance').mean()
    plt.figure(figsize=(10, 6))
    plt.plot(grouped_tol.index, grouped_tol['mse_train'], marker='o', label='MSE Train')
    plt.plot(grouped_tol.index, grouped_tol['mse_test'], marker='o', label='MSE Test')
    plt.title('MSE against Tolerance')
    plt.xlabel('Tolerance')
    plt.ylabel('MSE')
    plt.xscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="--", c='0.7')
    plt.savefig('files/mse_tol.png')

    # Visualization for R^2 against Learning Rate
    grouped = results.groupby('learn_rate').mean()
    plt.figure(figsize=(10, 6))
    plt.plot(grouped.index, grouped['r2_train'], marker='o', label='R^2 Train')
    plt.plot(grouped.index, grouped['r2_test'], marker='o', label='R^2 Test')
    plt.title('R^2 against Learning Rate')
    plt.xlabel('Learning Rate')
    plt.ylabel('R^2')
    plt.xscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="--", c='0.7')
    plt.savefig('files/r2_lr.png')

    # Visualization for R^2 against Number of Iterations
    grouped_iter = results.groupby('n_iter').mean()
    plt.figure(figsize=(10, 6))
    plt.plot(grouped_iter.index, grouped_iter['r2_train'], marker='o', label='R^2 Train')
    plt.plot(grouped_iter.index, grouped_iter['r2_test'], marker='o', label='R^2 Test')
    plt.title('R^2 against Number of Iterations')
    plt.xlabel('Number of Iterations')
    plt.ylabel('R^2')
    plt.xscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="--", c='0.7')
    plt.savefig('files/r2_iter.png')
    
    # Visualization for R^2 against Tolerance
    grouped_tol = results.groupby('tolerance').mean()
    plt.figure(figsize=(10, 6))
    plt.plot(grouped_tol.index, grouped_tol['r2_train'], marker='o', label='R^2 Train')
    plt.plot(grouped_tol.index, grouped_tol['r2_test'], marker='o', label='R^2 Test')
    plt.title('R^2 against Tolerance')
    plt.xlabel('Tolerance')
    plt.ylabel('R^2')
    plt.xscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="--", c='0.7')
    plt.savefig('files/r2_tol.png')

def run_linear_regression(num_runs=1):

    log_file = open('files/grad_descent_part_1.log', 'w')
    results = []
    # Parameters for gradient descent
    learn_rate = [0.01, 0.001, 0.0001]
    n_iter = [10, 100, 1000]
    tolerance = [1e-3, 1e-4, 1e-5]

    # Run the experiment 3 times with different initial weights
    for _ in range(0, num_runs):
        start = np.random.randn(X_train_bias.shape[1])
        log_file.write(f"--- Initial Model ---\n"
                        f"Weights: {start}\n\n"
                        "--- Final Models ---\n")

        for l in learn_rate:    
            for n in n_iter:
                for t in tolerance:

                    # Use gradient descent to find optimal weights
                    optimal_weights_training = gradient_descent(
                        gradient=lambda w: mse_gradient(w, X_train_bias, y_train),
                        start=start,
                        learn_rate=l,
                        n_iter=n,
                        tolerance=t
                    )

                    mse_train = mse_cost(optimal_weights_training, X_train_bias, y_train)
                    mse_test = mse_cost(optimal_weights_training, X_test_bias, y_test)
                    r2_train = r_squared(optimal_weights_training, X_train_bias, y_train)
                    r2_test = r_squared(optimal_weights_training, X_test_bias, y_test)

                    log_entry = (f"Learn Rate: {l}, Iterations: {n}, Tolerance: {t}\n"
                                f"Weights: {optimal_weights_training}\n"
                                f"Training MSE: {mse_train}\n"
                                f"Test MSE: {mse_test}\n"
                                f"Training R^2: {r2_train}\n"
                                f"Test R^2: {r2_test}\n\n")
                    log_file.write(log_entry)
                    results.append({
                        'learn_rate': l,
                        'n_iter': n,
                        'tolerance': t,
                        'mse_train': mse_train,
                        'mse_test': mse_test,
                        'r2_train': r2_train,
                        'r2_test': r2_test
                    })
    
    log_file.close()
    return pd.DataFrame(results)

##################################################
# Main Function
##################################################
if __name__ == '__main__':
    results = run_linear_regression(3)
    plot_results(results)