import pandas as pd  
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

##################################################
# Initial variable setup
##################################################
# Grab data from source
data_training = pd.read_csv('https://personal.utdallas.edu/~cwk200000/files/auto-mpg-training.csv')
data_test = pd.read_csv('https://personal.utdallas.edu/~cwk200000/files/auto-mpg-test.csv')

# Format the data into x and y axis
X_train = data_training.drop('mpg', axis=1)
y_train = data_training['mpg']
X_test = data_test.drop('mpg', axis=1)
y_test = data_test['mpg']

##################################################
# Function definitions
##################################################

def run_linear_regression():
    # Create a linear regression model
    model = LinearRegression()

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Predict on the sets
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    log_file = open('files/grad_descent_part_2.log', 'w')

    # Evaluate the model using Mean Squared Error (MSE)
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_train = model.score(X_train, y_train)
    r2_test = model.score(X_test, y_test)

    log_entry = ("--- Scikit Linear Regression ---\n"
                f"Weights: {model.coef_}\n"
                f"Training MSE: {mse_train}\n"
                f"Test MSE: {mse_test}\n"
                f"Training R^2: {r2_train}\n"
                f"Test R^2: {r2_test}\n\n")
    log_file.write(log_entry)

    log_file.close()

##################################################
# Main Function
##################################################

if __name__ == '__main__':
    run_linear_regression()