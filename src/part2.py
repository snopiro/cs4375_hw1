import pandas as pd  
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Grab data from source
data_training = pd.read_csv('https://personal.utdallas.edu/~cwk200000/files/auto-mpg-training.csv')
data_test = pd.read_csv('https://personal.utdallas.edu/~cwk200000/files/auto-mpg-test.csv')

# Format the data into x and y axis
X_train = data_training.drop('mpg', axis=1)
y_train = data_training['mpg']
X_test = data_test.drop('mpg', axis=1)
y_test = data_test['mpg']

# Create a linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Predict on the sets
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

log_file = open('grad_descent_part_2.log', 'w')

# Evaluate the model using Mean Squared Error (MSE)
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
log_entry = (f"Weights: {model.coef_}\n"
             f"Training MSE: {mse_train}, Test MSE: {mse_test}\n\n")
log_file.write(log_entry)

log_file.close()