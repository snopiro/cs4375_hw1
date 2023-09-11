import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd  
import seaborn as sns 
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing


california_housing = fetch_california_housing(as_frame=True)


california_housing.frame.hist(figsize=(12, 10), bins=30, edgecolor="black")
plt.subplots_adjust(hspace=0.7, wspace=0.4)

df = california_housing.frame
# X = df.drop(['MedHouseVal', 'Latitude', 'Longitude', 'Population'], axis = 1)

# print(df)

# print(X)

# M = df.corr()
# print(M)

X = df[['MedInc', 'HouseAge', 'AveRooms']]
Y = df[['MedHouseVal']]
print(X)

model = LinearRegression().fit(X, Y)
print(model.coef_)
print(model.intercept_)

#y =  .02 + .443 * MedInc + .017 * HouseAge - .027 * AveRooms