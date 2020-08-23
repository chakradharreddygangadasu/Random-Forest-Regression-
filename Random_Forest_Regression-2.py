# Random Forest Regression model

"""Building Random Forest Regression model to predict the salary of employee by knowing the years of experiance and 
corresponding position and comparing the model with all the regression models."""

# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

#importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2:3].values

#Creating the Random Forest Regression model
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 1)
regressor.fit(x,y)

#Predictions using the Random Forest Regression model
y_pred = regressor.predict([[6.5]])
# the predicted values is 165000 which is good but, this can be further improved by increasing number of trees.

# Visualizing the prediction model
x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape((len(x_grid)),1)
plt.scatter(x,y,color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('Experience vs Salary with Random Forest Regressor model')
plt.xlabel('experiance')
plt.ylabel('salary')
plt.show()

#Model with incresing number of trees
regressor = RandomForestRegressor(n_estimators = 300, random_state = 1)
regressor.fit(x,y)

#Visualizing with more trees
x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape((len(x_grid)),1)
plt.scatter(x,y,color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('Experience vs Salary with Random Forest Regressor model')
plt.xlabel('experiance')
plt.ylabel('salary')
plt.show()

# it contains more number of steps than the previous one.

# Predictins with large number of trees
y_pred = regressor.predict([[6.5]])

# shows more steps then the model with low number of estimators(Trees)
