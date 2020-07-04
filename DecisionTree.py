# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('/Users/tharunpeddisetty/Desktop/Machine Learning A-Z (Codes and Datasets)/Part 2 - Regression/Section 6 - Polynomial Regression/Python/Position_Salaries.csv')
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:, -1].values

#Training the Decision Tree Model
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X,y)

#Prediction for level 6.5
regressor.predict([[6.5]])

#Visualizing the Regression Results in High resolution. If we have dimensions > 2, we can't really plot it. Also, it makes no sense.
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid,regressor.predict(X_grid), color = 'blue')
plt.title('Decision Tree')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#visualizing normally makes no sense.Since predicted would be the same value; we used the entire data to run this regression
plt.scatter(X, y, color = 'red')
plt.plot(X,regressor.predict(X), color = 'blue')
plt.title('Decision Tree')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()