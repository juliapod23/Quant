import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# generate sample data
np.random.seed(0)
X = 10 * np.random.rand(100, 1)
Y = 2.5 * X**2 + np.random.randn(100, 1) * 2 # target: quadratic relationship with noise

# split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# use polynomial features and linear regression
poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

model = LinearRegression()
model.fit(X_poly_train, Y_train)

# plot the sample chart with the training and test data and the fitted model
plt.figure(figsize=(10, 6))
plt.scatter(X_train, Y_train, color='blue', label='Training data')
plt.scatter(X_test, Y_test, color='green', label='Test data')
X_plot = np.linspace(0, 10, 100).reshape(-1, 1)
Y_plot = model.predict(poly.transform(X_plot))
plt.plot(X_plot, Y_plot, color='red', label='Fitted curve')
plt.title('Polynomial Regression Model')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.legend()
plt.show()

# retrieve model fit statistics
Y_pred = model.predict(X_poly_test)
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# a lower MSE and a higher R^2 score indicate a better fit