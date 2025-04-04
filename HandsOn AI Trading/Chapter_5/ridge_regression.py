import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# generate sample data
np.random.seed(0)
X = np.random.rand(100, 10) # random values w/ 10 features
true_coeffs = np.array([20.5, -1.5, 0, 45, 3, 0, 0, 0, 1, 0])
y = X @ true_coeffs + np.random.randn(100) * 2 # linear combination + noise

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# use ridge regression
model = Ridge(alpha=1.0) # alpha is the regularization strength
model.fit(X_train_scaled, y_train)

# plot the sample chart with the training and test data and the fitted model
plt.figure(figsize=(10, 6))
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)
plt.scatter(y_train, y_pred_train, color='blue', label='Train Data', alpha=0.5)
plt.scatter(y_test, y_pred_test, color='green', label='Test Data', alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, label='Perfect Fit')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Ridge Regression: Predicted vs Actual')
plt.legend()
plt.show()

# plot the sample chart with the training and test data and the fitted model
# for high-dimensional data, we focus on the coefficients
plt.figure(figsize=(10, 6))
plt.plot(model.coef_, marker='o', linestyle='none', color='red', label='Ridge Coefficients')
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.title('Ridge Regression Coefficients')
plt.legend()
plt.show()

# retrieve the model statistics
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# ridge regression coefficients show the impact of regularization
# lower mse and higher r2 indicate a better fit
# the regularization parameter alpha controls the amount of shrinkage


