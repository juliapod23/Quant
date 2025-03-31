import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso 
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# generate random sample data
np.random.seed(0)
X = np.random.rand(100, 10)  # 100 samples, 10 features
true_coeffs = np.array([22.5, -1.5, 0, 100, 3, 0, 45, 0, 1, 0]) # true coefficients
Y = X @ true_coeffs + np.random.randn(100) * 2 # linear combination with noise

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# use lasso regression
model = Lasso(alpha=0.1)  # adjusted alpha for better performance
model.fit(X_train_scaled, y_train)

# plot sample chart 
plt.figure(figsize=(10, 6))
y_pred_train = model.predict(X_train_scaled)
plt.scatter(y_train, y_pred_train, color='blue', alpha= 0.5, label='Train')
y_pred_test = model.predict(X_test_scaled)
plt.scatter(y_test, y_pred_test, color='green', alpha= 0.5, label='Test')
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=2, label='Ideal Fit')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Lasso Regression: True vs Predicted Values')
plt.legend()
plt.show()

# plot the coefficients
plt.figure(figsize=(10, 6))
plt.plot(model.coef_, marker='o', linestyle='None', label='Lasso Coefficients')
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.title('Lasso Regression Coefficients')
plt.legend()
plt.show()

# model fit stats
mse = mean_squared_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)
print(f'Mean Squared Error: {mse:.2f}')
print(f'R^2 Score: {r2:.2f}')

# lasso regression coefficients help identify important features; lower mse and higher r2 indicate better model performance
# the regularization parameter (alpha) can be tweaked to control the amount of shrinkage applied to the coefficients 