import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pywt
from sklearn.preprocessing import StandardScaler

# generate sample data
np.random.seed(0)
n = 200
X = np.linspace(0, 20, n).reshape(-1, 1) # feature: random values between 0 and 20
y = 2.5 * np.sin(X).ravel() + np.random.randn(n) * 0.5 # target: non-linear relationship with noise

# apply wavelet transform
coeffs = pywt.wavedec(y, 'db1', level=2)
y_wavelet = pywt.waverec(coeffs, 'db1')

# split data into test and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y_wavelet, test_size=0.2, random_state=0)

# standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# use svm regression
model = SVR(kernel='rbf', C=100, epsilon=.1)
model.fit(X_train_scaled, y_train)

# plot the sample chart with the training data, test data, and the fitted model
plt.figure(figsize=(12, 6))
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.scatter(X_test, y_test, color='green', label='Test Data')
X_plot = np.linspace(0, 20, 200).reshape(-1, 1)
X_plot_scaled = scaler.transform(X_plot)
y_plot = model.predict(X_plot_scaled)
plt.plot(X_plot, y_plot, color='red', label='Fitted Model')
plt.xlabel('Feature (X)')
plt.ylabel('Target (y)')
plt.title('SVM Regression with Wavelet Forecasting')
plt.legend()
plt.show()

# retrieve model fit statistics
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
