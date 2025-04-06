import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pywt

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

# use svm regression with grid search for hyperparameter tuning
model = SVR(kernel='rbf')
param_grid = {'C': [0.1, 1, 10, 100], 'epsilon': [0.01, 0.1, 0.5, 1], 'gamma': ['scale', 'auto', 0.01, 0.1, 1, 10]}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring= 'r2')

# fit the model 
grid_search.fit(X_train_scaled, y_train)

# retrieve the best parameters and scores
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f'Best Parameters: {best_params}')
print(f'Best Cross-Validation (R^2) Score: {best_score}')

# use the best model
best_model = grid_search.best_estimator_

# evaluate the best model on the test set
y_pred = best_model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')