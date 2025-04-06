import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# generate sample data
np.random.seed(0)
X = np.random.rand(100, 1) * 10 # feature: random values between 0 and 10
y = 2.5 * np.sin(X).ravel() + np.random.randn(100) * 0.5 # target: non-linear relationship with noise

# split data into test and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# fit a decision tree regressor
model = DecisionTreeRegressor(max_depth=3, random_state=0)
model.fit(X_train, y_train)

# plot sample chart with the training data, test data, and the fitted model
plt.figure(figsize=(12, 6))
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.scatter(X_test, y_test, color='green', label='Test Data')
X_plot = np.linspace(0, 10, 100).reshape(-1, 1)
y_plot = model.predict(X_plot)
plt.plot(X_plot, y_plot, color='red', label='Fitted Model')
plt.xlabel('Feature (X)')
plt.ylabel('Target (y)')
plt.title('Decision Tree Regression')
plt.legend()
plt.show()

# display generated tree
plt.figure(figsize=(12, 6))
plot_tree(model, filled=True)
plt.title('Decision Tree Structure')
plt.show()

# retrieve model fit statistics
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# lower mse and higher r2 indicate better fit
# pruning can help reduce overfitting and improve model performance