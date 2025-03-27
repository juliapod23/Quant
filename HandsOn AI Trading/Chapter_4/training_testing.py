import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE

# generate random sample data with correlations
np.random.seed(42)
size = 100
feature1 = np.random.randn(size)
feature2 = feature1 + np.random.randn(size) * 0.1 # high correlation w/ feature1
feature3 = np.random.randn(size) # no correlation with feature1
target = feature1 * 0.5 + np.random.randn(size) * 0.1

data = pd.DataFrame({'feature1': feature1, 'feature2': feature2, 'feature3': feature3, 'target': target})
X = data.drop('target', axis=1)
Y = data['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# print the shapes of the training and testing sets
print(f'Training set shape: {X_train.shape}')
print(f'Testing set shape: {X_test.shape}')
print(f'Training target shape: {Y_train.shape}')
print(f'Testing target shape: {Y_test.shape}')
