import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

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

# remove highly correlated features
corr = X.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool_))
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
X_reduced = X.drop(columns = to_drop)

# train a random forest model
model = RandomForestRegressor()
model.fit(X_reduced, Y)

# get feature importances
importances = model.feature_importances_
feature_names = X_reduced.columns
importances_df = pd.DataFrame({'feature': feature_names, 'importance': importances})

# display feature importances
print(importances_df)

# plot feature importances
importances_df = importances_df.sort_values('importance', ascending=False).plot(kind='bar', x='feature', y='importance')
plt.title('Feature Importances')
plt.show()
