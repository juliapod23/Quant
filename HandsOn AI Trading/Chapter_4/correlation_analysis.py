import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# generate random sample data with correlations
np.random.seed(42)
size = 100
feature1 = np.random.randn(size)
feature2 = feature1 + np.random.randn(size) * 0.1 # high correlation w/ feature1
feature3 = np.random.randn(size) # no correlation with feature1
target = feature1 * 0.5 + np.random.randn(size) * 0.1

data = pd.DataFrame({'feature1': feature1, 'feature2': feature2, 'feature3': feature3, 'target': target})

# calculate the correlation matrix
corr = data.corr()

# display the correlation matrix
print(corr)

# plot heatmap
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()