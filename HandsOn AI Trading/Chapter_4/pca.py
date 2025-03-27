import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# generate random sample data with correlations
np.random.seed(42)
size = 100
feature1 = np.random.randn(size)
feature2 = feature1 + np.random.randn(size) * 0.1 # high correlation w/ feature1
feature3 = np.random.randn(size) # no correlation with feature1
target = feature1 * 0.5 + np.random.randn(size) * 0.1

data = pd.DataFrame({'feature1': feature1, 'feature2': feature2, 'feature3': feature3, 'target': target})
X = data.drop('target', axis=1)

# standardize the data
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# apply PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_standardized)

# create a DataFrame with the principal components
principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# display the explained variance ratio
print(f'Explained variance ratio: {pca.explained_variance_ratio_}')

# plot the principal components
plt.scatter(principal_df['PC1'], principal_df['PC2'])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Principal Component Analysis')
plt.show()