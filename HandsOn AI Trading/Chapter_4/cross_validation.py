import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# load the wine dataset
data = load_wine()
X, Y = data.data, data.target

# initialize the model
model = RandomForestClassifier(random_state=42)

# perform 5-fold cross-validation
scores = cross_val_score(model, X, Y, cv=5)
print(f'Cross-validation scores: {scores}')
print(f'Mean cross-validation score: {np.mean(scores)}')

# split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# train the model
model.fit(X_train, Y_train)

# test the model
Y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
print(f'Test accuracy: {accuracy}')

# compare the results
print('\nComparison of Results:')
print(f'Mean cross-validation score: {scores.mean():.4f}')
print(f'Train/Test split accuracy: {accuracy:.4f}')