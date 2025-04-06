import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

# generate sample data with different volatilities
np.random.seed(0)
n = 200
X = np.linspace(0, 20, n)
regime_1 = 2.5 * X[:n//2] + np.random.randn(n//2) * 5 # higher volatility in regime 1
regime_2 = -1.5 * X[n//2:] + np.random.randn(n//2) * 1 # lower volatility in regime 2
y = np.concatenate([regime_1, regime_2])

# create a DataFrame
data = pd.DataFrame({'X': X, 'Y': y})

# fit a markov switching dynamic regression
model = MarkovRegression(data['Y'], k_regimes=2, trend='c', switching_variance=True)
result = model.fit()

# plot the observed data and the fitted regimes
plt.figure(figsize=(12, 8))

# plot the observed data
plt.plot(data['X'], data['Y'], label='Observed Data', color='blue')

# extract smoothed probabilities
smoothed_probs = result.smoothed_marginal_probabilities

# plot the observed data and regime probabilities
for t in range(len(smoothed_probs)):
    if smoothed_probs.iloc[t][0] > 0.5:
        plt.plot(data['X'].iloc[t], data['Y'].iloc[t], 'ro', alpha=0.5) # regime 1
    else:
        plt.plot(data['X'].iloc[t], data['Y'].iloc[t], 'go', alpha=0.5) # regime 2
        
# highlight the regime changes
regime_changes = np.argmax(smoothed_probs.values, axis=1)
for i in range(1, len(regime_changes)):
    if regime_changes[i] != regime_changes[i-1]:
        plt.axvline(x=data['X'].iloc[i], color='gray', linestyle='--', linewidth=1)
        
plt.xlabel('Feature (X)')
plt.ylabel('Target (Y)')
plt.title('Markov Switching Dynamic Regression with Two Regimes')
plt.legend(['Observed Data', 'Regime 1', 'Regime 2'])
plt.show()

# plot the regime probabilities
plt.figure(figsize=(12, 6))
plt.plot(data['X'], smoothed_probs[0], label='Regime 1 Probability', color='red')
plt.plot(data['X'], smoothed_probs[1], label='Regime 2 Probability', color='green')
plt.xlabel('Feature (X)')
plt.ylabel('Probability')
plt.title('Smoothed Regime Probabilities')
plt.legend()
plt.show()

print(result.summary())