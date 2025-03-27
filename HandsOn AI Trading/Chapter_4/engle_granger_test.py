import numpy as np
import pandas as pd 
from statsmodels.tsa.stattools import coint, adfuller
import matplotlib.pyplot as plt

# generate synthetic time series data
np.random.seed(42)
n = 100
time = np.arange(n)

# simulate two non-stationary time series (random walks)
asset1 = np.cumsum(np.random.randn(n)) + 41
asset2 = asset1 + np.random.randn(n)

# create a data frame
data = pd.DataFrame({'asset1': asset1, 'asset2': asset2})

# function to perform ADF test
def adf_test(series, name):
    result = adfuller(series)
    print(f'ADF Statistic for {name}: {result[0]}')
    print(f'p-value: {result[1]}')
    for key, value in result[4].items():
        print('Critial Values:')
        print(f'   {key}, {value}')
    print('\n')
    
# Step 1: Check if the time series are stationary
print('ADF Test for Asset 1:')
adf_test(data['asset1'], 'asset1')
print('ADF Test for Asset 2:')
adf_test(data['asset2'], 'asset2')

# Step 2: Check if the time series are cointegrated using the Engle-Granger test
score, pvalue, _ = coint(data['asset1'], data['asset2'])
print(f'Cointegration test p-value: {pvalue}')
print(f'Cointegration test score: {score}\n')

# Step 3: Plot the time series and the spread
data['spread'] = data['asset1'] - data['asset2']

plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(time, data['asset1'], label='Asset 1')
plt.plot(time, data['asset2'], label='Asset 2')
plt.legend()
plt.title('Non-Stationary Time Series')

plt.subplot(2, 1, 2)
plt.plot(time, data['spread'], label='Spread')
plt.legend()
plt.title('Spread, Should be Stationary')
plt.tight_layout()
plt.show()

# Step 4: Check if the spread is stationary
print('ADF Test for Spread:')
adf_test(data['spread'], 'spread')