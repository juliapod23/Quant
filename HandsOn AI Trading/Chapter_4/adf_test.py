import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# generate sample non-stationary data
np.random.seed(42)
time_series = np.random.randn(1000).cumsum()

# perform ADF test
result = adfuller(time_series)
print(f'ADF Statistic: {result[0]}')
print(f'P-Value: {result[1]}')
for key, value in result[4].items():
    print(f'Critical Value {key}: {value}')
    
# plot the time series
plt.figure(figsize=(12, 6))
plt.plot(time_series, label = 'Original Time Series')
plt.title('Non-Stationary Time Series')
plt.legend()
plt.show()

# differencing to make the time series stationary
diff_series = np.diff(time_series, n = 1)

# perform ADF test on the differenced series
result_diff = adfuller(diff_series)
print(f'ADF Statistic after differencing: {result_diff[0]}')
print(f'P-Value after differencing: {result_diff[1]}')
for key, value in result_diff[4].items():
    print(f'Critical Value {key}: {value}')
    
# plot the differenced series
plt.figure(figsize=(12, 6))
plt.plot(diff_series, label = 'Differenced Time Series')
plt.title('Stationary Time Series After Differencing')
plt.legend()
plt.show()

# if the test statistic is less than the critical value at a certain confidence level, we reject the null hypothesis
# and conclude that the time series is stationary
# if the test statistic is greater than the critical value, we fail to reject the null hypothesis and conclude that
# the time series is non-stationary
