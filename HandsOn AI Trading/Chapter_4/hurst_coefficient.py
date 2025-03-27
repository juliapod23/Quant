from QuantConnect import Resolution
import numpy as np 
import pandas as pd 
from statsmodels.tsa.stattools import adfuller, coint
import matplotlib.pyplot as plt
from hurst import compute_Hc

# H < 0.5: mean-reverting, eventually returns to long-term avg
# H = 0.5: random walk
# H > 0.5: trending

def initialize(self) -> None:
    # Set up the start date, end date, and cash
    self.set_start_date(2022, 1, 1)
    self.set_end_date(2023, 1, 1)
    self.set_cash(100000)

    # Add securities, set benchmarks, etc.
    self.add_equity("SPY", Resolution.DAILY)
    self.set_benchmark("SPY")
    self.add_equity("VOO", Resolution.DAILY)
    self.set_benchmark("VOO")

# generate synthetic cointegrated time series data
np.random.seed(42)
n = 1000
time = np.arange(n)
# generate stochastic trend
trend = np.cumsum(np.random.randn(n))

# two cointegrated time series with added noise
asset1 = trend + 0.3 * np.random.randn(n)
asset2 = trend + 0.1 * np.random.randn(n)

data = pd.DataFrame({'asset1': asset1, 'asset2': asset2})

# function to perform ADF test
def adf_test(series, name):
    result = adfuller(series)
    print(f'ADF Statistic for {name}: {result[0]}')
    print(f'P-Value for {name}: {result[1]}')
    for key, value in result[4].items():
        print(f'Critical Value {key}: {value}')
    print('\n')

# test each series for stationarity
print("ADF Test for asset1: ")
adf_test(data['asset1'], 'asset1')
print("ADF Test for asset2: ")
adf_test(data['asset2'], 'asset2')

# perform the Engle-Grainger cointegration test
score, pvalue, _ = coint(data['asset1'], data['asset2'])

print(f'Engler-Granger Cointegration Test score: {score}')
print(f'Engle-Granger Cointegration Test p-value: {pvalue}\n')

# calculate spread
data['spread'] = data['asset1'] - data['asset2']

# test spread for stationarity
print("ADF Test for spread: ")
adf_test(data['spread'], 'spread')

# remove 0 and negative values in the spread for Hurst calculation
spread_non_zero = data['spread'][data['spread'] > 0]
if len(spread_non_zero) < len(data['spread']):
    print("Non-positive values in the spread were removed for Hurst calculation.")

# ensure spread_non_zero is not empty and does not contain invalid values
if len(spread_non_zero) > 0 and (spread_non_zero <= 0).sum() == 0:
    # calculate the Hurst exponent of the spread
    H, c, data_hurst = compute_Hc(spread_non_zero, kind='price', simplified=True)
    print(f'Hurst Exponent for spread: {H}')

    # visualize the series and their spread
    plt.figure(figsize=(14, 7))
    plt.subplot(2, 1, 1)
    plt.plot(data['asset1'], label = 'Asset 1')
    plt.plot(data['asset2'], label = 'Asset 2')
    plt.legend()
    plt.title('Cointegrated Time Series')

    plt.subplot(2, 1, 2)
    plt.plot(data['spread'], label = 'Spread (Asset 1 - Asset 2)')
    plt.legend()
    plt.title('Spread (Should Be Stationary)')
    plt.tight_layout()
    plt.show()
else:
    print("Error: Spread contains invalid values or is empty after removing non-positive values.")