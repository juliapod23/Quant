import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

def get_weights_ffd(d, thres):
    w, k = [1.], 1
    while True:
        w_ = -w[-1]/k*(d-k+1)
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
    return np.array(w[::-1]).reshape(-1, 1)

def frac_diff_ffd(series, d, thres=1e-5):
    # series is a pandas dataframe
    # d is the differentiation factor
    # thres is the threshold value for cutting off weights
    # returns the differentiated series as a pandas dataframe
    w = get_weights_ffd(d, thres)
    width = len(w)-1
    df = {}
    for name in series.columns:
        series_f, df_ = series[[name]].fillna(method='ffill').dropna(), pd.Series()
        for iloc1 in range(width, series_f.shape[0]):
            loc0, loc1 = series_f.index[iloc1-width], series_f.index[iloc1]
            if not np.isfinite(series.loc[loc1, name]):
                continue
            df_[loc1] = np.dot(w.T, series_f.loc[loc0:loc1])[0, 0]
        df[name] = df_.copy(deep=True)
    df = pd.concat(df, axis=1)
    return df

def ffd(process, thres = 0.01):
    for d in np.linspace(0, 1, 11):
        process_diff = frac_diff_ffd(pd.DataFrame(process), d, thres)
        test_results = adfuller(process_diff.iloc[:, 0], maxlag = 1, regression = 'c', autolag = None)
        if test_results[1] <= 0.05:
            break
    return process_diff[process.name]

# generate sample non-stationary data
np.random.seed(42)
time_series = np.random.randn(100).cumsum()

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
diff_series = ffd(time_series)

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