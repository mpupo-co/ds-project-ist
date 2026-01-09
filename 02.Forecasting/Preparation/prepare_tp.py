from pandas import DataFrame, read_csv, Series
from sklearn.preprocessing import StandardScaler
from dslabs_functions import *

'''
The original 15-minute traffic count data was aggregated to an hourly level using summation 
to reduce noise while preserving daily traffic patterns. Daily differencing (lag = 24) was 
applied to remove the dominant seasonal component and improve stationarity. No smoothing 
techniques were applied, as aggregation and differencing were sufficient for noise reduction, 
and smoothing could distort the autocorrelation structure required by ARIMA and remove 
informative peaks for LSTM models. The same prepared dataset was used for both models to 
ensure a fair comparison.
'''

# -----------------------------------------------
# Configuration
# -----------------------------------------------
file_tag = "TrafficTwoMonth"
filename = "datasets/TrafficTwoMonth_clean.csv"
target = "Total"
time_col = "datetime"
TRAIN_PCT = 0.90

# -----------------------------------------------
# Load data
# -----------------------------------------------
# The original dataset has 15-minute frequency traffic counts.
# The datetime column is used as the time index to preserve temporal order.
data: DataFrame = read_csv(
    filename,
    index_col=time_col,
    parse_dates=True,
)

# We work only with the target variable to keep a unified
# preprocessing pipeline suitable for both ARIMA and LSTM.
series: Series = data[target]

# -----------------------------------------------
# 1) Aggregation: Hourly SUM
# -----------------------------------------------
# Hourly aggregation reduces high-frequency noise present in 15-minute data
# while preserving the daily traffic pattern (rush hours).
# SUM is used because traffic variables represent counts.
series_hourly: Series = ts_aggregation_by(
    series, gran_level="H", agg_func="sum"
)

# -----------------------------------------------
# 2) Differencing: Daily differencing (lag = 24)
# -----------------------------------------------
# After hourly aggregation, the dominant seasonality is daily (24 hours).
# Daily differencing removes this seasonal component and helps stationarity,
# which is required by ARIMA and acceptable for LSTM when using a unified pipeline.
DAY_LAG = 24
series_diff: Series = series_hourly.diff(periods=DAY_LAG).dropna()
diff_df: DataFrame = series_diff.to_frame(name=target)

# -----------------------------------------------
# 3) Train/Test split (time-aware)
# -----------------------------------------------
# The split preserves temporal ordering (no shuffling),
# which is mandatory for forecasting problems.
train, test = series_train_test_split(
    diff_df, trn_pct=TRAIN_PCT
)

# -----------------------------------------------
# 4) Scaling (fit on TRAIN only)
# -----------------------------------------------
# Scaling is required for LSTM training and does not harm ARIMA
# as long as it is applied consistently.
# The scaler is fit only on the training set to avoid data leakage.
scaler = StandardScaler()

train_scaled = Series(
    scaler.fit_transform(train.to_frame()).ravel(),
    index=train.index,
    name=target
)

test_scaled = Series(
    scaler.transform(test.to_frame()).ravel(),
    index=test.index,
    name=target
)

# -----------------------------------------------
# 5) Smoothing: NOT applied
# -----------------------------------------------
# No smoothing (e.g., rolling mean) is applied because:
# - It distorts the autocorrelation structure required by ARIMA
# - It removes sharp traffic peaks (rush hours) that are informative
#   for LSTM learning
# - Aggregation and differencing already provide sufficient noise reduction

# -----------------------------------------------
# 6) Save train and test sets to CSV
# -----------------------------------------------
train_scaled.to_csv(
    f"datasets/{file_tag}_train_prepared.csv",
    header=True
)

test_scaled.to_csv(
    f"datasets/{file_tag}_test_prepared.csv",
    header=True
)
