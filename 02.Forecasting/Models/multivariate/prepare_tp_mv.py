from pandas import DataFrame, read_csv
from sklearn.preprocessing import StandardScaler
from dslabs_functions import *
import numpy as np
import pandas as pd

# --------------------------------------------------
# Configuration
# --------------------------------------------------
FILENAME = "datasets/TrafficTwoMonth_clean.csv"
TIME_COL = "datetime"
TRAIN_PCT = 0.90
DAY_LAG = 24

# --------------------------------------------------
# 1) Load data
# --------------------------------------------------
data: DataFrame = read_csv(FILENAME, index_col=TIME_COL, parse_dates=True)

# --------------------------------------------------
# 2) Hourly aggregation
# --------------------------------------------------
data_hourly: DataFrame = ts_aggregation_by(
    data,
    gran_level="H",
    agg_func="sum"
)

# --------------------------------------------------
# 3) Differencing features by DAY_LAG
# --------------------------------------------------
data_diff = data_hourly.diff(DAY_LAG)
data_diff = data_diff.iloc[DAY_LAG:]  # drop first DAY_LAG rows

# Fill any remaining NaNs in features
data_diff = data_diff.fillna(method='ffill').fillna(method='bfill')


print(f"Differenced data head:\n{data_diff.head(30)}")

# --------------------------------------------------
# 4) Train/Test split (time-aware, DataFrame-safe)
# --------------------------------------------------
def dataframe_temporal_train_test_split(data: DataFrame, trn_pct: float = 0.90) -> tuple[DataFrame, DataFrame]:
    trn_size: int = int(len(data) * trn_pct)
    df_cp: DataFrame = data.copy()
    train: DataFrame = df_cp.iloc[:trn_size]
    test: DataFrame = df_cp.iloc[trn_size:]
    return train, test

trn, tst = dataframe_temporal_train_test_split(data_diff, trn_pct=TRAIN_PCT)
train_df = DataFrame(trn, columns=data_diff.columns)
test_df  = DataFrame(tst, columns=data_diff.columns)

# --------------------------------------------------
# 5) Scaling
# --------------------------------------------------
def scale_all_dataframe(data: DataFrame) -> DataFrame:
    vars: list[str] = data.columns.to_list()
    transf: StandardScaler = StandardScaler().fit(data)
    df = DataFrame(transf.transform(data), index=data.index)
    df.columns = vars
    return df

train_scaled: DataFrame = scale_all_dataframe(train_df)
test_scaled: DataFrame  = scale_all_dataframe(test_df)

# --------------------------------------------------
# 6) Save CSV
# --------------------------------------------------
train_scaled.to_csv("datasets/TrafficTwoMonth_train_prepared_mv.csv")
test_scaled.to_csv("datasets/TrafficTwoMonth_test_prepared_mv.csv")