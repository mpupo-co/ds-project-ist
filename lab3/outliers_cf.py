import numpy as np
from dslabs_functions import evaluate_approach, plot_multibar_chart
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import figure, savefig
from numpy import ndarray, array
from pandas import DataFrame, concat
from pandas import read_csv, DataFrame, Series
from dslabs_functions import (
    NR_STDEV,
    determine_outlier_thresholds_for_var,
)
import pandas as pd

# ---- Load dataset ----
filename = 'datasets/Combined_Flights_2022_encoded.csv'
file_tag = 'Combined_Flights_2022'

index = 'FlightDate'
target = 'Cancelled'
data = pd.read_csv(filename, index_col=index)

print(f"Original data: {data.shape}")


n_std: int = NR_STDEV

numeric_vars = ['Distance', 'CRSElapsedTime']

df: DataFrame = data.copy(deep=True)
summary5: DataFrame = data[numeric_vars].describe()

# ----------------------------------------
# ---- 1st approach: Replace outliers ----
# ----------------------------------------

if [] != numeric_vars:
    df: DataFrame = data.copy(deep=True)
    for var in numeric_vars:
        top, bottom = determine_outlier_thresholds_for_var(summary5[var], std_based=False, threshold=5)
        median: float = df[var].median()
        df[var] = df[var].apply(lambda x: median if x > top or x < bottom else x)
    #df.to_csv(f"datasets/{file_tag}_replacing_outliers.csv", index=True)
    print("Data after replacing outliers:", df.shape)
    #print(df.describe())
else:
    print("There are no numeric variables")

# ---- Split into train and test sets ----
y: array = df.pop(target).to_list()
X: ndarray = df.values
trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y, random_state=42)

train: DataFrame = concat(
    [DataFrame(trnX, columns=df.columns), DataFrame(trnY, columns=[target])], axis=1
)

test: DataFrame = concat(
    [DataFrame(tstX, columns=df.columns), DataFrame(tstY, columns=[target])], axis=1
)

figure()
eval: dict[str, list] = evaluate_approach(train, test, target=target, metric="recall")
plot_multibar_chart(
    ["NB", "KNN"], eval, title=f"{file_tag} evaluation (replacing outliers)", percentage=True
)
savefig(f"images/{file_tag}_eval_out_replace.png")

# -----------------------------------------
# ---- 2nd approach: Drop outliers ----
# -----------------------------------------

n_std: int = NR_STDEV

if numeric_vars is not None:
    df: DataFrame = data.copy(deep=True)
    summary5: DataFrame = data[numeric_vars].describe()
    for var in numeric_vars:
        top_threshold, bottom_threshold = determine_outlier_thresholds_for_var(
            summary5[var], std_based=False, threshold=5
        )
        outliers: Series = df[(df[var] > top_threshold) | (df[var] < bottom_threshold)]
        df.drop(outliers.index, axis=0, inplace=True)
    #df.to_csv(f"data/{file_tag}_drop_outliers.csv", index=True)
    print(f"Data after dropping outliers: {df.shape}")
else:
    print("There are no numeric variables")

# ---- Split into train and test sets ----
y: array = df.pop(target).to_list()
X: ndarray = df.values
trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y, random_state=42)

train: DataFrame = concat(
    [DataFrame(trnX, columns=df.columns), DataFrame(trnY, columns=[target])], axis=1
)

test: DataFrame = concat(
    [DataFrame(tstX, columns=df.columns), DataFrame(tstY, columns=[target])], axis=1
)

figure()
eval: dict[str, list] = evaluate_approach(train, test, target=target, metric="recall")
plot_multibar_chart(
    ["NB", "KNN"], eval, title=f"{file_tag} evaluation (drop outliers)", percentage=True
)
savefig(f"images/{file_tag}_eval_out_drop.png")
