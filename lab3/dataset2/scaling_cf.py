import pandas as pd
from pandas import DataFrame, Series, concat
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import figure, savefig
from numpy import ndarray, array
from dslabs_functions import (
    evaluate_approach,
    plot_multibar_chart,
)

# ---- Load dataset ----
filename = "datasets/Combined_Flights_2022_encoded.csv"
file_tag = "Combined_Flights_2022"

target = "Cancelled"
index = "FlightDate"

data = pd.read_csv(filename, index_col=index)

# Separate target and features explicitly
target_data: Series = data[target]        # labels
data_features: DataFrame = data.drop(columns=[target])  # only features

feature_names: list[str] = data_features.columns.to_list()
'''
# ------------------------------
# ---- 1st Scaling: Z-Score ----
# ------------------------------
transf: StandardScaler = StandardScaler(
    with_mean=True, with_std=True, copy=True
).fit(data_features)

# Scale only features
X_scaled: ndarray = transf.transform(data_features)

# Rebuild DataFrame with scaled features
df_zscore: DataFrame = DataFrame(
    X_scaled, index=data.index, columns=feature_names
)

# Add *unscaled* target back
df_zscore[target] = target_data


# Split X and y
y: array = df_zscore[target].to_list()
X: ndarray = df_zscore[feature_names].values

trnX, tstX, trnY, tstY = train_test_split(
    X, y, train_size=0.7, stratify=y, random_state=42
)

# Rebuild train and test DataFrames in the expected format
train: DataFrame = concat(
    [
        DataFrame(trnX, columns=feature_names),
        DataFrame(trnY, columns=[target]),
    ],
    axis=1,
)

test: DataFrame = concat(
    [
        DataFrame(tstX, columns=feature_names),
        DataFrame(tstY, columns=[target]),
    ],
    axis=1,
)

figure()
results: dict[str, list] = evaluate_approach(
    train, test, target=target, metric="recall"
)
plot_multibar_chart(
    ["NB", "KNN"],
    results,
    title=f"{file_tag} evaluation (scaling z-score)",
    percentage=True,
)
savefig(f"images/{file_tag}_eval_sca_z.png")
'''
# ------------------------------
# ---- 2nd Scaling: Min-Max ----
# ------------------------------

from sklearn.preprocessing import MinMaxScaler

# fit Min-Max on features only
transf: MinMaxScaler = MinMaxScaler(feature_range=(0, 1), copy=True).fit(data_features)

X_minmax: ndarray = transf.transform(data_features)

# Build DataFrame with scaled features
df_minmax: DataFrame = DataFrame(
    X_minmax,
    index=data.index,
    columns=feature_names
)

# Add *unscaled* target back
df_minmax[target] = target_data

df_minmax.to_csv(f"datasets/{file_tag}_encoded_scaled.csv", index=index)

# Split X and y explicitly
y: array = df_minmax[target].to_list()
X: ndarray = df_minmax[feature_names].values

trnX, tstX, trnY, tstY = train_test_split(
    X, y, train_size=0.7, stratify=y, random_state=42
)

# Rebuild train and test DataFrames
train: DataFrame = concat(
    [
        DataFrame(trnX, columns=feature_names),
        DataFrame(trnY, columns=[target]),
    ],
    axis=1,
)

test: DataFrame = concat(
    [
        DataFrame(tstX, columns=feature_names),
        DataFrame(tstY, columns=[target]),
    ],
    axis=1,
)

figure()
eval: dict[str, list] = evaluate_approach(train, test, target=target, metric="recall")
plot_multibar_chart(
    ["NB", "KNN"],
    eval,
    title=f"{file_tag} evaluation (scaling min-max)",
    percentage=True,
)
savefig(f"images/{file_tag}_eval_sca_minmax.png")
