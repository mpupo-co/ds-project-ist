import os
import pandas as pd
from pandas import DataFrame
from matplotlib.pyplot import figure, savefig

from dslabs_functions import (
    select_redundant_variables,
    study_redundancy_for_feature_selection,
    apply_feature_selection,
    evaluate_approach,
    plot_multibar_chart,
    select_low_variance_variables,
    study_variance_for_feature_selection,
    HEIGHT,
)

# ----------------
# Load datasets you created
# ----------------
target = "Cancelled"
index_col = "FlightDate"
file_tag = "Combined_Flights_2022"

train: DataFrame = pd.read_csv(
    f"datasets/{file_tag}_train_smote.csv",
    index_col=index_col
)
test: DataFrame = pd.read_csv(
    f"datasets/{file_tag}_test.csv",
    index_col=index_col
)

# clean column names (prevents "Cancelled" not found due to whitespace)
train.columns = train.columns.str.strip()
test.columns = test.columns.str.strip()

# fail fast if target missing
assert target in train.columns, f"{target} missing from train columns"
assert target in test.columns, f"{target} missing from test columns"

eval_metric = "recall"
'''
# -----------------------------------------------
# Feature selection based on redundancy
# -----------------------------------------------
figure(figsize=(2 * HEIGHT, HEIGHT))
study_redundancy_for_feature_selection(
    train.copy(),              # copies because downstream may mutate
    test.copy(),
    target=target,
    min_threshold=0.1,
    lag=0.05,
    metric=eval_metric,
    file_tag=file_tag,
)
'''
mint = 0.7
vars2drop: list[str] = select_redundant_variables(train, min_threshold=mint, target=target)
vars2drop = [c for c in vars2drop if c != target]

print(f"===========REDUNDANCY=============\nvars2drop (min thres={mint}): {vars2drop}\n")

# Apply feature selection (drops columns from both train and test consistently)
train_red, test_red = apply_feature_selection(
    train.copy(), test.copy(), vars2drop, filename=f"datasets/{file_tag}", tag="redundant"
)

# (optional) save the reduced versions
train_red.to_csv(f"datasets/{file_tag}_train_smote_redund_{mint}.csv", index=True)
test_red.to_csv(f"datasets/{file_tag}_test_redund_{mint}.csv", index=True)
'''
figure()
eval_red: dict[str, list] = evaluate_approach(train_red.copy(), test_red.copy(), target=target, metric=eval_metric)
plot_multibar_chart(
    ["NB", "KNN"],
    eval_red,
    title=f"{file_tag} eval. (drop redund - min thr = {mint})",
    percentage=True
)
savefig(f"images/{file_tag}_eval_feature_redund_{mint}.png")

# ----------------------------------------------
# Feature selection based on low variance
# ----------------------------------------------
figure(figsize=(2 * HEIGHT, HEIGHT))
study_variance_for_feature_selection(
    train.copy(),
    test.copy(),
    target=target,
    max_threshold=0.2,
    lag=0.02,
    metric=eval_metric,
    file_tag=file_tag,
)

maxt = 0.18
vars2drop: list[str] = select_low_variance_variables(train, max_threshold=maxt, target=target)
vars2drop = [c for c in vars2drop if c != target]

print(f"===========LOW VARIANCE=============\nvars2drop (max thres={maxt}): {vars2drop}\n")

train_lowv, test_lowv = apply_feature_selection(
    train.copy(), test.copy(), vars2drop, filename=f"datasets/{file_tag}", tag="lowvar"
)

# (optional) save the reduced versions
#train_lowv.to_csv(f"datasets/{file_tag}_train_under_lowvar_{maxt}.csv", index=True)
#test_lowv.to_csv(f"datasets/{file_tag}_test_lowvar_{maxt}.csv", index=True)

figure()
eval_lowv: dict[str, list] = evaluate_approach(train_lowv.copy(), test_lowv.copy(), target=target, metric=eval_metric)
plot_multibar_chart(
    ["NB", "KNN"],
    eval_lowv,
    title=f"{file_tag} eval. (drop lowvar - max thr = {maxt})",
    percentage=True
)
savefig(f"images/{file_tag}_eval_feature_lowvar_{maxt}.png")
'''