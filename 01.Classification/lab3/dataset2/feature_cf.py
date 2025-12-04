from pandas import DataFrame, concat
from numpy import ndarray, array
from sklearn.model_selection import train_test_split
from dslabs_functions import select_redundant_variables
import pandas as pd
from dslabs_functions import study_redundancy_for_feature_selection, HEIGHT, apply_feature_selection, evaluate_approach, plot_multibar_chart, select_low_variance_variables, study_variance_for_feature_selection
from matplotlib.pyplot import figure, savefig

# ---- Load dataset ----
filename = 'datasets/Combined_Flights_2022_encoded_sca_smote.csv'
file_tag = 'Combined_Flights_2022'

target = 'Cancelled'
index = 'FlightDate'
data = pd.read_csv(filename, index_col=index)

y = data[target].to_list()
feature_cols = [c for c in data.columns if c != target]

X = data[feature_cols].values

trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y, random_state=42)

train: DataFrame = concat(
    [DataFrame(trnX, columns=feature_cols), DataFrame(trnY, columns=[target])],
    axis=1,
)

test: DataFrame = concat(
    [DataFrame(tstX, columns=feature_cols), DataFrame(tstY, columns=[target])],
    axis=1,
)

# -----------------------------------------------
# ---- Feature selection based on redundancy ----
# -----------------------------------------------
eval_metric = "recall"
'''
# Study

figure(figsize=(2 * HEIGHT, HEIGHT))
study_redundancy_for_feature_selection(
    train,
    test,
    target=target,
    min_threshold=0.1,
    lag=0.05,
    metric=eval_metric,
    file_tag=file_tag,
)
'''

# Apllication

mint = 0.7
vars2drop: list[str] = select_redundant_variables(
    train, min_threshold=mint, target=target
)
vars2drop = [c for c in vars2drop if c != target]
print(f'===========REDUNDANCY=============\nvars2drop (min thres={mint}): {vars2drop}\n')
'''
train_cp, test_cp = apply_feature_selection(
    train, test, vars2drop, filename=f"datasets/{file_tag}", tag="redundant"
)

figure()
eval: dict[str, list] = evaluate_approach(train_cp, test_cp, target=target, metric="recall")
plot_multibar_chart(
    ["NB", "KNN"], eval, title=f"{file_tag} eval. (drop redund - min thr = {mint})", percentage=True
)
savefig(f"images/{file_tag}_eval_feature_redund_{mint}.png")
'''
# ----------------------------------------------
# ---- Feature selection based on relevance ----
# ----------------------------------------------
'''
# Study
figure(figsize=(2 * HEIGHT, HEIGHT))
study_variance_for_feature_selection(
    train,
    test,
    target=target,
    max_threshold=0.2,
    lag=0.02,
    metric=eval_metric,
    file_tag=file_tag,
)
'''
# Aplication
maxt = 0.02
vars2drop: list[str] = select_low_variance_variables(
    train, max_threshold=maxt, target=target
)
vars2drop = [c for c in vars2drop if c != target]
print(f'===========RELEVANCE=============\nvars2drop (max thres={maxt}): {vars2drop}\n')

'''
train_cp, test_cp = apply_feature_selection(
    train, test, vars2drop, filename=f"datasets/{file_tag}", tag="lowvar"
)

figure()
eval: dict[str, list] = evaluate_approach(train_cp, test_cp, target=target, metric="recall")
plot_multibar_chart(
    ["NB", "KNN"], eval, title=f"{file_tag} eval. (drop lowvar - max thr = {maxt})", percentage=True
)
savefig(f"images/{file_tag}_eval_feature_lowvar_{maxt}.png")
'''
data = data.drop(columns=vars2drop)
data.to_csv('datasets/'+file_tag+'_encoded_sca_smote_relevant.csv', index=True)