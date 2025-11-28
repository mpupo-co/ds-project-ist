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
y: array = data.pop(target).to_list()
X: ndarray = data.values
trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y, random_state=42)

train: DataFrame = concat(
    [DataFrame(trnX, columns=data.columns), DataFrame(trnY, columns=[target])], axis=1
)

test: DataFrame = concat(
    [DataFrame(tstX, columns=data.columns), DataFrame(tstY, columns=[target])], axis=1
)

# -----------------------------------------------
# ---- Feature selection based on redundancy ----
# -----------------------------------------------
eval_metric = "recall"

# Study

figure(figsize=(2 * HEIGHT, HEIGHT))
study_variance_for_feature_selection(
    train,
    test,
    target=target,
    max_threshold=0.5,
    lag=0.005,
    metric=eval_metric,
    file_tag=file_tag,
)
'''
# Apllicance
print("Original variables", train.columns.values)
vars2drop: list[str] = select_redundant_variables(
    train, target=target, min_threshold=0.1
)
print("Variables to drop (redundant)", vars2drop)

vars2drop: list[str] = select_redundant_variables(
    train, min_threshold=0.7, target=target
)
train_cp, test_cp = apply_feature_selection(
    train, test, vars2drop, filename=f"datasets/{file_tag}", tag="redundant"
)

figure()
eval: dict[str, list] = evaluate_approach(train_cp, test_cp, target=target, metric="recall")
plot_multibar_chart(
    ["NB", "KNN"], eval, title=f"{file_tag} eval. (drop redund - min thr = 0.1)", percentage=True
)
savefig(f"images/{file_tag}_eval_feature_redund.png")

# ----------------------------------------------
# ---- Feature selection based on relevance ----
# ----------------------------------------------

vars2drop: list[str] = select_low_variance_variables(
    train, max_threshold=0.02, target=target
)
train_cp, test_cp = apply_feature_selection(
    train, test, vars2drop, filename=f"datasets/{file_tag}", tag="lowvar"
)

figure()
eval: dict[str, list] = evaluate_approach(train_cp, test_cp, target=target, metric="recall")
plot_multibar_chart(
    ["NB", "KNN"], eval, title=f"{file_tag} eval. (drop lowvar - min thr = 0.02)", percentage=True
)
savefig(f"images/{file_tag}_eval_feature_lowvar_0.02.png")
'''