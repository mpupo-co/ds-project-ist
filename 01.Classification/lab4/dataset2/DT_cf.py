import os
import pandas as pd
from dslabs_functions import *
from numpy import array, argsort, ndarray
from matplotlib.pyplot import figure, savefig
from typing import Literal
from sklearn.tree import DecisionTreeClassifier

# ----------------------------
# Load your pre-split datasets
# ----------------------------
file_tag = "Combined_Flights_2022"
target = "Cancelled"
index = "FlightDate"

train_file = "datasets/Combined_Flights_2022_train_smote_redund_0.7.csv"
test_file  = "datasets/Combined_Flights_2022_test_redund_0.7.csv"

os.makedirs("images", exist_ok=True)

train_df = pd.read_csv(train_file, index_col=index)
test_df  = pd.read_csv(test_file,  index_col=index)

# Clean column names (prevents hidden whitespace bugs)
train_df.columns = train_df.columns.str.strip()
test_df.columns = test_df.columns.str.strip()

assert target in train_df.columns, f"{target} missing from train columns"
assert target in test_df.columns, f"{target} missing from test columns"

metrics = ["precision", "recall", "accuracy"]

labels = train_df[target].unique()

# Build X/y for train and test
trnY: array = train_df[target].values
tstY: array = test_df[target].values

feature_cols = [c for c in train_df.columns if c != target]
vars = feature_cols[:]  # for importance + plot_tree

trnX: ndarray = train_df[feature_cols].values
tstX: ndarray = test_df[feature_cols].values

# ----------------------------------
def trees_study(
    trnX: ndarray,
    trnY: array,
    tstX: ndarray,
    tstY: array,
    d_max: int = 10,
    lag: int = 2,
    metric: str = "accuracy",
) -> tuple:
    criteria: list[Literal["entropy", "gini"]] = ["entropy", "gini"]
    depths: list[int] = [i for i in range(2, d_max + 1, lag)]

    best_model: DecisionTreeClassifier | None = None
    best_params: dict = {"name": "DT", "metric": metric, "params": ()}
    best_performance: float = 0.0

    values: dict = {}
    for c in criteria:
        y_tst_values: list[float] = []
        for d in depths:
            clf = DecisionTreeClassifier(max_depth=d, criterion=c, min_impurity_decrease=0, random_state=42)
            clf.fit(trnX, trnY)
            prdY: array = clf.predict(tstX)
            eval_v: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
            y_tst_values.append(eval_v)

            if eval_v - best_performance > DELTA_IMPROVE:
                best_performance = eval_v
                best_params["params"] = (c, d)
                best_model = clf

        values[c] = y_tst_values

    # fix quoting bug
    print(f"DT best with {best_params['params'][0]} and d={best_params['params'][1]}")
    plot_multiline_chart(
        depths,
        values,
        title=f"DT Models ({metric})",
        xlabel="d",
        ylabel=metric,
        percentage=True,
    )

    return best_model, best_params

# ---------- Models' Comparison ----------
eval_metric = metrics[0]  # or "recall" if you prefer
figure()
best_model, params = trees_study(trnX, trnY, tstX, tstY, d_max=20, metric=eval_metric)
savefig(f"images/{file_tag}_dt_{eval_metric}_study.png")

# ---------- Best model performance ----------
prd_trn: array = best_model.predict(trnX)
prd_tst: array = best_model.predict(tstX)
figure()
plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels)
savefig(f"images/{file_tag}_{params['name']}_best_{params['metric']}_eval.png")

# ---------- Variables importance ----------
from matplotlib.pyplot import tight_layout

importances = best_model.feature_importances_
indices: list[int] = argsort(importances)[::-1]

elems: list[str] = []
imp_values: list[float] = []
for f in range(len(vars)):
    elems.append(vars[indices[f]])
    imp_values.append(importances[indices[f]])
    print(f"{f+1}. {elems[f]} ({importances[indices[f]]})")

figure(figsize=(8, 5))
plot_horizontal_bar_chart(
    elems,
    imp_values,
    title="Decision Tree variables importance",
    xlabel="importance",
    ylabel="variables",
    percentage=True,
)
tight_layout()
savefig(f"images/{file_tag}_dt_{eval_metric}_vars_ranking.png")

# ---------- Overfitting study ----------
crit: Literal["entropy", "gini"] = params["params"][0]
d_max = 25
depths: list[int] = [i for i in range(2, d_max + 1, 1)]
y_tst_values: list[float] = []
y_trn_values: list[float] = []
acc_metric = "precision"

for d in depths:
    clf = DecisionTreeClassifier(max_depth=d, criterion=crit, min_impurity_decrease=0, random_state=42)
    clf.fit(trnX, trnY)

    prd_tst_Y: array = clf.predict(tstX)
    prd_trn_Y: array = clf.predict(trnX)

    y_tst_values.append(CLASS_EVAL_METRICS[acc_metric](tstY, prd_tst_Y))
    y_trn_values.append(CLASS_EVAL_METRICS[acc_metric](trnY, prd_trn_Y))

figure()
plot_multiline_chart(
    depths,
    {"Train": y_trn_values, "Test": y_tst_values},
    title=f"DT overfitting study for {crit}",
    xlabel="max_depth",
    ylabel=str(eval_metric),
    percentage=True,
)
savefig(f"images/{file_tag}_dt_{eval_metric}_overfitting.png")

# ---------- Plot best tree ----------
from sklearn.tree import plot_tree

tree_filename: str = f"images/{file_tag}_dt_{eval_metric}_best_tree"
max_depth2show = 3
st_labels: list[str] = [str(value) for value in labels]

figure(figsize=(14, 6))
plot_tree(
    best_model,
    max_depth=max_depth2show,
    feature_names=vars,
    class_names=st_labels,
    filled=True,
    rounded=True,
    impurity=False,
    precision=2,
)
savefig(tree_filename + ".png")
