import os
import pandas as pd
from dslabs_functions import *
from numpy import array, ndarray
from matplotlib.pyplot import figure, savefig
from typing import Literal
from sklearn.neighbors import KNeighborsClassifier

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

# Clean column names
train_df.columns = train_df.columns.str.strip()
test_df.columns = test_df.columns.str.strip()

assert target in train_df.columns, f"{target} missing from train columns"
assert target in test_df.columns, f"{target} missing from test columns"

metrics = ["precision", "recall", "accuracy"]

labels = train_df[target].unique()

feature_cols = [c for c in train_df.columns if c != target]

trnX: ndarray = train_df[feature_cols].values
trnY: array = train_df[target].values

tstX: ndarray = test_df[feature_cols].values
tstY: array = test_df[target].values

# ----------------------------------
def knn_study(
    trnX: ndarray,
    trnY: array,
    tstX: ndarray,
    tstY: array,
    k_max: int = 19,
    lag: int = 2,
    metric: str = "accuracy",
) -> tuple[KNeighborsClassifier | None, dict]:

    dist: list[Literal["manhattan", "euclidean", "chebyshev"]] = ["manhattan", "euclidean", "chebyshev"]
    kvalues: list[int] = [i for i in range(1, k_max + 1, lag)]

    best_model: KNeighborsClassifier | None = None
    best_params: dict = {"name": "KNN", "metric": metric, "params": ()}
    best_performance: float = 0.0

    values: dict[str, list] = {}
    for d in dist:
        y_tst_values: list[float] = []
        for k in kvalues:
            clf = KNeighborsClassifier(n_neighbors=k, metric=d)
            clf.fit(trnX, trnY)

            prdY: array = clf.predict(tstX)
            eval_v: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
            y_tst_values.append(eval_v)

            if eval_v - best_performance > DELTA_IMPROVE:
                best_performance = eval_v
                best_params["params"] = (k, d)
                best_model = clf

        values[d] = y_tst_values

    # fix quoting bug
    print(f"KNN best with k={best_params['params'][0]} and {best_params['params'][1]}")
    plot_multiline_chart(
        kvalues,
        values,
        title=f"KNN Models ({metric})",
        xlabel="k",
        ylabel=metric,
        percentage=True,
    )

    return best_model, best_params

# ---------- Models' Comparison ----------
eval_metric = metrics[0]  # change to "recall" if you want
figure()
best_model, params = knn_study(trnX, trnY, tstX, tstY, k_max=15, metric=eval_metric)
savefig(f"images/{file_tag}_knn_{eval_metric}_study.png")

# ---------- Best model performance ----------
prd_trn: array = best_model.predict(trnX)
prd_tst: array = best_model.predict(tstX)

figure()
plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels)
savefig(f"images/{file_tag}_{params['name']}_best_{params['metric']}_eval.png")

# ---------- Overfitting study ----------
distance: Literal["manhattan", "euclidean", "chebyshev"] = params["params"][1]
K_MAX = 15
kvalues: list[int] = [i for i in range(1, K_MAX, 2)]

y_tst_values: list[float] = []
y_trn_values: list[float] = []
acc_metric = "precision"

for k in kvalues:
    clf = KNeighborsClassifier(n_neighbors=k, metric=distance)
    clf.fit(trnX, trnY)

    prd_tst_Y: array = clf.predict(tstX)
    prd_trn_Y: array = clf.predict(trnX)

    y_tst_values.append(CLASS_EVAL_METRICS[acc_metric](tstY, prd_tst_Y))
    y_trn_values.append(CLASS_EVAL_METRICS[acc_metric](trnY, prd_trn_Y))

figure()
plot_multiline_chart(
    kvalues,
    {"Train": y_trn_values, "Test": y_tst_values},
    title=f"KNN overfitting study for {distance}",
    xlabel="K",
    ylabel=str(eval_metric),
    percentage=True,
)
savefig(f"images/{file_tag}_knn_overfitting.png")
