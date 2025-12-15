import os
import pandas as pd
from dslabs_functions import *
from numpy import array, ndarray
from matplotlib.pyplot import figure, savefig
from sklearn.linear_model import LogisticRegression

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

feature_cols = [c for c in train_df.columns if c != target]

trnX: ndarray = train_df[feature_cols].values
trnY: array = train_df[target].values

tstX: ndarray = test_df[feature_cols].values
tstY: array = test_df[target].values

# ----------------------------------
def logistic_regression_study(
    trnX: ndarray,
    trnY: array,
    tstX: ndarray,
    tstY: array,
    nr_max_iterations: int = 2500,
    lag: int = 500,
    metric: str = "accuracy",
) -> tuple[LogisticRegression | None, dict]:

    nr_iterations: list[int] = [lag] + [i for i in range(2 * lag, nr_max_iterations + 1, lag)]
    penalty_types: list[str] = ["l1", "l2"]  # only available if solver='liblinear'

    best_model: LogisticRegression | None = None
    best_params: dict = {"name": "LR", "metric": metric, "params": ()}
    best_performance: float = 0.0

    values: dict = {}
    for pen in penalty_types:
        y_tst_values: list[float] = []
        for it in nr_iterations:
            clf = LogisticRegression(
                penalty=pen,
                max_iter=it,              # ✅ important: use the iteration we are testing
                solver="liblinear",
                verbose=False,
                random_state=42,
            )
            clf.fit(trnX, trnY)
            prdY: array = clf.predict(tstX)
            eval_v: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
            y_tst_values.append(eval_v)

            if eval_v - best_performance > DELTA_IMPROVE:
                best_performance = eval_v
                best_params["params"] = (pen, it)
                best_model = clf

        values[pen] = y_tst_values

    plot_multiline_chart(
        nr_iterations,
        values,
        title=f"LR models ({metric})",
        xlabel="nr iterations",
        ylabel=metric,
        percentage=True,
    )
    print(f"LR best for {best_params['params'][1]} iterations (penalty={best_params['params'][0]})")

    return best_model, best_params

# ---------- Models' Comparison ----------
eval_metric = metrics[0]  # change to "recall" if you want
figure()

best_model, params = logistic_regression_study(
    trnX,
    trnY,
    tstX,
    tstY,
    nr_max_iterations=2000,
    lag=250,
    metric=eval_metric,
)
savefig(f"images/{file_tag}_lr_{eval_metric}_study.png")

# ---------- Best model performance ----------
prd_trn: array = best_model.predict(trnX)
prd_tst: array = best_model.predict(tstX)

figure()
plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels)
savefig(f"images/{file_tag}_{params['name']}_best_{params['metric']}_eval.png")

# ---------- Overfitting study ----------
penalty: str = params["params"][0]
nr_iterations2: list[int] = [i for i in range(100, 1001, 100)]

y_tst_values: list[float] = []
y_trn_values: list[float] = []
acc_metric = "precision"

for n in nr_iterations2:
    clf = LogisticRegression(
        penalty=penalty,
        max_iter=n,
        solver="liblinear",
        verbose=False,
        random_state=42,
    )
    clf.fit(trnX, trnY)
    prd_tst_Y: array = clf.predict(tstX)
    prd_trn_Y: array = clf.predict(trnX)

    y_tst_values.append(CLASS_EVAL_METRICS[acc_metric](tstY, prd_tst_Y))
    y_trn_values.append(CLASS_EVAL_METRICS[acc_metric](trnY, prd_trn_Y))

figure()
plot_multiline_chart(
    nr_iterations2,
    {"Train": y_trn_values, "Test": y_tst_values},
    title=f"LR overfitting study for penalty={penalty}",
    xlabel="nr_iterations",
    ylabel=str(eval_metric),
    percentage=True,
)
savefig(f"images/{file_tag}_lr_{eval_metric}_overfitting.png")
