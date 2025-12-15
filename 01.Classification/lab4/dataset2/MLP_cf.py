import os
import pandas as pd
from dslabs_functions import *
from numpy import array, ndarray, arange
from matplotlib.pyplot import figure, savefig
from typing import Literal
from sklearn.neural_network import MLPClassifier

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
def mlp_study(
    trnX: ndarray,
    trnY: array,
    tstX: ndarray,
    tstY: array,
    nr_max_iterations: int = 2500,
    lag: int = 500,
    metric: str = "accuracy",
) -> tuple[MLPClassifier | None, dict]:

    nr_iterations: list[int] = [lag] + [i for i in range(2 * lag, nr_max_iterations + 1, lag)]

    lr_types: list[Literal["constant", "invscaling", "adaptive"]] = ["constant", "invscaling", "adaptive"]
    learning_rates: list[float] = [0.5, 0.05, 0.005, 0.0005]

    best_model: MLPClassifier | None = None
    best_params: dict = {"name": "MLP", "metric": metric, "params": ()}
    best_performance: float = 0.0

    _, axs = subplots(1, len(lr_types), figsize=(len(lr_types) * HEIGHT, HEIGHT), squeeze=False)

    for i, lr_type in enumerate(lr_types):
        values: dict = {}

        for lr in learning_rates:
            y_tst_values: list[float] = []

            for n in nr_iterations:
                clf = MLPClassifier(
                    learning_rate=lr_type,
                    learning_rate_init=lr,
                    max_iter=n,              
                    activation="logistic",
                    solver="sgd",
                    verbose=False,
                    random_state=42,
                )
                clf.fit(trnX, trnY)

                prdY: array = clf.predict(tstX)
                eval_v: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
                y_tst_values.append(eval_v)

                if eval_v - best_performance > DELTA_IMPROVE:
                    best_performance = eval_v
                    best_params["params"] = (lr_type, lr, n)
                    best_model = clf

            values[lr] = y_tst_values

        plot_multiline_chart(
            nr_iterations,
            values,
            ax=axs[0, i],
            title=f"MLP with {lr_type}",
            xlabel="nr iterations",
            ylabel=metric,
            percentage=True,
        )

    print(
        f"MLP best for {best_params['params'][2]} iterations "
        f"(lr_type={best_params['params'][0]} and lr={best_params['params'][1]})"
    )

    return best_model, best_params

# ---------- Models' Comparison ----------
eval_metric = metrics[0]  # change to "recall" if you want
figure()
best_model, params = mlp_study(
    trnX,
    trnY,
    tstX,
    tstY,
    nr_max_iterations=2000,
    lag=250,
    metric=eval_metric,
)
savefig(f"images/{file_tag}_mlp_{eval_metric}_study.png")

# ---------- Best model performance ----------
prd_trn: array = best_model.predict(trnX)
prd_tst: array = best_model.predict(tstX)

figure()
plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels)
savefig(f"images/{file_tag}_{params['name']}_best_{params['metric']}_eval.png")

# ---------- Overfitting study ----------
lr_type: Literal["constant", "invscaling", "adaptive"] = params["params"][0]
lr: float = params["params"][1]

nr_iterations2: list[int] = [250] + [i for i in range(2 * 250, 2000 + 1, 250)]

y_tst_values: list[float] = []
y_trn_values: list[float] = []
acc_metric = "precision"

for n in nr_iterations2:
    clf = MLPClassifier(
        learning_rate=lr_type,
        learning_rate_init=lr,
        max_iter=n,
        activation="logistic",
        solver="sgd",
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
    title=f"MLP overfitting study for lr_type={lr_type} and lr={lr}",
    xlabel="nr_iterations",
    ylabel=str(eval_metric),
    percentage=True,
)
savefig(f"images/{file_tag}_mlp_{eval_metric}_overfitting.png")

# ---------------- Loss Curve -----------------
figure()
plot_line_chart(
    arange(len(best_model.loss_curve_)),
    best_model.loss_curve_,
    title="Loss curve for MLP best model training",
    xlabel="iterations",
    ylabel="loss",
    percentage=False,
)
savefig(f"images/{file_tag}_mlp_{eval_metric}_loss_curve.png")
