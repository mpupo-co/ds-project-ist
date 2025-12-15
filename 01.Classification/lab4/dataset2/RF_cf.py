import os
import pandas as pd
from dslabs_functions import *
from numpy import array, ndarray, std, argsort
from matplotlib.pyplot import figure, savefig, tight_layout
from sklearn.ensemble import RandomForestClassifier

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
vars = feature_cols[:]  # variable names for importance

trnX: ndarray = train_df[feature_cols].values
trnY: array = train_df[target].values

tstX: ndarray = test_df[feature_cols].values
tstY: array = test_df[target].values

# ----------------------------------
def random_forests_study(
    trnX,
    trnY,
    tstX,
    tstY,
    nr_max_trees: int = 2500,
    lag: int = 500,
    metric: str = "accuracy",
):
    n_estimators_list = [100] + [i for i in range(500, nr_max_trees + 1, lag)]
    max_depths = [2, 5, 7]
    max_features_list = [0.1, 0.3, 0.5, 0.7, 0.9]

    best_model = None
    best_params = {"name": "RF", "metric": metric, "params": ()}
    best_performance = 0.0

    cols = len(max_depths)
    _, axs = subplots(1, cols, figsize=(cols * HEIGHT, HEIGHT), squeeze=False)

    for i, d in enumerate(max_depths):
        values = {}

        for f in max_features_list:
            perf_line = []

            clf = RandomForestClassifier(
                n_estimators=n_estimators_list[0],
                max_depth=d,
                max_features=f,
                n_jobs=-1,
                warm_start=True,
                random_state=42,
            )
            clf.fit(trnX, trnY)

            prdY = clf.predict(tstX)
            score = CLASS_EVAL_METRICS[metric](tstY, prdY)
            perf_line.append(score)

            if score > best_performance + DELTA_IMPROVE:
                best_performance = score
                best_model = clf
                best_params["params"] = (d, f, n_estimators_list[0])

            for n in n_estimators_list[1:]:
                clf.n_estimators = n
                clf.fit(trnX, trnY)

                prdY = clf.predict(tstX)
                score = CLASS_EVAL_METRICS[metric](tstY, prdY)
                perf_line.append(score)

                if score > best_performance + DELTA_IMPROVE:
                    best_performance = score
                    best_model = clf
                    best_params["params"] = (d, f, n)

            values[f] = perf_line

        plot_multiline_chart(
            n_estimators_list,
            values,
            ax=axs[0, i],
            title=f"Random Forests with max_depth={d}",
            xlabel="nr estimators",
            ylabel=metric,
            percentage=True,
        )

    print(
        f"RF best for {best_params['params'][2]} trees (d={best_params['params'][0]} and f={best_params['params'][1]})"
    )

    return best_model, best_params

# ---------- Models' Comparison ----------
eval_metric = metrics[0]  # change to "recall" if you want
figure()
best_model, params = random_forests_study(
    trnX,
    trnY,
    tstX,
    tstY,
    nr_max_trees=1000,
    lag=250,
    metric=eval_metric,
)
savefig(f"images/{file_tag}_rf_{eval_metric}_study.png")

# ---------- Best model performance ----------
prd_trn: array = best_model.predict(trnX)
prd_tst: array = best_model.predict(tstX)

figure()
plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels)
savefig(f"images/{file_tag}_{params['name']}_best_{params['metric']}_eval.png")

# ---------- Variables Importance ----------
stdevs: list[float] = list(
    std([tree.feature_importances_ for tree in best_model.estimators_], axis=0)
)
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
    error=stdevs,
    title="RF variables importance",
    xlabel="importance",
    ylabel="variables",
    percentage=True,
)
tight_layout()
savefig(f"images/{file_tag}_rf_{eval_metric}_vars_ranking.png")

# ---------- Overfitting study ----------
d_max: int = params["params"][0]
feat: float = params["params"][1]
nr_estimators: list[int] = [i for i in range(2, 2501, 500)]

y_tst_values: list[float] = []
y_trn_values: list[float] = []
acc_metric = "precision"

for n in nr_estimators:
    clf = RandomForestClassifier(
        n_estimators=n,
        max_depth=d_max,
        max_features=feat,
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(trnX, trnY)

    prd_tst_Y: array = clf.predict(tstX)
    prd_trn_Y: array = clf.predict(trnX)

    y_tst_values.append(CLASS_EVAL_METRICS[acc_metric](tstY, prd_tst_Y))
    y_trn_values.append(CLASS_EVAL_METRICS[acc_metric](trnY, prd_trn_Y))

figure()
plot_multiline_chart(
    nr_estimators,
    {"Train": y_trn_values, "Test": y_tst_values},
    title=f"RF overfitting study for d={d_max} and f={feat}",
    xlabel="nr_estimators",
    ylabel=str(eval_metric),
    percentage=True,
)
savefig(f"images/{file_tag}_rf_{eval_metric}_overfitting.png")
