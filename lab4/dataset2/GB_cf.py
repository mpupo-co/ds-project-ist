import pandas as pd
from dslabs_functions import *
from sklearn.model_selection import train_test_split
from numpy import array, argsort
from matplotlib.pyplot import figure, savefig
from typing import Literal
from sklearn.ensemble import GradientBoostingClassifier

filename = 'datasets/Combined_Flights_2022_prep.csv'
file_tag = 'Combined_Flights_2022'

target = 'Cancelled'
index = 'FlightDate'
data = pd.read_csv(filename, index_col=index)

metrics = ['accuracy', 'precision', 'recall']

labels = data[target].unique()
y = data.pop(target).values    
X = data.values

vars = data.columns.to_list() 
# ----------------------------------

from typing import Tuple, Dict, Any
from numpy import ndarray
from sklearn.ensemble import GradientBoostingClassifier

def gradient_boosting_study(
    trnX: ndarray,
    trnY: ndarray,
    tstX: ndarray,
    tstY: ndarray,
    nr_max_trees: int = 2500,
    lag: int = 500,
    metric: str = "accuracy",
) -> Tuple[GradientBoostingClassifier | None, Dict[str, Any]]:
    # Grid of hyperparameters
    n_estimators_grid: list[int] = [100] + [i for i in range(500, nr_max_trees + 1, lag)]
    max_depths: list[int] = [2, 5, 7]
    learning_rates: list[float] = [0.1, 0.3, 0.5, 0.7, 0.9]

    metric_fn = CLASS_EVAL_METRICS[metric]

    best_model: GradientBoostingClassifier | None = None
    best_params: Dict[str, Any] = {"name": "GB", "metric": metric, "params": ()}
    best_performance: float = 0.0

    cols: int = len(max_depths)
    _, axs = subplots(1, cols, figsize=(cols * HEIGHT, HEIGHT), squeeze=False)

    # Precompute mapping from stage index -> position in our grid
    idx_map = {n: i for i, n in enumerate(n_estimators_grid)}

    for col_idx, d in enumerate(max_depths):
        values: Dict[float, list[float]] = {}  # lr -> scores
        ax = axs[0, col_idx]

        for lr in learning_rates:
            # Fit ONCE with the maximum number of estimators
            clf = GradientBoostingClassifier(
                n_estimators=nr_max_trees,
                max_depth=d,
                learning_rate=lr,
                # optional but good for reproducibility:
                # random_state=0
            )
            clf.fit(trnX, trnY)

            # Prepare container for scores aligned with n_estimators_grid
            scores = [0.0] * len(n_estimators_grid)

            # Iterate over boosting stages and evaluate only at desired stages
            for stage_idx, prdY in enumerate(clf.staged_predict(tstX), start=1):
                if stage_idx > nr_max_trees:
                    break

                pos = idx_map.get(stage_idx)
                if pos is None:
                    continue  # stage not in our grid

                eval_score: float = metric_fn(tstY, prdY)
                scores[pos] = eval_score

                if eval_score - best_performance > DELTA_IMPROVE:
                    best_performance = eval_score
                    best_params["params"] = (d, lr, stage_idx)

            values[lr] = scores

        plot_multiline_chart(
            n_estimators_grid,
            values,
            ax=ax,
            title=f"Gradient Boosting with max_depth={d}",
            xlabel="nr estimators",
            ylabel=metric,
            percentage=True,
        )

    # Retrain best model with the selected hyperparameters
    if best_params["params"]:
        d_best, lr_best, n_best = best_params["params"]
        best_model = GradientBoostingClassifier(
            n_estimators=n_best,
            max_depth=d_best,
            learning_rate=lr_best,
            # random_state=0,
        )
        best_model.fit(trnX, trnY)

        print(
            f"GB best for {n_best} trees (d={d_best} and lr={lr_best}) "
            f"with {metric}={best_performance:.4f}"
        )
    else:
        print("No GB configuration improved the baseline.")

    return best_model, best_params

# ---------- Train-test split ----------

trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y, random_state=42)

# ---------- Models' Comparision ----------

eval_metric = metrics[0]
figure()
best_model, params = gradient_boosting_study(
    trnX,
    trnY,
    tstX,
    tstY,
    nr_max_trees=1000,
    lag=250,
    metric=eval_metric,
)
savefig(f"images/{file_tag}_gb_{eval_metric}_study.png")

# ---------- Best model performance ----------
prd_trn: array = best_model.predict(trnX)
prd_tst: array = best_model.predict(tstX)
figure()
plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels)
savefig(f'images/{file_tag}_{params["name"]}_best_{params["metric"]}_eval.png')

# ------------- Variables Importance ---------

from numpy import std, argsort

trees_importances: list[float] = []
for lst_trees in best_model.estimators_:
    for tree in lst_trees:
        trees_importances.append(tree.feature_importances_)

stdevs: list[float] = list(std(trees_importances, axis=0))
importances = best_model.feature_importances_
indices: list[int] = argsort(importances)[::-1]
elems: list[str] = []
imp_values: list[float] = []
for f in range(len(vars)):
    elems += [vars[indices[f]]]
    imp_values.append(importances[indices[f]])
    print(f"{f+1}. {elems[f]} ({importances[indices[f]]})")

figure()
plot_horizontal_bar_chart(
    elems,
    imp_values,
    error=stdevs,
    title="GB variables importance",
    xlabel="importance",
    ylabel="variables",
    percentage=True,
)
savefig(f"images/{file_tag}_gb_{eval_metric}_vars_ranking.png")

# ---------- Overfitting study ----------
d_max: int = params["params"][0]
lr: float = params["params"][1]
nr_estimators: list[int] = [i for i in range(2, 2501, 500)]

y_tst_values: list[float] = []
y_trn_values: list[float] = []
acc_metric: str = "accuracy"

for n in nr_estimators:
    clf = GradientBoostingClassifier(n_estimators=n, max_depth=d_max, learning_rate=lr)
    clf.fit(trnX, trnY)
    prd_tst_Y: array = clf.predict(tstX)
    prd_trn_Y: array = clf.predict(trnX)
    y_tst_values.append(CLASS_EVAL_METRICS[acc_metric](tstY, prd_tst_Y))
    y_trn_values.append(CLASS_EVAL_METRICS[acc_metric](trnY, prd_trn_Y))

figure()
plot_multiline_chart(
    nr_estimators,
    {"Train": y_trn_values, "Test": y_tst_values},
    title=f"GB overfitting study for d={d_max} and lr={lr}",
    xlabel="nr_estimators",
    ylabel=str(eval_metric),
    percentage=True,
)
savefig(f"images/{file_tag}_gb_{eval_metric}_overfitting.png")