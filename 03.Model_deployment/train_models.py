import json
import time
import joblib
import pandas as pd
from typing import Literal, Any, Tuple, Dict
from pandas import DataFrame
from numpy import ndarray
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import _BaseNB, GaussianNB, MultinomialNB, BernoulliNB
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from utils.dslabs_functions import CLASS_EVAL_METRICS, DELTA_IMPROVE

def split_train_test_data(df: DataFrame, test_size: float = 0.3, random_state: int = 42) -> tuple[DataFrame, DataFrame]:
    train_df = df.sample(frac=1-test_size, random_state=random_state)
    test_df = df.drop(train_df.index)
    print(f'Train shape: {train_df.shape}, Test shape: {test_df.shape}')

    return train_df, test_df

def naive_bayes_study(
    trnX: ndarray, trnY: ndarray, tstX: ndarray, tstY: ndarray, metric: str = "recall"
) -> tuple[_BaseNB | None, Dict[str, Any], float]:
    estimators: dict = {
        "GaussianNB": GaussianNB(),
        #"MultinomialNB": MultinomialNB(),
        "BernoulliNB": BernoulliNB(),
    }
    best_model: _BaseNB | None = None
    best_params: Dict[str, Any] = {"name": "", "metric": metric, "hyperparams": {}}
    best_performance = 0.0

    for clf_name, clf in estimators.items():
        clf.fit(trnX, trnY)

        prdY: ndarray = clf.predict(tstX)
        eval_v: float = CLASS_EVAL_METRICS[metric](tstY, prdY)

        if eval_v - best_performance > DELTA_IMPROVE:
            best_performance = eval_v
            best_params["name"] = clf_name
            best_params["metric"] = metric
            best_params["hyperparams"] = {"name": clf_name, "metric": metric}
            best_model = clf
    return best_model, best_params, best_performance

def logistic_regression_study(
    trnX: ndarray,
    trnY: ndarray,
    tstX: ndarray,
    tstY: ndarray,
    nr_max_iterations: int = 2500,
    lag: int = 500,
    metric: str = "recall",
) -> tuple[LogisticRegression | None, Dict[str, Any], float]:

    nr_iterations: list[int] = [lag] + [i for i in range(2 * lag, nr_max_iterations + 1, lag)]
    penalty_types: list[str] = ["l1", "l2"]  # only available if solver='liblinear'

    best_model: LogisticRegression | None = None
    best_params: Dict[str, Any] = {"name": "LR", "metric": metric, "hyperparams": {}}
    best_performance: float = 0.0

    for pen in penalty_types:
        for it in nr_iterations:
            clf = LogisticRegression(
                penalty=pen,
                max_iter=it,
                solver="liblinear",
                verbose=False,
                random_state=42,
            )
            clf.fit(trnX, trnY)
            prdY: ndarray = clf.predict(tstX)
            eval_v: float = CLASS_EVAL_METRICS[metric](tstY, prdY)

            if eval_v - best_performance > DELTA_IMPROVE:
                best_performance = eval_v
                best_params["hyperparams"] = {"penalty": pen, "max_iter": it, "solver": "liblinear"}
                best_model = clf
    return best_model, best_params, best_performance


def random_forests_study(
    trnX,
    trnY,
    tstX,
    tstY,
    nr_max_trees: int = 2500,
    lag: int = 500,
    metric: str = "recall",
) -> tuple[RandomForestClassifier | None, Dict[str, Any], float]:
    n_estimators_list = [100] + [i for i in range(500, nr_max_trees + 1, lag)]
    max_depths = [2, 5, 7]
    max_features_list = [0.1, 0.3, 0.5, 0.7, 0.9]

    best_model: RandomForestClassifier | None = None
    best_params: Dict[str, Any] = {"name": "RF", "metric": metric, "hyperparams": {}}
    best_performance: float = 0.0

    for d in max_depths:
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
            prdY: ndarray = clf.predict(tstX)
            score: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
            perf_line.append(score)

            if score > best_performance + DELTA_IMPROVE:
                best_performance = score
                best_model = clf
                best_params["hyperparams"] = {"max_depth": d, "max_features": f, "n_estimators": n_estimators_list[0]}

            for n in n_estimators_list[1:]:
                clf.n_estimators = n
                clf.fit(trnX, trnY)

                prdY = clf.predict(tstX)
                score = CLASS_EVAL_METRICS[metric](tstY, prdY)
                perf_line.append(score)

                if score > best_performance + DELTA_IMPROVE:
                    best_performance = score
                    best_model = clf
                    best_params["hyperparams"] = {"max_depth": d, "max_features": f, "n_estimators": n}
    return best_model, best_params, best_performance

def mlp_study(
    trnX: ndarray,
    trnY: ndarray,
    tstX: ndarray,
    tstY: ndarray,
    nr_max_iterations: int = 2500,
    lag: int = 500,
    metric: str = "recall",
) -> tuple[MLPClassifier | None, Dict[str, Any], float]:

    nr_iterations: list[int] = [lag] + [i for i in range(2 * lag, nr_max_iterations + 1, lag)]

    lr_types: list[Literal["constant", "invscaling", "adaptive"]] = ["constant", "invscaling", "adaptive"]
    learning_rates: list[float] = [0.5, 0.05, 0.005, 0.0005]

    best_model: MLPClassifier | None = None
    best_params: Dict[str, Any] = {"name": "MLP", "metric": metric, "hyperparams": ()}
    best_performance: float = 0.0

    for lr_type in lr_types:
        for lr in learning_rates:
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

                prdY: ndarray = clf.predict(tstX)
                eval_v: float = CLASS_EVAL_METRICS[metric](tstY, prdY)

                if eval_v - best_performance > DELTA_IMPROVE:
                    best_performance = eval_v
                    best_params["hyperparams"] = {"learning_rate_type": lr_type, "learning_rate": lr, "max_iter": n, "activation": "logistic", "solver": "sgd"}
                    best_model = clf
    return best_model, best_params, best_performance

def decision_trees_study(
    trnX: ndarray,
    trnY: ndarray,
    tstX: ndarray,
    tstY: ndarray,
    d_max: int = 10,
    lag: int = 2,
    metric: str = "recall",
) -> Tuple[DecisionTreeClassifier | None, Dict[str, Any], float]:
    criteria: list[Literal["entropy", "gini"]] = ["entropy", "gini"]
    depths: list[int] = [i for i in range(2, d_max + 1, lag)]

    best_model: DecisionTreeClassifier | None = None
    best_params: Dict[str, Any] = {"name": "DT", "metric": metric, "hyperparams": {}}
    best_performance: float = 0.0

    for c in criteria:
        for d in depths:
            clf = DecisionTreeClassifier(max_depth=d, criterion=c, min_impurity_decrease=0, random_state=42)
            clf.fit(trnX, trnY)
            prdY: ndarray = clf.predict(tstX)
            eval_v: float = CLASS_EVAL_METRICS[metric](tstY, prdY)

            if eval_v - best_performance > DELTA_IMPROVE:
                best_performance = eval_v
                best_params["hyperparams"] = {"criterion": c, "max_depth": d, "min_impurity_decrease": 0}
                best_model = clf
    return best_model, best_params, best_performance

def gradient_boosting_study(
    trnX: ndarray,
    trnY: ndarray,
    tstX: ndarray,
    tstY: ndarray,
    nr_max_trees: int = 2500,
    lag: int = 500,
    metric: str = "recall",
) -> Tuple[GradientBoostingClassifier | None, Dict[str, Any], float]:

    n_estimators_grid: list[int] = [100] + [i for i in range(500, nr_max_trees + 1, lag)]
    max_depths: list[int] = [2, 5, 7]
    learning_rates: list[float] = [0.1, 0.3, 0.5, 0.7, 0.9]

    metric_fn = CLASS_EVAL_METRICS[metric]

    best_model: GradientBoostingClassifier | None = None
    best_params: Dict[str, Any] = {"name": "GB", "metric": metric, "hyperparams": {}}
    best_performance: float = 0.0
    # map: stage -> position in grid
    idx_map = {n: i for i, n in enumerate(n_estimators_grid)}

    for _, d in enumerate(max_depths):

        for lr in learning_rates:
            clf = GradientBoostingClassifier(
                n_estimators=nr_max_trees,
                max_depth=d,
                learning_rate=lr,
                random_state=42,
            )
            clf.fit(trnX, trnY)
            for stage_idx, prdY in enumerate(clf.staged_predict(tstX), start=1):
                if stage_idx > nr_max_trees:
                    break
                pos = idx_map.get(stage_idx)
                if pos is None:
                    continue
                eval_score: float = metric_fn(tstY, prdY)

                if eval_score - best_performance > DELTA_IMPROVE:
                    best_performance = eval_score
                    best_params["hyperparams"]= {"max_depth": d, "learning_rate": lr, "n_estimators": stage_idx}
                    best_model = clf
    return best_model, best_params, best_performance    

def knn_study(
    trnX: ndarray,
    trnY: ndarray,
    tstX: ndarray,
    tstY: ndarray,
    k_max: int = 19,
    lag: int = 2,
    metric: str = "recall",
) -> tuple[KNeighborsClassifier | None, Dict[str, Any], float]:

    dist: list[Literal["manhattan", "euclidean", "chebyshev"]] = ["manhattan", "euclidean", "chebyshev"]
    kvalues: list[int] = [i for i in range(1, k_max + 1, lag)]

    best_model: KNeighborsClassifier | None = None
    best_params: Dict[str, Any] = {"name": "KNN", "metric": metric, "hyperparams": {}}
    best_performance: float = 0.0

    for d in dist:
        for k in kvalues:
            clf = KNeighborsClassifier(n_neighbors=k, metric=d)
            clf.fit(trnX, trnY)

            prdY: ndarray = clf.predict(tstX)
            eval_v: float = CLASS_EVAL_METRICS[metric](tstY, prdY)

            if eval_v - best_performance > DELTA_IMPROVE:
                best_performance = eval_v
                best_params["hyperparams"] = {"n_neighbors": k, "metric": d}
                best_model = clf
    return best_model, best_params, best_performance


def train_models(train: DataFrame, test: DataFrame, model_names: list[str], target: str, eval_metric: str) -> None:
    def _log_stage(name, start_time):
        duration = time.perf_counter() - start_time
        print(f"[pipeline] {name} took {duration:.2f}s")

    print("== Training models ==")
    feature_cols = [c for c in train.columns if c != target]
    trnX: ndarray = train[feature_cols].values
    trnY: ndarray = train[target].values
    tstX: ndarray = test[feature_cols].values
    tstY: ndarray = test[target].values

    if "NaiveBayes" in model_names:
        t = time.perf_counter()
        best_model, hyperparams, performance = naive_bayes_study(trnX, trnY, tstX, tstY, 
                                        metric=eval_metric)
        _log_stage("Finished training Naive Bayes model", t)
        joblib.dump(best_model, f"models/model_nb_cf.joblib")
        json.dump(hyperparams["hyperparams"], open(f"models/hyperparams_nb_cf.json", "w"))
        print(f"Naive Bayes model saved with {hyperparams} and performance {performance:.4f} for metric {eval_metric}")

    if "KNN" in model_names:
        t = time.perf_counter()
        best_model, hyperparams, performance = knn_study(trnX, trnY, tstX, tstY, 
                                        k_max=15, metric=eval_metric)
        _log_stage("Finished training KNN model", t)
        joblib.dump(best_model, f"models/model_knn_cf.joblib")
        json.dump(hyperparams["hyperparams"], open(f"models/hyperparams_knn_cf.json", "w"))
        print(f"KNN model saved with {hyperparams} and performance {performance:.4f} for metric {eval_metric}")
    
    if "DecisionTree" in model_names:
        t = time.perf_counter()
        best_model, hyperparams, performance = decision_trees_study(trnX, trnY, tstX, tstY, 
                                                  d_max=20, metric=eval_metric)
        _log_stage("Finished training Decision Tree model", t)
        joblib.dump(best_model, f"models/model_dt_cf.joblib")
        json.dump(hyperparams["hyperparams"], open(f"models/hyperparams_dt_cf.json", "w"))
        print(f"Decision Tree model saved with {hyperparams} and performance {performance:.4f} for metric {eval_metric}")

    if "RandomForest" in model_names:
        t = time.perf_counter()
        best_model, hyperparams, performance = random_forests_study(trnX, trnY, tstX, tstY, 
                                                  nr_max_trees=1000, lag=250, metric=eval_metric)
        _log_stage("Finished training Random Forest model", t)
        joblib.dump(best_model, f"models/model_rf_cf.joblib")
        json.dump(hyperparams["hyperparams"], open(f"models/hyperparams_rf_cf.json", "w"))
        print(f"Random Forest model saved with {hyperparams} and performance {performance:.4f} for metric {eval_metric}")

    if "LogisticRegression" in model_names:
        t = time.perf_counter()
        best_model, hyperparams, performance = logistic_regression_study(trnX, trnY, tstX, tstY,
                                    nr_max_iterations=2000, lag=250, metric=eval_metric)
        _log_stage("Finished training LR model", t)
        joblib.dump(best_model, f"models/model_lr_cf.joblib")
        json.dump(hyperparams["hyperparams"], open(f"models/hyperparams_lr_cf.json", "w"))
        print(f"LR model saved with {hyperparams} and performance {performance:.4f} for metric {eval_metric}")

    if "GradientBoosting" in model_names:
        t = time.perf_counter()
        best_model, hyperparams, performance = gradient_boosting_study(trnX, trnY, tstX, tstY,
                                    nr_max_trees=1000, lag=250, metric=eval_metric)
        _log_stage("Finished training GB model", t)
        joblib.dump(best_model, f"models/model_gb_cf.joblib")
        json.dump(hyperparams["hyperparams"], open(f"models/hyperparams_gb_cf.json", "w"))
        print(f"GB model saved with {hyperparams} and performance {performance:.4f} for metric {eval_metric}")

    if "MLP" in model_names:
        t = time.perf_counter()
        best_model, hyperparams, performance = mlp_study(trnX, trnY, tstX, tstY,
                                    nr_max_iterations=2000, lag=500, metric=eval_metric)
        _log_stage("Finished training MLP model", t)
        joblib.dump(best_model, f"models/model_mlp_cf.joblib")
        json.dump(hyperparams["hyperparams"], open(f"models/hyperparams_mlp_cf.json", "w"))
        print(f"MLP model saved with {hyperparams} and performance {performance:.4f} for metric {eval_metric}")

if __name__ == "__main__":
    df_file = "data/processed_cf.csv"
    model_names = ["KNN", "DecisionTree", "RandomForest", "LogisticRegression", "GradientBoosting", "NaiveBayes", "MLP"]
    target = "Cancelled"
    df = pd.read_csv(df_file)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    train_df, test_df = split_train_test_data(df, test_size=0.25, random_state=42)
    train_models(train_df, test_df, model_names=model_names, target=target, eval_metric="recall")