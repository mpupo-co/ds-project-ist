from math import ceil
from pandas import DataFrame, Index
from matplotlib.pyplot import savefig, figure
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dslabs_functions import evaluate_approach, plot_multiline_chart

def select_low_variance_variables(data: DataFrame, max_threshold: float, target: str = "class") -> list:
    summary5: DataFrame = data.describe()
    vars2drop: Index[str] = summary5.columns[
        summary5.loc["std"] * summary5.loc["std"] < max_threshold
    ]
    vars2drop = vars2drop.drop(target) if target in vars2drop else vars2drop
    return list(vars2drop.values)

def select_redundant_variables(data: DataFrame, min_threshold: float = 0.90, target: str = "class") -> list:
    # Drop the target if present, compute absolute correlation matrix
    df: DataFrame = data.drop(columns=[target], errors="ignore", inplace=False)
    corr: DataFrame = df.corr().abs()

    # Use upper triangle only to avoid duplicate pairs and fragile drop logic
    from numpy import tril, ones

    mask = tril(ones(corr.shape), k=0).astype(bool)
    upper = corr.where(~mask)

    # any column that has a correlation >= min_threshold with some earlier column
    vars2drop: list = [col for col in upper.columns if (upper[col] >= min_threshold).any()]
    return vars2drop

def study_redundancy_for_feature_selection(
    train: DataFrame,
    test: DataFrame,
    target: str = "class",
    min_threshold: float = 0.90,
    lag: float = 0.05,
    metric: str = "accuracy",
    file_tag: str = "",
) -> dict:
    options: list[float] = [round(min_threshold + i * lag, 3) for i in range(ceil((1 - min_threshold) / lag) + 1)]

    results: dict[str, list] = {"NB": [], "KNN": []}
    for thresh in options:
        vars2drop: list = select_redundant_variables(train, min_threshold=thresh, target=target)

        train_copy: DataFrame = train.drop(vars2drop, axis=1, inplace=False)
        test_copy: DataFrame = test.drop(vars2drop, axis=1, inplace=False)
        eval: dict | None = evaluate_approach(train_copy, test_copy, target=target, metric=metric)
        if eval is not None:
            results["NB"].append(eval[metric][0])
            results["KNN"].append(eval[metric][1])
    figure()
    plot_multiline_chart(
        options,
        results,
        title=f"{file_tag} redundancy study ({metric})",
        xlabel="correlation threshold",
        ylabel=metric,
        percentage=True,
    )
    savefig(f"images/{file_tag}_fs_redundancy_{metric}_study.png")
    return results

def study_variance_for_feature_selection(
    train: DataFrame,
    test: DataFrame,
    target: str = "class",
    max_threshold: float = 1,
    lag: float = 0.05,
    metric: str = "accuracy",
    file_tag: str = "",
) -> dict:
    options: list[float] = [
        round(i * lag, 3) for i in range(1, ceil(max_threshold / lag + lag))
    ]
    results: dict[str, list] = {"NB": [], "KNN": []}
    for thresh in options:
        vars2drop = select_low_variance_variables(train, max_threshold=thresh, target=target)

        train_copy: DataFrame = train.drop(vars2drop, axis=1, inplace=False)
        test_copy: DataFrame = test.drop(vars2drop, axis=1, inplace=False)
        eval: dict[str, list] | None = evaluate_approach(
            train_copy, test_copy, target=target, metric=metric
        )
        if eval is not None:
            results["NB"].append(eval[metric][0])
            results["KNN"].append(eval[metric][1])
    figure()
    plot_multiline_chart(
        options,
        results,
        title=f"{file_tag} variance study ({metric})",
        xlabel="variance threshold",
        ylabel=metric,
        percentage=True,
    )
    savefig(f"images/{file_tag}_fs_low_var_{metric}_study.png")
    return results

def apply_feature_selection(
    train: DataFrame,
    test: DataFrame,
    vars2drop: list,
    filename: str = "",
    tag: str = "",
) -> tuple[DataFrame, DataFrame]:
    
    train_copy: DataFrame = train.drop(vars2drop, axis=1, inplace=False)
    train_copy.to_csv(f"{filename}_train_{tag}.csv", index=True)
    test_copy: DataFrame = test.drop(vars2drop, axis=1, inplace=False)
    test_copy.to_csv(f"{filename}_test_{tag}.csv", index=True)
    return train_copy, test_copy

def feature_selection(train: DataFrame, test: DataFrame, strategy: str, target: str, file_tag: str, 
                      max_threshold: float = 0.45, min_threshold: float = 0.4) -> tuple[DataFrame, DataFrame]:
    print(f"=== Applying feature selection strategy: {strategy} ===")
    if strategy == "lowvar":
        vars2drop: list[str] = select_low_variance_variables(train, max_threshold=max_threshold, target=target)
        #study_variance_for_feature_selection(train, test, target=target, max_threshold=max_threshold+max_threshold*0.5, file_tag=file_tag)
        print(f"Dropping {len(vars2drop)} low variance variables based on threshold {max_threshold}: {vars2drop}")
    elif strategy == "redundant":
        vars2drop: list[str] = select_redundant_variables(train, min_threshold=min_threshold, target=target)
        print(f"Dropping {len(vars2drop)} redundant variables based on threshold {min_threshold}: {vars2drop}")
        #study_redundancy_for_feature_selection(train, test, target=target, min_threshold=min_threshold-min_threshold*0.5, file_tag=file_tag)
    else:
        raise ValueError(f"Unknown feature selection strategy: {strategy}")
    train, test = apply_feature_selection(train, test, vars2drop, filename=f"data/{file_tag}", tag=strategy)
    train.to_csv(f"data/{file_tag}_train_{strategy}.csv", index=True)
    test.to_csv(f"data/{file_tag}_test_{strategy}.csv", index=True)
    
    return train, test


def feature_generation(df: DataFrame) -> DataFrame:
    # map the grade part: extract trailing phrase e.g. 'LEVEL', 'ON GRADE', 'ON HILLCREST'
    def extract_grade(s):
        if "LEVEL" in s:
            return "LEVEL"
        if "ON GRADE" in s:
            return "ON GRADE"
        if "HILLCREST" in s:
            return "ON HILLCREST"
        return "LEVEL"

    grade_map = {
        "LEVEL": 0,
        "ON GRADE": 1,
        "ON HILLCREST": 2
    }
    df["is_curve"] = df["alignment"].str.contains("CURVE").astype("int8")
    df["alignment_grade"] = df["alignment"].apply(extract_grade).map(grade_map).astype("int8")
    return df