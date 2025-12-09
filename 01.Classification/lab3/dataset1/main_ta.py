from pandas import DataFrame, read_csv, concat
from encoding_ta import data_encoding
from outliers_ta import process_outliers, get_variable_types
from balancing_ta import data_balancing
from scaling_ta import data_scaling
from feature_ta import feature_selection
from numpy import ndarray
import matplotlib
from matplotlib.pyplot import savefig, figure, show
import os, sys
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dslabs_functions import plot_multibar_chart, CLASS_EVAL_METRICS, run_NB, run_KNN

# Use non-interactive backend by default (suitable for headless/server)
matplotlib.use('Agg')

def load_dataset(filename: str, sample: bool = False, random_state: int = 42) -> DataFrame:
    df: DataFrame = read_csv(filename, na_values="")
    print(f'Loading dataset from {filename}...')
    if sample:
        df = df.sample(frac=0.3, random_state=random_state)
        print(f'Sampling 30% of the dataset for quicker processing...')
    return df

def split_train_test_data(df: DataFrame, test_size: float = 0.3, random_state: int = 42) -> tuple[DataFrame, DataFrame]:
    train_df = df.sample(frac=1-test_size, random_state=random_state)
    test_df = df.drop(train_df.index)
    print(f'Train shape: {train_df.shape}, Test shape: {test_df.shape}')
    train_df.to_csv("data/traffic_accidents_train.csv", index=False)
    test_df.to_csv("data/traffic_accidents_test.csv", index=False)

    return train_df, test_df

def get_label(strategies: list[str | None]) -> str:
    label = ""
    for strategy in strategies:
        if strategy:
            label += (f"_{strategy}")
    return label

def derive_date_variables(df: DataFrame, date_vars: list[str]) -> DataFrame:
    for date in date_vars:
        df[f"{date}_year"] = df[date].dt.year
        df[f"{date}_quarter"] = df[date].dt.quarter
        df[f"{date}_month"] = df[date].dt.month
        df[f"{date}_day"] = df[date].dt.day
    df.drop(columns=date_vars, inplace=True)
    return df

def data_cleaning(df: DataFrame, vars: dict[str, list[str]]) -> DataFrame:
    print("=== Performing data cleaning ===")
    #go through all categorical columns and for values with less than 0.5% frequency, replace with 'OTHERS' 
    for col in vars["symbolic"]:
        freq = df[col].value_counts(normalize=True)
        rare_labels = freq[freq < 0.005].index
        df[col] = df[col].replace(rare_labels, 'OTHERS')
    # Droping data leaking variables
    df.drop(df.filter(regex='injur').columns, axis=1, inplace=True)
    df.drop(columns=["damage"], inplace=True)
    df.drop(columns=["crash_month"], inplace=True)
    # Deriving date variables
    df = derive_date_variables(df, ["crash_date"])

    return df

def evaluate_approach(train: DataFrame, test: DataFrame, target: str = "class", metric: str = "accuracy") -> dict[str, list]:
    trnY = train.pop(target).values
    trnX: ndarray = train.values
    tstY = test.pop(target).values
    tstX: ndarray = test.values
    eval: dict[str, list] = {}

    eval_NB: dict[str, float] = run_NB(trnX, trnY, tstX, tstY, metric=metric)
    eval_KNN: dict[str, float] = run_KNN(trnX, trnY, tstX, tstY, metric=metric)
    if eval_NB != {} and eval_KNN != {}:
        for met in CLASS_EVAL_METRICS:
            eval[met] = [eval_NB[met], eval_KNN[met]]

    return eval

def evaluate(train_df: DataFrame, test_df: DataFrame, file_tag: str, target: str, strategies: list[str], label: str):
    strategy_names = ["Outlier handling", "Data scaling", "Data balancing", "Feature selection"]
    print("Evaluating approach for the following configuration:")
    title_label = ""
    for i, strategy in enumerate(strategies):
        if strategy:
            print(f"{strategy_names[i]} - {strategy} strategy")
            if strategy == "fixed":
                title_label = "(replacing outliers)"
            if strategy == "truncate":
                title_label = "(truncating outliers)"
            if strategy == "zscore":
                title_label = "(scaling z-score)"
            if strategy == "minmax":
                title_label = "(scaling min-max)"        
            if strategy == "under":
                title_label = "(underbalancing)"
            if strategy == "over":
                title_label = "(overbalancing)"    
            if strategy == "smote":
                title_label = "(smote)" 
            if strategy == "lowvar":
                title_label = "(low variance)" 
            if strategy == "redundant":
                title_label = "(redundant)" 

    eval: dict[str, list] = evaluate_approach(train_df, test_df, target=target, metric="recall")
    plot_multibar_chart(["NB", "KNN"], eval, title=f"{file_tag} evaluation {title_label}", percentage=True)
    savefig(f"images/{file_tag}_eval{label}.png")

def main():
    parser = argparse.ArgumentParser(description="Run the lab3 pipeline on traffic accidents dataset")
    bool_group = parser.add_argument_group("flags", "Enable/disable pipeline stages (defaults set to current script values)")
    # sample defaults to True; provide --no-sample to disable
    parser.add_argument("--sample", dest="sample", action="store_true", help="Sample 30 percent of dataset (default: enabled)")
    parser.add_argument("--no-sample", dest="sample", action="store_false", help="Do not sample the dataset")
    parser.set_defaults(sample=True)

    bool_group.add_argument("--outliers", dest="flag_outliers", action="store_true", help="Apply outlier processing (default: enabled)")
    bool_group.add_argument("--no-outliers", dest="flag_outliers", action="store_false", help="Do not apply outlier processing")
    parser.set_defaults(flag_outliers=False)

    bool_group.add_argument("--scaling", dest="flag_scaling", action="store_true", help="Apply scaling (default: enabled)")
    bool_group.add_argument("--no-scaling", dest="flag_scaling", action="store_false", help="Do not apply scaling")
    parser.set_defaults(flag_scaling=False)

    bool_group.add_argument("--balancing", dest="flag_balancing", action="store_true", help="Apply balancing (default: enabled)")
    bool_group.add_argument("--no-balancing", dest="flag_balancing", action="store_false", help="Do not apply balancing")
    parser.set_defaults(flag_balancing=False)

    bool_group.add_argument("--feature-selection", dest="flag_feature_sel", action="store_true", help="Apply feature selection (default: enabled)")
    bool_group.add_argument("--no-feature-selection", dest="flag_feature_sel", action="store_false", help="Do not apply feature selection")
    parser.set_defaults(flag_feature_sel=False)

    # file/strategy arguments with defaults matching previous hard-coded values
    parser.add_argument("--filename", default="data/traffic_accidents.csv", help="Path to CSV dataset (default: data/traffic_accidents.csv)")
    parser.add_argument("--target", default="crash_type", help="Target column name (default: crash_type)")
    parser.add_argument("--file-tag", dest="file_tag", default="traffic_accidents", help="File tag used for outputs (default: traffic_accidents)")

    parser.add_argument("--outlier-strategy", dest="outlier_strategy", choices=["discard", "truncate", "fixed"], default="fixed", help="Outlier handling strategy (default: fixed)")
    parser.add_argument("--scaling-strategy", dest="scaling_strategy", choices=["minmax", "zscore"], default="zscore", help="Scaling strategy (default: zscore)")
    parser.add_argument("--balancing-strategy", dest="balancing_strategy", choices=["under", "over", "smote"], default="smote", help="Balancing strategy (default: smote)")
    parser.add_argument("--fs-strategy", dest="feature_sel_strategy", choices=["lowvar", "redundant"], default="redundant", help="Feature selection strategy (default: redundant)")

    parser.add_argument("--max-threshold", dest="max_threshold", type=float, default=0.45, help="Max threshold for low-variance feature selection (default: 0.45)")
    parser.add_argument("--min-threshold", dest="min_threshold", type=float, default=0.4, help="Min threshold for redundancy selection (default: 0.4)")
    parser.add_argument("--random-state", dest="random_state", type=int, default=42, help="Random state for reproducible sampling/splitting (default: 42)")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", help="Print resolved configuration and exit (no processing)")

    args = parser.parse_args()

    # assign local variables from args (keeps previous variable names used below)
    sample = args.sample
    flag_outliers = args.flag_outliers
    flag_scaling = args.flag_scaling
    flag_balancing = args.flag_balancing
    flag_feature_sel = args.flag_feature_sel
    filename = args.filename
    target = args.target
    file_tag = args.file_tag
    outlier_strategy = args.outlier_strategy
    scaling_strategy = args.scaling_strategy
    balancing_strategy = args.balancing_strategy
    feature_sel_strategy = args.feature_sel_strategy
    max_threshold = args.max_threshold
    min_threshold = args.min_threshold
    random_state = args.random_state

    if getattr(args, "dry_run", False):
        cfg = {
            "sample": sample,
            "flag_outliers": flag_outliers,
            "flag_scaling": flag_scaling,
            "flag_balancing": flag_balancing,
            "flag_feature_sel": flag_feature_sel,
            "filename": filename,
            "target": target,
            "file_tag": file_tag,
            "outlier_strategy": outlier_strategy,
            "scaling_strategy": scaling_strategy,
            "balancing_strategy": balancing_strategy,
            "feature_sel_strategy": feature_sel_strategy,
            "max_threshold": max_threshold,
            "min_threshold": min_threshold,
            "random_state": random_state,
        }
        print("Dry run - resolved configuration:")
        for k, v in cfg.items():
            print(f"  {k}: {v}")
        return

    df = load_dataset(filename, sample=sample, random_state=random_state)
    vars = get_variable_types(df)
    df = data_cleaning(df, vars)
    df = data_encoding(df, file_tag)
    train_df, test_df = split_train_test_data(df)
    if flag_outliers:
       train_df, flag_outliers = process_outliers(train_df, file_tag=file_tag, strategy=outlier_strategy)
    if flag_scaling:
       train_df, test_df, flag_scaling = data_scaling([train_df, test_df], file_tag=file_tag, target=target, strategy=scaling_strategy)
    if flag_balancing:
       df, flag_balancing = data_balancing(concat([train_df, test_df]), target, strategy=balancing_strategy)
    
    train_df, test_df = split_train_test_data(df)
    if flag_feature_sel:
       train_df, test_df = feature_selection(train_df, test_df, feature_sel_strategy, target, file_tag, max_threshold=max_threshold, min_threshold=min_threshold)

    strategies = [outlier_strategy if flag_outliers else None,
                  scaling_strategy if flag_scaling else None,
                  balancing_strategy if flag_balancing else None,
                  feature_sel_strategy if flag_feature_sel else None]
    
    label = get_label(strategies)
    train_df.to_csv(f"data/{file_tag}_train{label}.csv", index=False)
    test_df.to_csv(f"data/{file_tag}_test{label}.csv", index=False)

    evaluate(train_df, test_df, file_tag, target, strategies, get_label(strategies))

if __name__ == "__main__":
    main()