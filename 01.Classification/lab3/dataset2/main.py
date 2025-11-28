from pandas import DataFrame, Series, read_csv, concat, to_datetime, to_numeric
from encoding_ta import data_encoding
from outliers_ta import process_outliers
from balancing_ta import data_balancing
from scaling_ta import data_scaling
from numpy import ndarray
from matplotlib.pyplot import savefig, show, figure
from dslabs_functions import plot_multibar_chart, CLASS_EVAL_METRICS, run_NB, run_KNN

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
    train_df.to_csv("lab3/data/traffic_accidents_train.csv", index=False)
    test_df.to_csv("lab3/data/traffic_accidents_test.csv", index=False)

    return train_df, test_df

def get_variable_types(df: DataFrame, ignore_date: bool = False) -> dict[str, list]:
    variable_types: dict = {"numeric": [], "binary": [], "date": [], "symbolic": []}
    nr_values: Series = df.nunique(axis=0, dropna=True)
    for c in df.columns:
        if 2 == nr_values[c]:
            variable_types["binary"].append(c)
            df[c].astype("bool")
        else:
            try:
                to_numeric(df[c], errors="raise")
                if "crash" in c and not ignore_date:
                    print(f"Column {c} seems numeric but is actually symbolic.")
                    variable_types["symbolic"].append(c)
                else:
                    variable_types["numeric"].append(c)
            except ValueError:
                try:
                    df[c] = to_datetime(df[c], errors="raise")
                    variable_types["date"].append(c)
                except ValueError:
                    variable_types["symbolic"].append(c)
    return variable_types

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

def evaluate(train_df: DataFrame, test_df: DataFrame, file_tag: str, target: str, outlier_strategy: str, 
             scaling_strategy: str, balancing_strategy: str, flags: list[bool]):
    
    print("Evaluating approach for the following configuration:")
    print(f"Outlier strategy: {outlier_strategy} (applied: {flags[0]})")
    print(f"Scaling strategy: {scaling_strategy} (applied: {flags[1]})")
    print(f"Balancing strategy: {balancing_strategy} (applied: {flags[2]})")

    train_cp = train_df.copy(deep=True)
    test_cp = test_df.copy(deep=True)
    label = ""
    title_label = ""
    
    if flags[0]:
        label += (f"_{outlier_strategy}_outlier")
    if flags[1]:
        label += (f"_{scaling_strategy}_scaling")
    if flags[2]:
        label += (f"_{balancing_strategy}_balancing")

    if flags[0]:
        if outlier_strategy == "fixed":
            title_label = "(replacing outliers)"
        if outlier_strategy == "truncate":
            title_label = "(truncating outliers)"
    if flags[1]:
        if scaling_strategy == "zscore":
            title_label = "(scaling z-score)"
        if scaling_strategy == "minmax":
            title_label = "(scaling min-max)"        
    if flags[2]:
        if balancing_strategy == "under":
            title_label = "(underbalancing)"
        if balancing_strategy == "over":
            title_label = "(overbalancing)"    
        if balancing_strategy == "smote":
            title_label = "(smote)" 
    figure()
    eval: dict[str, list] = evaluate_approach(train_cp, test_cp, target=target, metric="recall")
    plot_multibar_chart(["NB", "KNN"], eval, title=f"{file_tag} evaluation {title_label}", percentage=True)
    savefig(f"images/{file_tag}_eval{label}.png")
    show()

def main():
    sample = True
    flag_outliers = False
    flag_scaling = False
    flag_balancing = False
    filename = "data/traffic_accidents.csv"
    target = "crash_type"
    file_tag = "traffic_accidents"
    outlier_strategy="fixed"
    scaling_strategy = "zscore"
    balancing_strategy = "smote"
    flags = [flag_outliers, flag_scaling, flag_balancing]

    df = load_dataset(filename, sample=sample)
    df.drop(df.filter(regex='injur').columns, axis=1, inplace=True)
    df = data_encoding(df)
    train_df, test_df = split_train_test_data(df)
    train_df, flag_outliers = process_outliers(train_df, strategy=outlier_strategy)
    train_df, test_df, flag_scaling = data_scaling([train_df, test_df], file_tag, target, strategy=scaling_strategy)
    balanced_df, flag_balancing = data_balancing(concat([train_df, test_df]), target, strategy=balancing_strategy)
    train_df, test_df = split_train_test_data(balanced_df)
    flags = [flag_outliers, flag_scaling, flag_balancing]
    evaluate(train_df, test_df, file_tag, target, outlier_strategy, scaling_strategy, balancing_strategy, flags)


if __name__ == "__main__":
    main()