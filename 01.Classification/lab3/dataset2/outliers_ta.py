from pandas import DataFrame, Series
from dslabs_functions import determine_outlier_thresholds_for_var, get_variable_types

def process_outliers(df: DataFrame, strategy: str = "discard") -> tuple[DataFrame, bool]:
    print(f"Shape of original data: {df.shape}")
    file_tag = "traffic_accidents"
    variable_types = get_variable_types(df, ignore_date=True)
    numeric_vars: list[str] = variable_types["numeric"] + variable_types["binary"]
    if numeric_vars:
        if strategy == "discard":
            df = discard_outliers(df, file_tag, numeric_vars)
        elif strategy == "truncate":
            df = truncate_outliers(df, file_tag, numeric_vars)
        else:
            df = replace_fixed_outliers(df, file_tag, numeric_vars)
        print(f"Shape after performing outlier processing with {strategy} strategy: {df.shape}")
    else:
        print("There are no numeric or binary variables to process.")
    return df, True

def discard_outliers(df: DataFrame, file_tag: str, numeric_vars: list[str]) -> DataFrame:
    summary5: DataFrame = df[numeric_vars].describe()
    for var in numeric_vars:
        top, bottom = determine_outlier_thresholds_for_var(summary5[var])
        outliers: Series = df[(df[var] > top) | (df[var] < bottom)]
        df = df.drop(outliers.index, axis=0, inplace=False)

    df.to_csv(f"lab3/data/{file_tag}_train_drop_outliers.csv", index=True)
    return df

def truncate_outliers(df: DataFrame, file_tag: str, numeric_vars: list[str]) -> DataFrame:
    summary5: DataFrame = df[numeric_vars].describe()
    for var in numeric_vars:
        top, bottom = determine_outlier_thresholds_for_var(summary5[var])
        df[var] = df[var].apply(lambda x: top if x > top else bottom if x < bottom else x)

    df.to_csv(f"lab3/data/{file_tag}_train_truncate_outliers.csv", index=True)
    return df

def replace_fixed_outliers(df: DataFrame, file_tag: str, numeric_vars: list[str]) -> DataFrame:
    summary5: DataFrame = df[numeric_vars].describe()
    for var in numeric_vars:
        top, bottom = determine_outlier_thresholds_for_var(summary5[var])
        median: float = df[var].median()
        df[var] = df[var].apply(lambda x: median if x > top or x < bottom else x)

    df.to_csv(f"lab3/data/{file_tag}_train_fixed_outliers.csv", index=True)
    return df