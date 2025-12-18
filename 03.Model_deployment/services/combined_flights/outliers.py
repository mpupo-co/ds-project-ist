from pandas import DataFrame, Series, to_datetime, to_numeric
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from dslabs_functions import determine_outlier_thresholds_for_var

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

def process_outliers(df: DataFrame, strategy: str = "discard") -> tuple[DataFrame, bool]:
    print(f"=== Applying outlier processing strategy: {strategy} ===")
    print(f"Shape of original data: {df.shape}")
    variable_types = get_variable_types(df, ignore_date=True)
    numeric_vars: list[str] = variable_types["numeric"]
    if numeric_vars:
        if strategy == "discard":
            df = discard_outliers(df, numeric_vars)
        elif strategy == "truncate":
            df = truncate_outliers(df, numeric_vars)
        elif strategy == "fixed":
            df = replace_fixed_outliers(df, numeric_vars)
        else:
            raise ValueError(f"Outlier processing strategy {strategy} not recognized.")
        print(f"Shape after performing outlier processing with {strategy} strategy: {df.shape}")
    else:
        print("There are no numeric or binary variables to process.")
    return df, True

def discard_outliers(df: DataFrame, numeric_vars: list[str]) -> DataFrame:
    summary5: DataFrame = df[numeric_vars].describe()
    for var in numeric_vars:
        top, bottom = determine_outlier_thresholds_for_var(summary5[var], std_based=False, threshold=7)
        outliers: Series = df[(df[var] > top) | (df[var] < bottom)]
        df = df.drop(outliers.index, axis=0, inplace=False)
    return df

def truncate_outliers(df: DataFrame, numeric_vars: list[str]) -> DataFrame:
    summary5: DataFrame = df[numeric_vars].describe()
    for var in numeric_vars:
        top, bottom = determine_outlier_thresholds_for_var(summary5[var], std_based=False, threshold=7)
        df[var] = df[var].apply(lambda x: top if x > top else bottom if x < bottom else x)
    return df

def replace_fixed_outliers(df: DataFrame, numeric_vars: list[str]) -> DataFrame:
    summary5: DataFrame = df[numeric_vars].describe()
    for var in numeric_vars:
        top, bottom = determine_outlier_thresholds_for_var(summary5[var], std_based=False, threshold=7)
        median: float = df[var].median()
        df[var] = df[var].apply(lambda x: median if x > top or x < bottom else x)
    return df