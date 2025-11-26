from pandas import read_csv, DataFrame
from dslabs_functions import get_variable_types

filename = 'datasets/traffic_accidents.csv'
file_tag = 'traffic_accidents'
target = 'crash_type'
index = 'crash_date'
data: DataFrame = read_csv(filename, index_col=index, na_values="")
data = data.sample(frac=0.05, replace=False)

cols_to_drop = data.columns[
    data.columns.str.contains(target, case=False)]

data = data.drop(columns=cols_to_drop)

def truncate_df_strings(df, max_len=12):
    df = df.copy()
    for col in df.select_dtypes(include=["object", "string"]):
        df[col] = df[col].astype(str).apply(
            lambda x: x if len(x) <= max_len else x[:max_len-3] + "..."
        )
    return df
data = truncate_df_strings(data, max_len=12)
data = data.dropna()

variables_types: dict[str, list] = get_variable_types(data)
numeric = variables_types["numeric"]
symbolic = variables_types["symbolic"]
binary = variables_types["binary"]

move_cols = ['crash_hour', 'crash_day_of_week', 'crash_month']
# move move_cols from numeric → symbolic
for var in move_cols:
    numeric.remove(var)
    symbolic.append(var)

from sparsity_plot import plot_sparsity_matrix

plot_sparsity_matrix(data, file_tag, numeric, symbolic, binary)
