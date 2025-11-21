from numpy import ndarray
from pandas import read_csv, DataFrame
from matplotlib.figure import Figure
from matplotlib.pyplot import figure, subplots, savefig, show
from dslabs_functions import HEIGHT, plot_multi_scatters_chart

filename = 'datasets/Combined_Flights_2022.csv'
file_tag = 'Combined_Flights_2022'

target = 'Cancelled'
index = 'FlightDate'
data: DataFrame = read_csv(filename, index_col=index, na_values="")
data = data.sample(frac=0.01, replace=False)

def truncate_df_strings(df, max_len=12):
    df = df.copy()
    for col in df.select_dtypes(include=["object", "string"]):
        df[col] = df[col].astype(str).apply(
            lambda x: x if len(x) <= max_len else x[:max_len-3] + "..."
        )
    return df
data = truncate_df_strings(data, max_len=12)

cols_to_drop = data.columns[
    data.columns.str.contains('ID', case=False) |
    data.columns.str.contains('Flight_Num', case=False) |
    data.columns.str.contains('Code', case=False) |
    data.columns.str.contains(target, case=False) |
    data.columns.to_series().apply(
        lambda col: any(f'Div{i}' in col for i in range(1, 6))
    ) |
    data.columns.str.contains('Tail', case=False)
]

#print(cols_to_drop)
data = data.drop(columns=cols_to_drop)
data = data.dropna()


from sparsity_plot import plot_sparsity_matrix

plot_sparsity_matrix(data, file_tag=file_tag)