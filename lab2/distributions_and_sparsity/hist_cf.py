import pandas as pd 
from pandas import DataFrame, read_csv

#----------Load dataset----------
filename = 'datasets/Combined_Flights_2022.csv'
file_tag = 'Combined_Flights_2022'

target = 'Cancelled'
index = 'FlightDate'
data = pd.read_csv(filename, index_col=index)

cols_to_drop = data.columns[
    data.columns.str.contains('ID', case=False) |
    data.columns.str.contains('Flight_Num', case=False) |
    data.columns.str.contains('Code', case=False) |
    data.columns.to_series().apply(
        lambda col: any(f'Div{i}' in col for i in range(1, 6))
    ) |
    data.columns.str.contains('Tail', case=False)
]

data = data.drop(columns=cols_to_drop)

def truncate_cat(x, max_len=15):
    return x if len(x) <= max_len else x[:12] + "..."

from matplotlib.pyplot import savefig, figure, tight_layout, subplots
from dslabs_functions import get_variable_types
from numpy import ndarray
from matplotlib.figure import Figure
from matplotlib.pyplot import savefig,  subplots, figure
from dslabs_functions import define_grid, HEIGHT, get_variable_types

variables_types: dict[str, list] = get_variable_types(data)
numeric: list[str] = variables_types["numeric"]

#---------Numeric Variables Histograms----------

from dslabs_functions import set_chart_labels
if [] != numeric:
    rows = 6
    cols = 5
    fig, axs = subplots(
        rows, cols, figsize=(cols * 5, rows * 5), squeeze=False
    )
    i: int
    j: int
    i, j = 0, 0
    for n in range(len(numeric)):
        set_chart_labels(
            axs[i, j],
            title=f"Histogram for {numeric[n]}",
            xlabel=numeric[n],
            ylabel="nr records",
        )
        axs[i, j].hist(data[numeric[n]].dropna().values, "auto")
        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
    tight_layout()
    savefig(f"images/{file_tag}_single_histograms_numeric.png")
else:
    print("There are no numeric variables.")

#---------Symbolic Variables Histograms----------

from dslabs_functions import plot_bar_chart
from pandas import Series

symbolic: list[str] = variables_types["symbolic"] + variables_types["binary"]
if target in symbolic:
    symbolic.remove(target)
if [] != symbolic:
    rows = 6
    cols = 3
    fig, axs = subplots(
        rows, cols, figsize=(cols * 6, rows * 4), squeeze=False
    )
    i, j = 0, 0
    for n in range(len(symbolic)):
        counts: Series = data[symbolic[n]].value_counts()
        plot_bar_chart(
            [truncate_cat(str(c)) for c in counts.index],
            counts.to_list(),
            ax=axs[i, j],
            title="Histogram for %s" % symbolic[n],
            xlabel= symbolic[n],
            ylabel="nr records",
            percentage=False,
        )
        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
    tight_layout()
    savefig(f"images/{file_tag}_histograms_symbolic.png")
else:
    print("There are no symbolic variables.")
