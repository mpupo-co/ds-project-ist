import pandas as pd 
from pandas import DataFrame, read_csv

#----------Load dataset----------
filename = 'datasets/Combined_Flights_2022_clean.csv'
file_tag = 'Combined_Flights_2022'

target = 'Cancelled'
index = 'FlightDate'
data = pd.read_csv(filename, index_col=index)


def truncate_cat(x, max_len=15):
    return x if len(x) <= max_len else x[:12] + "..."

from matplotlib.pyplot import savefig, figure, tight_layout, subplots
from dslabs_functions import get_variable_types
from numpy import ndarray
from matplotlib.figure import Figure
from matplotlib.pyplot import savefig,  subplots, figure
from dslabs_functions import define_grid, HEIGHT, get_variable_types

variables_types: dict[str, list] = get_variable_types(data)
numeric = variables_types["numeric"]
symbolic = variables_types["symbolic"]

move_cols = data.columns[data.columns.str.contains('Group', case=False)].tolist() + ['Quarter', 'Month', 'DayofMonth', 'DayOfWeek']
# move move_cols from numeric → symbolic
for var in move_cols:
    numeric.remove(var)
    symbolic.append(var)

#---------Numeric Variables Histograms----------

from dslabs_functions import set_chart_labels
if [] != numeric:
    rows, cols = define_grid(len(numeric))
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

symbolic = symbolic + variables_types["binary"]
if target in symbolic:
    symbolic.remove(target)
if [] != symbolic:
    rows, cols = define_grid(len(symbolic))
    fig, axs = subplots(
        rows, cols, figsize=(cols * 5, rows * 5), squeeze=False
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
