import pandas as pd 
from pandas import DataFrame, read_csv

from matplotlib.pyplot import savefig, figure, tight_layout, subplots
from dslabs_functions import get_variable_types
from numpy import ndarray
from matplotlib.figure import Figure
from matplotlib.pyplot import savefig, subplots, figure
from dslabs_functions import define_grid, HEIGHT, get_variable_types
from dslabs_functions import set_chart_labels, histogram_with_distributions

def truncate_cat(x, max_len=15):
    return x if len(x) <= max_len else x[:12] + "..."

#----------Load dataset----------
filename = 'datasets/traffic_accidents.csv'
file_tag = 'traffic_accidents'

target = 'crash_type'
index = 'crash_date'
data = pd.read_csv(filename, index_col=index)
data = data.sample(frac=0.05, replace=False)

variables_types: dict[str, list] = get_variable_types(data)
numeric: list[str] = variables_types["numeric"]

# choose which numeric vars you want distributions for
vars_with_dists = ["num_units", "injuries_total", "injuries_no_indication", "crash_hour"]

if [] != numeric:
    rows, cols = define_grid(len(numeric))
    fig, axs = subplots(
        rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False
    )
    i, j = 0, 0
    for n in range(len(numeric)):
        if numeric[n] in vars_with_dists:
            histogram_with_distributions(axs[i, j], data[numeric[n]].dropna(), numeric[n])
        else:
            set_chart_labels(
            axs[i, j],
            title=f"Histogram for {numeric[n]}",
            xlabel=numeric[n],
            ylabel="nr records",
        )
            axs[i, j].hist(data[numeric[n]].dropna().values, "auto")
        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
    savefig(f"images/{file_tag}_histogram_numeric_distribution.png")
else:
    print("There are no numeric variables.")


