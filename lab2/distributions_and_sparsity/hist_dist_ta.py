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
numeric = variables_types["numeric"]
symbolic = variables_types["symbolic"]

move_cols = ['crash_hour', 'crash_day_of_week', 'crash_month']
# move move_cols from numeric → symbolic
for var in move_cols:
    numeric.remove(var)
    symbolic.append(var)

if [] != numeric:
    rows, cols = define_grid(len(numeric))
    fig, axs = subplots(
        rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False
    )
    i, j = 0, 0
    for n in range(len(numeric)):
        histogram_with_distributions(axs[i, j], data[numeric[n]].dropna(), numeric[n])
        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
    savefig(f"images/{file_tag}_histogram_numeric_distribution.png")
else:
    print("There are no numeric variables.")


