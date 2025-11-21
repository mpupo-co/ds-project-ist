import pandas as pd 
from pandas import DataFrame, read_csv

def truncate_cat(x, max_len=15):
    return x if len(x) <= max_len else x[:12] + "..."

#----------Load dataset----------
filename = 'datasets/traffic_accidents.csv'
file_tag = 'traffic_accidents'

target = 'crash_type'
index = 'crash_date'
data = pd.read_csv(filename, index_col=index)


#---------Global Numeric Variables Boxplot----------
from matplotlib.pyplot import savefig, figure, tight_layout, subplots
from dslabs_functions import get_variable_types

variables_types: dict[str, list] = get_variable_types(data)
numeric: list[str] = variables_types["numeric"]
if [] != numeric:
    fig, ax = subplots(figsize=(10, 6))
    data[numeric].boxplot(rot=45, ax=ax)
    tight_layout()
    savefig(f"images/{file_tag}_global_boxplot.png")
else:
    print("There are no numeric variables.")

#---------Numeric Variables Boxplots----------
from numpy import ndarray
from matplotlib.figure import Figure
from matplotlib.pyplot import savefig, subplots, figure
from dslabs_functions import define_grid, HEIGHT, get_variable_types

variables_types: dict[str, list] = get_variable_types(data)
numeric: list[str] = variables_types["numeric"]
if [] != numeric:
    rows: int
    cols: int
    rows, cols = define_grid(len(numeric))
    fig: Figure
    axs: ndarray
    fig, axs = subplots(
        rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False
    )
    i, j = 0, 0
    for n in range(len(numeric)):
        axs[i, j].set_title("Boxplot for %s" % numeric[n])
        axs[i, j].boxplot(data[numeric[n]].dropna().values, whis=1.5)
        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
    savefig(f"images/{file_tag}_single_boxplots.png")
else:
    print("There are no numeric variables.")

#---------Outliers Analysis----------
from dslabs_functions import count_outliers, plot_multibar_chart
if [] != numeric:
    outliers = count_outliers(data, numeric, nrstdev=2, iqrfactor=1.5)
    iqr_out = outliers["iqr"]
    stdev_out = outliers["stdev"]

    filtered_numeric = []
    filtered_iqr = []
    filtered_stdev = []

    for i in range(len(numeric)):
        if iqr_out[i] > 0 or stdev_out[i] > 0:
            filtered_numeric.append(numeric[i])
            filtered_iqr.append(iqr_out[i])
            filtered_stdev.append(stdev_out[i])

    filtered_outliers = {"iqr": filtered_iqr, "stdev": filtered_stdev}

    if filtered_numeric:
        figure(figsize=(17, 5))
        plot_multibar_chart(
            filtered_numeric,
            filtered_outliers,
            title="Nr of outliers per variable",
            xlabel="variables",
            ylabel="nr outliers",
            percentage=False,
        )
        savefig(f"images/{file_tag}_outliers.png")
    else:
        print("There are no outliers in any numeric variable.")
else:
    print("There are no numeric variables.")

#---------Numeric Variables Histograms----------

from dslabs_functions import set_chart_labels
if [] != numeric:
    fig, axs = subplots(
        rows, cols, figsize=(cols * 4, rows * 4), squeeze=False
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
    tight_layout(rect=[0, 0.05, 1, 1])
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
    rows, cols = define_grid(len(symbolic))
    fig, axs = subplots(
        rows, cols, figsize=(cols * 6, rows * 6), squeeze=False
    )
    i, j = 0, 0
    for n in range(len(symbolic)):
        counts: Series = data[symbolic[n]].value_counts()
        plot_bar_chart(
            [truncate_cat(str(c)) for c in counts.index],
            counts.to_list(),
            ax=axs[i, j],
            title="Histogram for %s" % symbolic[n],
            xlabel=symbolic[n],
            ylabel="nr records",
            percentage=False,
        )
        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
    tight_layout()
    savefig(f"images/{file_tag}_histograms_symbolic.png")
else:
    print("There are no symbolic variables.")

#---------Target Variable Distribution----------
values: Series = data[target].value_counts()

figure(figsize=(6, 3))
plot_bar_chart(
    values.index.to_list(),
    values.to_list(),
    title=f"Target distribution (target={target})",
)
savefig(f"images/{file_tag}_class_distribution.png")

