import pandas as pd 
from pandas import DataFrame, read_csv

#----------Load dataset----------
filename = 'datasets/Combined_Flights_2022_clean.csv'
file_tag = 'Combined_Flights_2022'

target = 'Cancelled'
index = 'FlightDate'
data = pd.read_csv(filename, index_col=index)

#---------Global Numeric Variables Boxplot----------
from matplotlib.pyplot import savefig, figure, tight_layout, subplots
from dslabs_functions import get_variable_types

variables_types: dict[str, list] = get_variable_types(data)
numeric = variables_types["numeric"]
symbolic = variables_types["symbolic"]

move_cols = data.columns[data.columns.str.contains('Group', case=False)].tolist() + ['Quarter', 'Month', 'DayofMonth', 'DayOfWeek']
# move move_cols from numeric → symbolic
for var in move_cols:
    numeric.remove(var)
    symbolic.append(var)

#---------Global Numeric Variables Boxplot----------
if [] != numeric:
    fig, ax = subplots(figsize=(18, 10))
    data[numeric].boxplot(rot=45, ax=ax)
    tight_layout()
    savefig(f"images/{file_tag}_global_boxplot.png")
else:
    print("There are no numeric variables.")

#---------Numeric Variables Boxplots----------
from numpy import ndarray
from matplotlib.figure import Figure
from matplotlib.pyplot import savefig,  subplots
from dslabs_functions import define_grid, HEIGHT, get_variable_types
from matplotlib.pyplot import savefig, figure, tight_layout, subplots
from dslabs_functions import get_variable_types

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
        axs[i, j].boxplot(data[numeric[n]].dropna().values, whis=3.2)
        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
    savefig(f"images/{file_tag}_single_boxplots.png")
else:
    print("There are no numeric variables.")

#---------Outliers Analysis----------
from dslabs_functions import count_outliers, plot_multibar_chart
if [] != numeric:
    outliers = count_outliers(data, numeric, nrstdev=2, iqrfactor=3.2)
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