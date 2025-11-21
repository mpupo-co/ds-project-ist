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
filename = 'datasets/Combined_Flights_2022.csv'
file_tag = 'Combined_Flights_2022'

target = 'Cancelled'
index = 'FlightDate'
data = pd.read_csv(filename, index_col=index)
data = data.sample(frac=0.01, replace=False)
cols_to_drop = data.columns[
    data.columns.str.contains('ID', case=False) |
    data.columns.str.contains('Flight_Num', case=False) |
    data.columns.str.contains('Code', case=False) |
    data.columns.to_series().apply(
        lambda col: any(f'Div{i}' in col for i in range(1, 6))
    ) |
    data.columns.str.contains('Tail', case=False)
]

#print(cols_to_drop)
data = data.drop(columns=cols_to_drop)

variables_types: dict[str, list] = get_variable_types(data)
numeric: list[str] = variables_types["numeric"]

# choose which numeric vars you want distributions for
vars_with_dists = ["CRCDepTime", "DepTime","ArrTime", "AirTime", "CRSElapsedTime", "ActualElapsedTime", "Distance", "WheelsOff", "WheelsOn", "CRSArrTime", "ArrDelay", "ArrivalDelayGoups", "TaxiIn", "TaxiOut"] 

if [] != numeric:
    rows = 6
    cols = 5
    fig, axs = subplots(
        rows, cols, figsize=(cols * 5, rows * 5), squeeze=False
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


