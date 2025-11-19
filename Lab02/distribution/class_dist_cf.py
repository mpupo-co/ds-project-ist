import pandas as pd 
from pandas import read_csv, Series
from matplotlib.pyplot import savefig, figure
from dslabs_functions import plot_bar_chart

#----------Load dataset----------
filename = 'datasets/Combined_Flights_2022.csv'
file_tag = 'Combined_Flights_2022'

target = 'Cancelled'
index = 'FlightDate'
data = pd.read_csv(filename, index_col=index)

#---------Target Variable Distribution----------
values: Series = data[target].value_counts()

figure(figsize=(6, 3))
plot_bar_chart(
    values.index.to_list(),
    values.to_list(),
    title=f"Target distribution (target={target})",
)
savefig(f"images/{file_tag}_class_distribution.png")