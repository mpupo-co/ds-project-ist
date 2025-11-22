import pandas as pd 
from pandas import DataFrame, read_csv

# ---- Load dataset ----
filename = 'datasets/Combined_Flights_2022.csv'
file_tag = 'Combined_Flights_2022'

index = 'FlightDate'
data = pd.read_csv(filename, index_col=index)

cols_to_drop = data.columns[
    data.columns.str.contains('ID', case=False) |
    data.columns.str.contains('Flight_Num', case=False) |
    data.columns.str.contains('Code', case=False) |
    data.columns.to_series().apply(
        lambda col: any(f'Div{i}' in col for i in range(1, 6))
    ) |
    data.columns.str.contains('Tail', case=False) |
    (data.nunique()<=1) # colunas com todos os valores iguais
]

data = data.drop(columns=cols_to_drop)

# ---- Save cleaned dataset ----
data.to_csv(file_tag+'_clean.csv')