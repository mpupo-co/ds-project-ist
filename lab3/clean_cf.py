import pandas as pd 
from pandas import DataFrame, read_csv

# ---- Load dataset ----
filename = 'datasets/Combined_Flights_2022.csv'
file_tag = 'Combined_Flights_2022'

index = 'FlightDate'
data = pd.read_csv(filename, index_col=index)

# columns to drop based on domain knowledge:
# - removing identifiers
# - removing columns with a single unique value
# - removing columns with redundant information
# - removing columns with many missing values (avoidd data leakage), since cancelled flights may not have certain data recorded
cols_to_drop = data.columns[
    data.columns.str.contains('ID', case=False) |
    data.columns.str.contains('Flight_Num', case=False) |
    data.columns.str.contains('Code', case=False) |
    data.columns.str.contains('Div', case=False) |
    data.columns.str.contains('Wac', case=False) |
    data.columns.str.contains('Fips', case=False) |
    data.columns.str.contains('Marketing', case=False) |
    data.columns.str.contains('Tail', case=False) |
    (data.isna().sum() > 0) |  # columns with missing values
    (data.nunique()<=1) # colunas com todos os valores iguais
].tolist() + ['Origin', 'Dest', 'OriginStateName', 'DestStateName', 'Operating_Airline'] # dropping columns with redundant information

data = data.drop(columns=cols_to_drop)
# format time variables to hh:mm
time_var = ['CRSDepTime', 'CRSArrTime'] #variables in format hhmm
for var in time_var:
    def format_time(x):
        if pd.isna(x):
            return ""
        else:
            x_int = int(x)
            x_str = str(x_int).zfill(4)
            return x_str[:2] + ':' + x_str[2:]
    data[var] = data[var].apply(format_time)  
sample=data.sample(frac=0.05, replace=False, random_state=42)

# ---- Save cleaned dataset ----
data.to_csv('datasets/'+file_tag+'_clean.csv')
sample.to_csv('datasets/'+file_tag+'_clean_sample.csv') 
