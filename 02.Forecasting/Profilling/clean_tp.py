import pandas as pd
from matplotlib.pyplot import figure, tight_layout
from dslabs_functions import *

file_tag = "TrafficTwoMonth"
filename = "datasets/TrafficTwoMonth.csv"
target = "Total"

data = pd.read_csv(filename)


# ------------------------
# ---- DIMENSIONALITY ----
# ------------------------

# ---- Create time series index with date formtat -> YYYY-MM-dd HH:MM:SS ----
# Detect "month blocks" from Date resets ( 31 -> 1)
# First block -> month 7 (July), second -> 8 (August), etc.
data['month_block'] = (data['Date'].diff() < 0).cumsum() + 7

YEAR = 2025

# Parse the Time column into hour/minute/second
time_parsed = pd.to_datetime(data['Time'], format='%I:%M:%S %p')

# Build a full datetime from year, month_block, Date and Time
data['datetime'] = pd.to_datetime({
    'year': YEAR,
    'month': data['month_block'],
    'day': data['Date'],
    'hour': time_parsed.dt.hour,
    'minute': time_parsed.dt.minute,
    'second': time_parsed.dt.second,
})

# Make this time series index
data.set_index('datetime', inplace=True)

# drop columns
data.drop(columns=['Time', 'month_block', 'Date', 'Day of the week', 'Traffic Situation'], inplace=True)


print("Nr. Records = ", data.shape)
print("First timestamp", data.index[0])
print("Last timestamp", data.index[-1])

# ----  Create/ save the encoded dataset in a csv file ----
data.to_csv(f'datasets/{file_tag}_clean.csv', index=True)



