import pandas as pd
import numpy as np
from dslabs_functions import *

file_tag = "TrafficTwoMonth"
filename = "datasets/TrafficTwoMonth.csv"
target = "Total"

data = pd.read_csv(filename)

# ------------------------
# ---- DIMENSIONALITY ----
# ------------------------

# ---- Create time series index ----
# Detect month blocks from Date resets (31 -> 1)
data['month_block'] = (data['Date'].diff() < 0).cumsum() + 7

YEAR = 2025

# Parse time
time_parsed = pd.to_datetime(data['Time'], format='%I:%M:%S %p')

# Build datetime
data['datetime'] = pd.to_datetime({
    'year': YEAR,
    'month': data['month_block'],
    'day': data['Date'],
    'hour': time_parsed.dt.hour,
    'minute': time_parsed.dt.minute,
    'second': time_parsed.dt.second,
})

data.set_index('datetime', inplace=True)

# ------------------------
# ---- EXOGENOUS VARS ----
# ------------------------

# ---- Day of Week → cyclic encoding ----
# Assume values like: Monday, Tuesday, ...
dow_map = {
    'Monday': 0, 'Tuesday': 1, 'Wednesday': 2,
    'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6
}

data['dow_num'] = data['Day of the week'].map(dow_map)

data['dow_sin'] = np.sin(2 * np.pi * data['dow_num'] / 7)
data['dow_cos'] = np.cos(2 * np.pi * data['dow_num'] / 7)

# ---- Traffic Situation → one-hot encoding ----
# Example categories: Low, Moderate, Heavy (adapt automatically)
traffic_dummies = pd.get_dummies(
    data['Traffic Situation'],
    prefix='traffic',
    drop_first=False
)

data = pd.concat([data, traffic_dummies], axis=1)

# ------------------------
# ---- DROP RAW VARS ----
# ------------------------

data.drop(
    columns=[
        'Time',
        'month_block',
        'Date',
        'Day of the week',
        'Traffic Situation',
        'dow_num'
    ],
    inplace=True
)

# ------------------------
# ---- FINAL CHECK ----
# ------------------------

print("Nr. Records =", data.shape[0])
print("Nr. Variables =", data.shape[1])
print("First timestamp:", data.index[0])
print("Last timestamp :", data.index[-1])

# ------------------------
# ---- SAVE DATASET ----
# ------------------------

data.to_csv(f'datasets/{file_tag}_clean.csv', index=True)
