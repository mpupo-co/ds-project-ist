import pandas as pd 
from dslabs_functions import dummify


# ---- Load dataset ----
filename = 'datasets/Combined_Flights_2022_clean_sample.csv'
file_tag = 'Combined_Flights_2022'

target = 'Cancelled'
index = 'FlightDate'
data = pd.read_csv(filename, index_col=index)


'''
Econde variables include:
 1. Ordinal+logic encoding:
    - geographic variables (cities and states names): 'OriginCityName', 'OriginState', 'DestCityName', 'DestState'
 2. Cyclic encoding
    - time variables (hh:mm): 'CRSDepTime', 'CRSArrTime'
    - time blocks: 'DepTimeBlk', 'ArrTimeBlk'
    - day of week: 'DayOfWeek'
 3. One-hot encoding:
    - airlines companies: 'Airline'
 4. Frequency encoding
    - airports: 'Origin', 'Dest'
'''

# ---- Encode geographic variables ----

state_to_region = {
    # Northeast
    "CT":"Northeast", "ME":"Northeast", "MA":"Northeast", "NH":"Northeast",
    "RI":"Northeast", "VT":"Northeast", "NJ":"Northeast", "NY":"Northeast",
    "PA":"Northeast",

    # Midwest
    "IL":"Midwest", "IN":"Midwest", "MI":"Midwest", "OH":"Midwest",
    "WI":"Midwest", "IA":"Midwest", "KS":"Midwest", "MN":"Midwest",
    "MO":"Midwest", "NE":"Midwest", "ND":"Midwest", "SD":"Midwest",

    # South
    "DE":"South", "FL":"South", "GA":"South", "MD":"South", "NC":"South",
    "SC":"South", "VA":"South", "DC":"South", "WV":"South", "AL":"South",
    "KY":"South", "MS":"South", "TN":"South", "AR":"South", "LA":"South",
    "OK":"South", "TX":"South",

    # West
    "AZ":"West", "CO":"West", "ID":"West", "MT":"West", "NV":"West",
    "NM":"West", "UT":"West", "WY":"West", "AK":"West", "CA":"West",
    "HI":"West", "OR":"West", "WA":"West",

    # U.S. Outside territories / Other
    "PR": "Outside",  # Puerto Rico
    "VI": "Outside",  # U.S. Virgin Islands
    "MP": "Outside",  # Northern Mariana Islands
    "AS": "Outside",  # American Samoa
    "TT": "Outside"   # Used in your data for Pago Pago, Guam, Saipan
}

# Add Origin and Destination region columns
data["OriginRegion"] = data["OriginState"].map(state_to_region)
data["DestRegion"] = data["DestState"].map(state_to_region)

# Create a dictionary mapping states to their cities
state_dict = {}
for value in pd.concat([data['OriginCityName'], data['DestCityName']]).unique():
    # skip missing or non-string values
    if not isinstance(value, str):
        continue
    
    # split "City, State" into two parts
    parts = value.split(",")
    
    if len(parts) != 2:
        continue  # malformed entry, skip it
    
    city = parts[0].strip()
    state = parts[1].strip()
    
    # Add city to corresponding state in the dictionary
    if state not in state_dict:
        state_dict[state] = []
    
    state_dict[state].append(city)

# Create a dictionary mapping regions to states and their cities
region_dict = {}
for state, cities in state_dict.items():
    region = state_to_region.get(state)

    if region is None:
        continue  # caso estado não exista no mapa

    if region not in region_dict:
        region_dict[region] = {}

    region_dict[region][state] = cities

# Create an encode for regions
regions = sorted(set(state_to_region.values()))
region_enc = {region: i+1 for i, region in enumerate(regions)}

# Create an encode for states inside regions
state_enc = {}
for region in regions:
    states_in_region = sorted([s for s, r in state_to_region.items() if r == region])
    state_enc.update({state: i+1 for i, state in enumerate(states_in_region)})

# Create an encode for cities inside states
city_enc = {}
for state, cities in state_dict.items():
    unique_cities = sorted(set(cities))
    city_enc[state] = {city: i+1 for i, city in enumerate(unique_cities)}

# Extract city names from "state, city"
data['OriginCityName'] = data['OriginCityName'].str.split(',', n=1).str[0].str.strip()
data['DestCityName']   = data['DestCityName'].str.split(',', n=1).str[0].str.strip()

# City enconding
data['OriginCity_enc'] = data.apply(
    lambda row: city_enc.get(row['OriginState'], {}).get(row['OriginCityName'], 0),
    axis=1
)
data['DestCity_enc'] = data.apply(
    lambda row: city_enc.get(row['DestState'], {}).get(row['DestCityName'], 0),
    axis=1
)
# Region and State encoding
data['OriginRegion_enc'] = data['OriginRegion'].map(region_enc)
data['DestRegion_enc']   = data['DestRegion'].map(region_enc)
data['OriginState_enc']  = data['OriginState'].map(state_enc)
data['DestState_enc']    = data['DestState'].map(state_enc)

for var in ['OriginRegion_enc', 'OriginState_enc', 'OriginCity_enc',
            'DestRegion_enc', 'DestState_enc', 'DestCity_enc']:
    data[var] = data[var].astype('Int64')

# Combine into numeric 5-digit encoding
data['OriginGeoNum'] = (
    data['OriginRegion_enc'] * 1000 +   # Region in first digit
    data['OriginState_enc'] * 100 +      # State in next 2 digits
    data['OriginCity_enc']                # City in last 2 digits
)

data['DestGeoNum'] = (
    data['DestRegion_enc'] * 1000 +
    data['DestState_enc'] * 100 +
    data['DestCity_enc']
)

cols_to_drop = ['OriginRegion', 'DestRegion',
                'OriginRegion_enc', 'DestRegion_enc',
                'OriginState_enc', 'DestState_enc',
                'OriginCity_enc', 'DestCity_enc',
                'OriginCityName', 'DestCityName',
                'OriginState', 'DestState']

data = data.drop(columns=cols_to_drop)

# ---- Encode time variables ----
from math import pi, sin, cos

week_days: dict[int, float] = {
    1: 0,
    2: 2 * pi / 7,
    3: 4 * pi / 7,
    4: 6 * pi / 7,
    5: 8 * pi / 7,
    6: 10 * pi / 7,
    7: 12 * pi / 7
}

quarter: dict[int, float] = {
    1: 0,
    2: 2 * pi / 4,
    3: 4 * pi / 4,
    4: 6 * pi / 4
}

month: dict[int, float] = {}
for m in range(1, 13):
    angle = 2 * pi * (m - 1) / 12
    month[m] = angle

day: dict[int, float] = {}
for d in range(1, 32):
    angle = 2 * pi * (d - 1) / 31
    day[d] = angle

time: dict[str, float] = {}
for h in range(0, 24):
    for m in range(0, 60):
        time_str = f"{h:02d}:{m:02d}"
        total_min = h * 60 + m
        angle = 2 * pi * total_min / (24 * 60)
        time[time_str] = angle

        time[time_str] = angle
# add only the valid extra time
time["24:00"] = time["00:00"]

time_slots: dict[str, float] = {}
for h in range(24):
    start_min = h * 60
    end_min = h * 60 + 59
    slot = f"{h:02d}00-{h:02d}59"
    mid_min = (start_min + end_min) / 2
    angle = 2 * pi * mid_min / (24 * 60)
    time_slots[slot] = angle
irregular_slot = "0001-0559"
start_str, end_str = irregular_slot.split('-')
time_slots[irregular_slot] = 2 * pi * ((int(start_str[:2])*60 + int(start_str[2:]) + int(end_str[:2])*60 + int(end_str[2:])) / 2) / (24*60)

encode_cyclic: dict[str, dict] = {
    'DayOfWeek': week_days,
    'CRSDepTime': time,
    'CRSArrTime': time,
    'DepTimeBlk': time_slots,
    'ArrTimeBlk': time_slots,
    'Quarter': quarter,
    'Month': month,
    'DayofMonth': day
}
data = data.replace(encode_cyclic)


for var in encode_cyclic.keys():
    data[var + '_sin'] = data[var].apply(lambda x: sin(x) if pd.notnull(x) else x)
    data[var + '_cos'] = data[var].apply(lambda x: cos(x) if pd.notnull(x) else x)
    data = data.drop(columns=[var])

# ---- Dummify 'Airline' variables ----
data = dummify(data, ['Airline'])


# ---- Frequency Encoding for Dest, Origin ----
airports = ['Origin', 'Dest']
for col in airports:
    freq = data[col].value_counts(normalize=True)  # frequência relativa
    data[col + '_freq'] = data[col].map(freq)
data = data.drop(columns=airports)

# ---- Save encoded dataset ----
data.to_csv('datasets/'+file_tag+'_encoded.csv', index=True)
print (data.shape)