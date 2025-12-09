from pandas import DataFrame
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dslabs_functions import encode_cyclic_variables, dummify
from math import pi

def ordinal_encoding(df: DataFrame):
    # Order lighting conditions from darkest to brightest; keep unknown last for clarity.
    lighting_condition_values: dict[str, int] = {
        "DARKNESS": 0,
        "DARKNESS, LIGHTED ROAD": 1,
        "DAWN": 2,
        "DAYLIGHT": 3,
        "DUSK": 4,
        "UNKNOWN": 5,
    }

    ordinal_encoding: dict[str, dict[str, int]] = {
        "lighting_condition": lighting_condition_values,
    }

    return df.replace(ordinal_encoding, inplace=False)

def one_hot_encoding(df: DataFrame) -> DataFrame:
    vars: list[str] = [
        "alignment",
        "trafficway_type",
        "traffic_control_device",
        "weather_condition",
        "first_crash_type",
        "roadway_surface_cond",
        "road_defect",
        "prim_contributory_cause",
    ]
    return dummify(df, vars)

def cyclic_encoding(df: DataFrame) -> DataFrame:
    cyclic_vars = [
        "crash_date_year",
        "crash_date_quarter",
        "crash_date_month",
        "crash_date_day",
        "crash_day_of_week",
        "crash_hour",
    ]

    year_list = df["crash_date_year"].unique().tolist()
    year_list.sort()
    # run through years and give them the value of 2*pi* (index/number_of_years)
    year_values: dict[int, float] = {
        year: 2 * pi * (i / len(year_list))
        for i, year in enumerate(sorted(year_list))
    }

    month_values: dict[int, float] = {}
    for month in range(1, 13):
        month_values[month] = 2 * pi * ((month - 1) / 12)
    
    day_values: dict[int, float] = {}
    for day in range(1, 32):
        day_values[day] = 2 * pi * ((day - 1) / 31)

    quarter_values: dict[str, int] = {
        "1": 0,
        "2": pi / 2,
        "3": pi,
        "4": 3 * pi / 2,
    }

    day_of_week_values: dict[int, float] = {}
    for day in range(1, 8):
        day_of_week_values[day] = 2 * pi * ((day - 1) / 7)

    hour_values: dict[int, float] = {}
    for hour in range(0, 24):
        hour_values[hour] = 2 * pi * (hour / 24)

    cyclic_encoding_values: dict[str, dict[int, float]] = {
        "crash_date_year": year_values,
        "crash_date_quarter": quarter_values,
        "crash_date_month": month_values,
        "crash_date_day": day_values,
        "crash_day_of_week": day_of_week_values,
        "crash_hour": hour_values,
    }
    df = df.replace(cyclic_encoding_values, inplace=False)

    encode_cyclic_variables(df, cyclic_vars)
    df.drop(columns=cyclic_vars, inplace=True)
    return df

def binary_encoding(df: DataFrame) -> DataFrame:
    crash_type_values: dict[str, int] = {
        "NO INJURY / DRIVE AWAY": 0,
        "INJURY AND / OR TOW DUE TO CRASH": 1,
    }

    intersection_related_i_values: dict[str, int] = {
        "N": 0,
        "Y": 1,
    }

    binary_encoding = {
        "crash_type": crash_type_values,
        "intersection_related_i": intersection_related_i_values,
    }

    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype("int8")

    return df.replace(binary_encoding, inplace=False)

def data_encoding(df: DataFrame, file_tag: str) -> DataFrame:
    print("=== Applying data encoding ===")
    df = ordinal_encoding(df)
    df = one_hot_encoding(df)
    df = cyclic_encoding(df)
    df = binary_encoding(df)
    df.to_csv(f"data/{file_tag}_encoded.csv", index=False)
    return df