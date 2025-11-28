from pandas import DataFrame
from dslabs_functions import encode_cyclic_variables, dummify

def ordinal_encoding(df: DataFrame):
    damage_values: dict[str, int] = {
        "$500 OR LESS": 0,
        "$501 - $1,500": 1,
        "OVER $1,500": 2,
    }
    most_severe_injury_values: dict[str, int] = {
        "NO INDICATION OF INJURY": 0,
        "REPORTED, NOT EVIDENT": 1,
        "NONINCAPACITATING INJURY": 2,
        "INCAPACITATING INJURY": 3,
        "FATAL": 4,
    }
    weather_condition_values: dict[str, int] = {
        "CLEAR": 0,
        "CLOUDY/OVERCAST": 1,
        "FOG/SMOKE/HAZE": 2,
        "RAIN": 3,
        "SLEET/HAIL": 4,
        "SNOW": 5,
        "BLOWING SNOW": 6,
        "FREEZING RAIN/DRIZZLE": 7,
        "BLOWING SAND, SOIL, DIRT": 8,
        "SEVERE CROSS WIND GATE": 9,
        "OTHER": 10,
        "UNKNOWN": 11,
    }
    # Order lighting conditions from darkest to brightest; keep unknown last for clarity.
    lighting_condition_values: dict[str, int] = {
        "DARKNESS": 0,
        "DARKNESS, LIGHTED ROAD": 1,
        "DAWN": 2,
        "DAYLIGHT": 3,
        "DUSK": 4,
        "UNKNOWN": 5,
    }

    roadway_surface_cond_values: dict[str, int] = {
        "DRY": 0,
        "WET": 1,
        "SNOW OR SLUSH": 2,
        "SAND, MUD, DIRT": 3,
        "ICE": 4,
        "OTHER": 5,
        "UNKNOWN": 6,
    }

    ordinal_encoding: dict[str, dict[str, int]] = {
        "damage": damage_values,
        #"most_severe_injury": most_severe_injury_values,
        #"weather_condition": weather_condition_values,
        "lighting_condition": lighting_condition_values,
        "roadway_surface_cond": roadway_surface_cond_values,
    }

    return df.replace(ordinal_encoding, inplace=False)

def one_hot_encoding(df: DataFrame) -> DataFrame:
    vars: list[str] = [
        "alignment",
        "traffic_control_device",
        "trafficway_type",
        "road_defect",
        "prim_contributory_cause",
        "first_crash_type",
        "weather_condition",
    ]
    return dummify(df, vars)

def cyclic_encoding(df: DataFrame) -> DataFrame:
    df = derive_date_variables(df, ["crash_date"])
    cyclic_vars = [
        "crash_date_year",
        "crash_date_quarter",
        "crash_date_month",
        "crash_date_day"
    ]
    encode_cyclic_variables(df, cyclic_vars)
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

def derive_date_variables(df: DataFrame, date_vars: list[str]) -> DataFrame:
    for date in date_vars:
        df[f"{date}_year"] = df[date].dt.year
        df[f"{date}_quarter"] = df[date].dt.quarter
        df[f"{date}_month"] = df[date].dt.month
        df[f"{date}_day"] = df[date].dt.day
    df.drop(columns=date_vars, inplace=True)
    return df

def data_encoding(df: DataFrame) -> DataFrame:
    df = ordinal_encoding(df)
    df = one_hot_encoding(df)
    df = cyclic_encoding(df)
    df = binary_encoding(df)
    return df