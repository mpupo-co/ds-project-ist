import os
import sys
from math import pi

import numpy as np
from pandas import DataFrame

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../utils")))
from dslabs_functions import dummify

def ordinal_encoding(df: DataFrame) -> DataFrame:
    # Order lighting conditions from darkest to brightest; keep unknown last for clarity.
    lighting_condition_values: dict[str, int] = {
        "DARKNESS": 0,
        "DARKNESS, LIGHTED ROAD": 1,
        "DAWN": 2,
        "DAYLIGHT": 3,
        "DUSK": 4,
        "UNKNOWN": 5,
        "OTHERS": 6,
    }

    ordinal_encoding: dict[str, dict[str, int]] = {
        "lighting_condition": lighting_condition_values,
    }
    df.replace(ordinal_encoding, inplace=True)
    return df

def one_hot_encoding(df: DataFrame, encoder=None) -> DataFrame:
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
    return dummify(df, vars, encoder)

CYCLIC_COLUMNS: dict[str, int] = {
    "crash_date_quarter": 4,
    "crash_date_month": 12,
    "crash_date_day": 31,
    "crash_day_of_week": 7,
    "crash_hour": 24,
}


def _encode_angle(values: np.ndarray, period: int) -> tuple[np.ndarray, np.ndarray]:
    angles = 2 * pi * (values % period) / period
    sin_vals = np.sin(angles).astype(np.float32, copy=False)
    cos_vals = np.cos(angles).astype(np.float32, copy=False)
    return sin_vals, cos_vals


def cyclic_encoding(df: DataFrame) -> DataFrame:
    processed: list[str] = []

    if "crash_date_year" in df.columns:
        years = df["crash_date_year"].to_numpy(copy=False)
        uniq, inv = np.unique(years, return_inverse=True)
        if len(uniq):
            sin_vals, cos_vals = _encode_angle(inv.astype(np.float32), max(len(uniq), 1))
            df["crash_date_year_sin"] = sin_vals
            df["crash_date_year_cos"] = cos_vals
            processed.append("crash_date_year")

    for column, period in CYCLIC_COLUMNS.items():
        if column not in df.columns:
            continue
        values = df[column].to_numpy(copy=False).astype(np.float32, copy=False)
        sin_vals, cos_vals = _encode_angle(values, period)
        df[f"{column}_sin"] = sin_vals
        df[f"{column}_cos"] = cos_vals
        processed.append(column)

    if processed:
        df.drop(columns=processed, inplace=True, errors="ignore")
    return df

def _encode_binary_column(df: DataFrame, column: str, mapping: dict[str, int]) -> None:
    if column not in df.columns:
        return
    df[column] = df[column].map(mapping).astype("int8")


def binary_encoding(df: DataFrame, single_flag: bool) -> DataFrame:
    bool_cols = df.select_dtypes(include="bool").columns
    if len(bool_cols):
        df.loc[:, bool_cols] = df.loc[:, bool_cols].astype("int8")

    _encode_binary_column(
        df,
        "intersection_related_i",
        {
            "N": 0,
            "Y": 1,
        },
    )

    if not single_flag:
        _encode_binary_column(
            df,
            "crash_type",
            {
                "NO INJURY / DRIVE AWAY": 0,
                "INJURY AND / OR TOW DUE TO CRASH": 1,
            },
        )

    return df

def data_encoding(df: DataFrame, single_flag: bool, encoder=None, return_encoder: bool = False):
    print("=== Applying data encoding ===")
    df = ordinal_encoding(df)
    df, fitted_encoder = one_hot_encoding(df, encoder)
    df = cyclic_encoding(df)
    df = binary_encoding(df, single_flag)
    if return_encoder:
        return df, fitted_encoder
    return df