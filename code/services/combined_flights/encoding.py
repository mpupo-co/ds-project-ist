import os
import sys
import numpy as np
import pandas as pd
from math import pi
from pandas import DataFrame
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../utils")))
from dslabs_functions import dummify

STATE_TO_REGION = {
    # Northeast
    "CT": "Northeast",
    "ME": "Northeast",
    "MA": "Northeast",
    "NH": "Northeast",
    "RI": "Northeast",
    "VT": "Northeast",
    "NJ": "Northeast",
    "NY": "Northeast",
    "PA": "Northeast",
    # Midwest
    "IL": "Midwest",
    "IN": "Midwest",
    "MI": "Midwest",
    "OH": "Midwest",
    "WI": "Midwest",
    "IA": "Midwest",
    "KS": "Midwest",
    "MN": "Midwest",
    "MO": "Midwest",
    "NE": "Midwest",
    "ND": "Midwest",
    "SD": "Midwest",
    # South
    "DE": "South",
    "FL": "South",
    "GA": "South",
    "MD": "South",
    "NC": "South",
    "SC": "South",
    "VA": "South",
    "DC": "South",
    "WV": "South",
    "AL": "South",
    "KY": "South",
    "MS": "South",
    "TN": "South",
    "AR": "South",
    "LA": "South",
    "OK": "South",
    "TX": "South",
    # West
    "AZ": "West",
    "CO": "West",
    "ID": "West",
    "MT": "West",
    "NV": "West",
    "NM": "West",
    "UT": "West",
    "WY": "West",
    "AK": "West",
    "CA": "West",
    "HI": "West",
    "OR": "West",
    "WA": "West",
    # Territories / Other
    "PR": "Outside",
    "VI": "Outside",
    "MP": "Outside",
    "AS": "Outside",
    "TT": "Outside",
}

def ordinal_encoding(df: DataFrame) -> DataFrame:
    """Encode geographic hierarchy (region -> state -> city)."""
    if {"OriginState", "DestState"}.issubset(df.columns):
        df["OriginRegion"] = df["OriginState"].map(STATE_TO_REGION)
        df["DestRegion"] = df["DestState"].map(STATE_TO_REGION)

    city_series = pd.concat(
        [
            df.get("OriginCityName", pd.Series(dtype=object)),
            df.get("DestCityName", pd.Series(dtype=object)),
        ],
        ignore_index=True,
    )

    city_by_state: dict[str, set[str]] = {}
    for value in city_series.dropna():
        if not isinstance(value, str):
            continue
        parts = value.split(",", 1)
        if len(parts) != 2:
            continue
        city = parts[0].strip()
        state = parts[1].strip()
        city_by_state.setdefault(state, set()).add(city)

    regions = sorted(set(STATE_TO_REGION.values()))
    region_enc = {region: idx + 1 for idx, region in enumerate(regions)}
    state_enc = {}
    for region in regions:
        states_in_region = sorted([s for s, r in STATE_TO_REGION.items() if r == region])
        state_enc.update({state: idx + 1 for idx, state in enumerate(states_in_region)})

    city_enc: dict[str, dict[str, int]] = {}
    for state, cities in city_by_state.items():
        city_enc[state] = {city: idx + 1 for idx, city in enumerate(sorted(cities))}

    for column in ("OriginCityName", "DestCityName"):
        if column in df.columns:
            df[column] = df[column].astype(str).str.split(",", n=1).str[0].str.strip()

    df["OriginCity_enc"] = df.apply(
        lambda row: city_enc.get(row.get("OriginState"), {}).get(row.get("OriginCityName"), 0),
        axis=1,
    )
    df["DestCity_enc"] = df.apply(
        lambda row: city_enc.get(row.get("DestState"), {}).get(row.get("DestCityName"), 0),
        axis=1,
    )

    origin_region_series = df.get("OriginRegion", pd.Series(index=df.index, dtype=object))
    dest_region_series = df.get("DestRegion", pd.Series(index=df.index, dtype=object))
    origin_state_series = df.get("OriginState", pd.Series(index=df.index, dtype=object))
    dest_state_series = df.get("DestState", pd.Series(index=df.index, dtype=object))

    df["OriginRegion_enc"] = origin_region_series.map(region_enc)
    df["DestRegion_enc"] = dest_region_series.map(region_enc)
    df["OriginState_enc"] = origin_state_series.map(state_enc)
    df["DestState_enc"] = dest_state_series.map(state_enc)

    for column in [
        "OriginRegion_enc",
        "DestRegion_enc",
        "OriginState_enc",
        "DestState_enc",
        "OriginCity_enc",
        "DestCity_enc",
    ]:
        if column in df.columns:
            df[column] = df[column].astype("Int64")

    df["OriginGeoNum"] = (
        df.get("OriginRegion_enc", 0).fillna(0) * 1000
        + df.get("OriginState_enc", 0).fillna(0) * 100
        + df.get("OriginCity_enc", 0).fillna(0)
    )
    df["DestGeoNum"] = (
        df.get("DestRegion_enc", 0).fillna(0) * 1000
        + df.get("DestState_enc", 0).fillna(0) * 100
        + df.get("DestCity_enc", 0).fillna(0)
    )

    drop_cols = [
        "OriginRegion",
        "DestRegion",
        "OriginRegion_enc",
        "DestRegion_enc",
        "OriginState_enc",
        "DestState_enc",
        "OriginCity_enc",
        "DestCity_enc",
        "OriginCityName",
        "DestCityName",
        "OriginState",
        "DestState",
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors="ignore")
    return df

def one_hot_encoding(df: DataFrame, encoder=None) -> DataFrame:
    """One-hot encode airline company identifiers."""

    categorical = ["Airline"] if "Airline" in df.columns else []
    if not categorical:
        return df, encoder
    return dummify(df, categorical, encoder)

def _build_time_encoders() -> dict[str, dict]:
    week_days = {i: 2 * pi * (i - 1) / 7 for i in range(1, 8)}
    quarter = {i: 2 * pi * (i - 1) / 4 for i in range(1, 5)}
    month = {m: 2 * pi * (m - 1) / 12 for m in range(1, 13)}
    day = {d: 2 * pi * (d - 1) / 31 for d in range(1, 32)}

    time_lookup = {}
    for hour in range(0, 24):
        for minute in range(0, 60):
            label = f"{hour:02d}:{minute:02d}"
            total_minutes = hour * 60 + minute
            time_lookup[label] = 2 * pi * total_minutes / (24 * 60)
    time_lookup["24:00"] = time_lookup["00:00"]

    slots = {}
    for hour in range(24):
        start = hour * 60
        end = start + 59
        midpoint = (start + end) / 2
        slots[f"{hour:02d}00-{hour:02d}59"] = 2 * pi * midpoint / (24 * 60)
    irregular = "0001-0559"
    start_str, end_str = irregular.split("-")
    start_min = int(start_str[:2]) * 60 + int(start_str[2:])
    end_min = int(end_str[:2]) * 60 + int(end_str[2:])
    slots[irregular] = 2 * pi * ((start_min + end_min) / 2) / (24 * 60)

    return {
        "DayOfWeek": week_days,
        "CRSDepTime": time_lookup,
        "CRSArrTime": time_lookup,
        "DepTimeBlk": slots,
        "ArrTimeBlk": slots,
        "Quarter": quarter,
        "Month": month,
        "DayofMonth": day,
    }

ENCODE_CYCLIC = _build_time_encoders()

def cyclic_encoding(df: DataFrame) -> DataFrame:
    """Replace temporal features with sin/cos projections."""

    columns_to_drop: list[str] = []
    for column, mapping in ENCODE_CYCLIC.items():
        if column not in df.columns:
            continue
        df[column] = df[column].map(mapping)
        values = df[column].to_numpy(dtype=np.float32, copy=False)
        df[f"{column}_sin"] = np.sin(values, dtype=np.float32)
        df[f"{column}_cos"] = np.cos(values, dtype=np.float32)
        columns_to_drop.append(column)

    if columns_to_drop:
        df.drop(columns=columns_to_drop, inplace=True, errors="ignore")
    return df


def _encode_binary_column(df: DataFrame, column: str, mapping: dict[str, int]) -> None:
    if column not in df.columns:
        return
    df[column] = df[column].map(mapping).astype("Int8")


def binary_encoding(df: DataFrame, single_flag: bool) -> DataFrame:
    """Convert binary flags and add frequency encodings for airports."""
    bool_cols = df.select_dtypes(include="bool").columns
    if len(bool_cols):
        df.loc[:, bool_cols] = df.loc[:, bool_cols].astype("Int8")

    _encode_binary_column(df, "Cancelled", {"N": 0, "Y": 1, 0: 0, 1: 1})

    for column in ("Origin", "Dest"):
        if column not in df.columns:
            continue
        freq = df[column].value_counts(normalize=True)
        df[f"{column}_freq"] = df[column].map(freq)
        df.drop(columns=column, inplace=True)

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