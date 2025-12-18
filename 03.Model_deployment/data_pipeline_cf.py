
from pandas import DataFrame, concat, read_csv, to_datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from services.combined_flights.encoding import data_encoding
from services.combined_flights.outliers import get_variable_types, process_outliers
from services.combined_flights.balancing import data_balancing
from services.combined_flights.feature import feature_selection
import time

class DataPipeline:
    def __init__(self, outliers: str | None = None,
                 scaling: str | None = None,
                 balancing: str | None = None,
                 feature_selection: str | None = None,
                 target: str = "Cancelled"):
        self.outliers = outliers
        self.scaling = scaling
        self.balancing = balancing
        self.feature_selection = feature_selection
        self.target = target
        self.min_threshold = 0.7
        self.max_threshold = 0.18
        self.frequent_categories_: dict[str, set[str]] = {}
        self.one_hot_encoder = None
        self.scaler: StandardScaler | MinMaxScaler | None = None
        self.feature_columns_: list[str] = []
        self.features_to_drop: list[str] = []
        self.fitted: bool = False
        self.variable_types_: dict[str, list[str]] | None = None
        self.input_columns_: list[str] | None = None

    def fit(self, df: DataFrame) -> DataFrame:
        print("=== Fitting data pipeline ===")
        df_prepared = self._fit_internal(df.copy())
        df_prepared.to_csv("data/processed_cf.csv", index=False)
        self.fitted = True
        return df_prepared

    def transform(self, df: DataFrame, single_flag: bool = False) -> DataFrame:
        if not self.fitted:
            if single_flag:
                raise RuntimeError("Pipeline must be fit before transforming single rows")
            return self.fit(df)
        
        if self.input_columns_ is None:
            raise RuntimeError("Pipeline input columns are undefined. Fit the pipeline first.")

        vars = self._get_variable_types(df)
        df = data_cleaning(
            df,
            vars,
            frequent_categories=self.frequent_categories_,
            apply_rare_encoding=not single_flag,
        )
        df = data_encoding(df, single_flag=single_flag, encoder=self.one_hot_encoder)
        if not single_flag:
            df = self._scale_dataframe(df)
        df = self._drop_selected_features(df)
        features = df.drop(columns=[self.target], errors="ignore")
        features = features.reindex(columns=self.feature_columns_, fill_value=0)
        if self.feature_columns_:
            zero_only_cols = (features == 0).all(axis=0).sum()
            zero_ratio = zero_only_cols / len(self.feature_columns_)
            if zero_ratio > 0.99:
                raise ValueError(
                    "Input dataframe appears incompatible with fitted features (over 99% of feature columns are zero)."
                )

        if single_flag or self.target not in df.columns:
            return features

        features[self.target] = df[self.target].values
        return features

    def _fit_internal(self, df: DataFrame) -> DataFrame:
        vars = get_variable_types(df)
        self.variable_types_ = vars
        self.input_columns_ = list(df.columns)
        df, frequent_categories = data_cleaning(df, vars, return_metadata=True)
        self.frequent_categories_ = frequent_categories
        df, self.one_hot_encoder = data_encoding(
            df,
            single_flag=False,
            encoder=None,
            return_encoder=True,
        )

        train_df, test_df = split_train_test_data(df)
        if self.outliers:
            train_df, _ = process_outliers(train_df, file_tag="pipeline", strategy=self.outliers)

        if self.scaling:
            self._init_scaler(train_df)
            train_df = self._scale_dataframe(train_df)
            test_df = self._scale_dataframe(test_df)

        if self.balancing:
            train_df = data_balancing(
                train_df,
                target=self.target,
                strategy=self.balancing,
            )
        if self.feature_selection:
            before_cols = set(train_df.columns)
            train_df, test_df = feature_selection(
                [train_df, test_df],
                self.feature_selection,
                self.target,
                self.min_threshold,
                self.max_threshold,
            )
            dropped = [c for c in before_cols - set(train_df.columns) if c != self.target]
            self.features_to_drop = dropped
        else:
            self.features_to_drop = []

        self.feature_columns_ = [c for c in train_df.columns if c != self.target]

        df_processed = concat([train_df, test_df])
        return df_processed


    def _init_scaler(self, train_df: DataFrame) -> None:
        numeric_cols = train_df.select_dtypes(include="number").columns
        feature_cols = [c for c in numeric_cols if c != self.target]
        if not feature_cols:
            raise ValueError("No numeric features available for scaling.")
        if self.scaling == "minmax":
            scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
        elif self.scaling == "zscore":
            scaler = StandardScaler(with_mean=True, with_std=True, copy=True)
        else:
            raise ValueError(f"Scaling strategy {self.scaling} not recognized.")
        scaler.fit(train_df[feature_cols])
        self.scaler = scaler
        self.scaler_features_ = feature_cols

    def _scale_features(self, features: DataFrame) -> DataFrame:
        if not self.scaler:
            return features
        scaled = self.scaler.transform(features)
        return DataFrame(scaled, columns=features.columns, index=features.index)

    def _scale_dataframe(self, df: DataFrame) -> DataFrame:
        if not self.scaler:
            return df
        features = df.reindex(columns=self.scaler_features_, fill_value=0)
        scaled_features = self._scale_features(features)
        if self.target in df.columns:
            scaled_features[self.target] = df[self.target].values
        return scaled_features

    def _drop_selected_features(self, df: DataFrame) -> DataFrame:
        if not self.features_to_drop:
            return df
        return df.drop(columns=self.features_to_drop, errors="ignore")

    def _get_variable_types(self, df: DataFrame) -> dict[str, list[str]]:
        columns = list(df.columns)
        if self.variable_types_ and _columns_match(self.input_columns_, columns):
            return self.variable_types_
        return get_variable_types(df)

def _log_stage(name, start_time):
    duration = time.perf_counter() - start_time
    print(f"[pipeline] {name} took {duration:.2f}s")

def _columns_match(cols_a: list[str] | None, cols_b: list[str]) -> bool:
    if cols_a is None:
        return False
    return set(cols_a) == set(cols_b)

def load_dataset(filename: str, sample: bool = False, random_state: int = 42) -> DataFrame:
    df: DataFrame = read_csv(filename, na_values="")
    print(f'Loading dataset from {filename}...')
    if sample:
        df = df.sample(frac=0.3, random_state=random_state)
        print(f'Sampling 30% of the dataset for quicker processing...')
    return df

def split_train_test_data(df: DataFrame, test_size: float = 0.3, random_state: int = 42) -> tuple[DataFrame, DataFrame]:
    train_df = df.sample(frac=1-test_size, random_state=random_state)
    test_df = df.drop(train_df.index)
    print(f'Train shape: {train_df.shape}, Test shape: {test_df.shape}')

    return train_df, test_df

def derive_date_variables(df: DataFrame, date_vars: list[str]) -> DataFrame:
    for date in date_vars:
        date_series = to_datetime(df[date], errors="coerce", cache=True, format="%m/%d/%Y %I:%M:%S %p")
        if date_series.isna().all():
            continue
        df[date] = date_series
        df[f"{date}_year"] = date_series.dt.year
        df[f"{date}_month"] = date_series.dt.month
        df[f"{date}_day"] = date_series.dt.day
        df[f"{date}_quarter"] = ((date_series.dt.month - 1) // 3 + 1).astype("Int8")
    df.drop(columns=date_vars, inplace=True, errors="ignore")
    return df

def data_cleaning(
    df: DataFrame,
    vars: dict[str, list[str]],
    frequent_categories: dict[str, set[str]] | None = None,
    return_metadata: bool = False,
    rare_threshold: float = 0.005,
    apply_rare_encoding: bool = True,
) -> DataFrame | tuple[DataFrame, dict[str, set[str]]]:
    
    import pandas as pd
    print("=== Performing data cleaning ===")
    learned_categories: dict[str, set[str]] = {} if frequent_categories is None else frequent_categories

    if apply_rare_encoding:
        compute_freq = frequent_categories is None
        for col in vars["symbolic"]:
            if compute_freq:
                freq = df[col].value_counts(normalize=True)
                keep_values = set(freq[freq >= rare_threshold].index)
                learned_categories[col] = keep_values
            else:
                keep_values = frequent_categories.get(col)
                if keep_values is None:
                    continue

            mask = ~df[col].isin(keep_values)
            df.loc[mask, col] = "OTHERS"

    cols_to_drop = df.columns[
        df.columns.str.contains('Flight_Num', case=False) |
        df.columns.str.contains('Code', case=False) |
        df.columns.str.contains('Div', case=False) |
        df.columns.str.contains('Wac', case=False) |
        df.columns.str.contains('Fips', case=False) |
        df.columns.str.contains('Marketing', case=False) |
        df.columns.str.contains('Tail', case=False) |
        (df.isna().sum() > 0) |  # columns with missing values
        ((df.nunique() <= 1) if apply_rare_encoding else False) # colunas com todos os valores iguais
    ].tolist() + ['OriginStateName', 'DestStateName', 'Operating_Airline'] # dropping columns with redundant information
    df.drop(columns=cols_to_drop, inplace=True, errors="ignore")
    # format time variables to hh:mm
    time_var = ['CRSDepTime', 'CRSArrTime'] #variables in format hhmm
    for var in time_var:
        if var not in df.columns:
            continue

        def format_time(x):
            if pd.isna(x):
                return ""
            try:
                x_int = int(float(x))
            except (TypeError, ValueError):
                return ""
            x_str = str(x_int).zfill(4)
            return f"{x_str[:2]}:{x_str[2:]}"

        df[var] = df[var].apply(format_time)
    if return_metadata:
        return df, learned_categories
    return df