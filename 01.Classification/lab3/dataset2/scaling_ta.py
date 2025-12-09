from pandas import DataFrame, Series
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def data_scaling(dfs: list[DataFrame], file_tag: str, target: str, strategy: str = "zscore") -> tuple[DataFrame, DataFrame, bool]:
    print(f"=== Applying data scaling strategy: {strategy} ===")
    train_df_transf: DataFrame
    test_df_transf: DataFrame
    for i, df in enumerate(dfs):
        features = df.drop(columns=[target], inplace=False)
        target_data: Series = df[target]
        if strategy == "minmax":
            scaler = MinMaxScaler(feature_range=(0, 1), copy=True).fit(features)
        elif strategy == "zscore":
            scaler = StandardScaler(with_mean=True, with_std=True, copy=True).fit(features)
        else:
            raise ValueError(f"Scaling strategy {strategy} not recognized.")
        df_transf = DataFrame(scaler.transform(features), index=df.index, columns=features.columns)
        df_transf[target] = target_data.values
        if i == 0:
            train_df_transf = df_transf
        else:
            test_df_transf = df_transf

    train_df_transf.to_csv(f"data/{file_tag}_train_{strategy}.csv", index=False)
    test_df_transf.to_csv(f"data/{file_tag}_test_{strategy}.csv", index=False)

    return train_df_transf, test_df_transf, True
