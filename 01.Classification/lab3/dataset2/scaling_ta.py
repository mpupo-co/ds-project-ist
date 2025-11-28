from pandas import DataFrame, Series
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def data_scaling(dfs: list[DataFrame], file_tag: str, target: str, strategy: str = "zscore") -> tuple[DataFrame, DataFrame, bool]:
    for i, df in enumerate(dfs):
        features = df.drop(columns=[target], inplace=False)
        target_data: Series = df[target]
        if strategy == "minmax":
            scaler = MinMaxScaler(feature_range=(0, 1), copy=True).fit(features)
        if strategy == "zscore":
            scaler = StandardScaler(with_mean=True, with_std=True, copy=True).fit(features)
        else:
            raise ValueError(f"Scaling strategy {strategy} not recognized.")
        df_transf = DataFrame(scaler.transform(features), index=df.index, columns=features.columns)
        df_transf[target] = target_data.values
        if i == 0:
            train_df_transf = df_transf
            df_transf.to_csv(f"lab3/data/{file_tag}_train_scaled_{strategy}.csv", index="id")
        else:
            test_df_transf = df_transf
            df_transf.to_csv(f"lab3/data/{file_tag}_test_scaled_{strategy}.csv", index="id")

    return train_df_transf, test_df_transf, True
