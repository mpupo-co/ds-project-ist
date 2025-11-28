from numpy import ndarray
from pandas import DataFrame, Series, concat

def under_balancing(df: DataFrame, target: str, file_tag: str) -> DataFrame:
    target_count: Series = df[target].value_counts()
    positive_class = target_count.idxmin()
    negative_class = target_count.idxmax()

    df_positives: Series = df[df[target] == positive_class]
    df_negatives: Series = df[df[target] == negative_class]

    df_neg_sample: DataFrame = DataFrame(df_negatives.sample(len(df_positives)))
    df_under: DataFrame = concat([df_positives, df_neg_sample], axis=0)
    df_under.to_csv(f"data/{file_tag}_under.csv", index=False)
    return df_under

def over_balancing(df: DataFrame, target: str, file_tag: str) -> DataFrame:
    target_count: Series = df[target].value_counts()
    positive_class = target_count.idxmin()
    negative_class = target_count.idxmax()

    df_positives: Series = df[df[target] == positive_class]
    df_negatives: Series = df[df[target] == negative_class]

    df_pos_sample: DataFrame = DataFrame(df_positives.sample(len(df_negatives), replace=True))
    df_over: DataFrame = concat([df_pos_sample, df_negatives], axis=0)
    df_over.to_csv(f"lab3/data/{file_tag}_over.csv", index=False)
    return df_over

def smote_balancing(df: DataFrame, target: str, file_tag: str, random_state: int) -> DataFrame:
    from imblearn.over_sampling import SMOTE

    smote: SMOTE = SMOTE(sampling_strategy="minority", random_state=random_state)
    df_cp = df.copy(deep=True)
    y = df_cp.pop(target).values
    X: ndarray = df_cp.values
    smote_X, smote_y = smote.fit_resample(X, y)
    df_smote: DataFrame = concat([DataFrame(smote_X), DataFrame(smote_y)], axis=1)
    df_smote.columns = list(df_cp.columns) + [target]
    df_smote.to_csv(f"lab3/data/{file_tag}_smote.csv", index=False)

    return df_smote

def data_balancing(df: DataFrame, target: str, strategy: str = "under", file_tag: str = "traffic_accidents", random_state: int = 42) -> tuple[DataFrame, bool]:
    if strategy == "under":
        df_balanced = under_balancing(df, target, file_tag)
    elif strategy == "over":
        df_balanced = over_balancing(df, target, file_tag)
    else:
        df_balanced = smote_balancing(df, target, file_tag, random_state=random_state)

    return df_balanced, True