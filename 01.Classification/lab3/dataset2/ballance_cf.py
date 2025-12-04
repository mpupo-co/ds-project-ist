import pandas as pd
from pandas import read_csv, concat, DataFrame, Series
from numpy import ndarray, array
from sklearn.model_selection import train_test_split
from dslabs_functions import (
    evaluate_approach,
    plot_multibar_chart
)
from matplotlib.pyplot import figure, savefig
from dslabs_functions import plot_bar_chart

# ---- Load dataset ----
filename = 'datasets/Combined_Flights_2022_encoded_scaled.csv'
file_tag = 'Combined_Flights_2022'

index = 'FlightDate'
target = 'Cancelled'
original = pd.read_csv(filename, index_col=index)

target_count: Series = original[target].value_counts()
positive_class = target_count.idxmin()
negative_class = target_count.idxmax()
# -----------------------
# ---- Undersampling ----
# -----------------------

df_positives: Series = original[original[target] == positive_class]
df_negatives: Series = original[original[target] == negative_class]

df_neg_sample: DataFrame = DataFrame(df_negatives.sample(len(df_positives)))
df_under: DataFrame = concat([df_positives, df_neg_sample], axis=0)
#df_under.to_csv(f"data/{filename}_under.csv", index=False)

y: array = df_under.pop(target).to_list()
X: ndarray = df_under.values
trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y, random_state=42)

train: DataFrame = concat(
    [DataFrame(trnX, columns=df_under.columns), DataFrame(trnY, columns=[target])], axis=1
)

test: DataFrame = concat(
    [DataFrame(tstX, columns=df_under.columns), DataFrame(tstY, columns=[target])], axis=1
)

figure()
eval: dict[str, list] = evaluate_approach(train, test, target=target, metric="recall")
plot_multibar_chart(
    ["NB", "KNN"], eval, title=f"{file_tag} evaluation (undersampling)", percentage=True
)
savefig(f"images/{file_tag}_eval_ballance.png")


# ---------------
# ---- SMOTE ----
# ---------------
from numpy import ndarray
from pandas import Series
from imblearn.over_sampling import SMOTE

# keep a reference to the feature columns (without target)
feature_cols = [c for c in original.columns if c != target]

y: ndarray = original[target].values
X: ndarray = original[feature_cols].values

smote: SMOTE = SMOTE(sampling_strategy="minority", random_state=42)
smote_X, smote_y = smote.fit_resample(X, y)

df_smote: DataFrame = concat(
    [
        DataFrame(smote_X, columns=feature_cols),
        Series(smote_y, name=target)
    ],
    axis=1
)

#  Recreate the index
n_orig = len(original)
n_total = len(df_smote)
n_synth = n_total - n_orig

orig_index = original.index
synth_index = pd.Index([f"SMOTE_{i}" for i in range(n_synth)])

new_index = orig_index.append(synth_index)

df_smote.index = new_index
df_smote.index.name = index 

y_series: Series = df_smote[target]
X_df: DataFrame = df_smote.drop(columns=[target])

trnX, tstX, trnY, tstY = train_test_split(
    X_df,
    y_series,
    train_size=0.7,
    stratify=y_series,
    random_state=42,
)

# build train/test DataFrames (index is preserved automatically)
train: DataFrame = concat([trnX, trnY], axis=1)
test: DataFrame = concat([tstX, tstY], axis=1)

figure()
eval_results: dict[str, list] = evaluate_approach(train, test, target=target, metric="recall")
plot_multibar_chart(
    ["NB", "KNN"], eval_results, title=f"{file_tag} evaluation (smote)", percentage=True
)
savefig(f"images/{file_tag}_eval_smote.png")

df_smote.to_csv(
    f"datasets/{file_tag}_encoded_sca_smote.csv",
    index=True,           # write index to file
    index_label=index,    # column name for the index (FlightDate)
)