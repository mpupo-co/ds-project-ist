import os
import pandas as pd
from pandas import concat, DataFrame, Series
from sklearn.model_selection import train_test_split
from dslabs_functions import evaluate_approach, plot_multibar_chart
from matplotlib.pyplot import figure, savefig

filename = 'datasets/Combined_Flights_2022_encoded_scaled.csv'
file_tag = 'Combined_Flights_2022'
index_col = 'FlightDate'
target = 'Cancelled'

os.makedirs("images", exist_ok=True)
os.makedirs("datasets", exist_ok=True)

original: DataFrame = pd.read_csv(filename, index_col=index_col)
original.columns = original.columns.str.strip()

if target not in original.columns:
    raise KeyError(f"Target '{target}' not in columns: {original.columns.tolist()}")

y_all: Series = original[target]
X_all: DataFrame = original.drop(columns=[target])

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.3, stratify=y_all, random_state=42
)

train_orig: DataFrame = concat([X_train, y_train], axis=1)
test: DataFrame = concat([X_test, y_test], axis=1)

# -------------------
# Undersampling (train only)
# -------------------
target_count: Series = y_train.value_counts()
positive_class = target_count.idxmin()
negative_class = target_count.idxmax()

df_positives: DataFrame = train_orig[train_orig[target] == positive_class]
df_negatives: DataFrame = train_orig[train_orig[target] == negative_class]

df_neg_sample: DataFrame = df_negatives.sample(n=len(df_positives), random_state=42)
train_under: DataFrame = concat([df_positives, df_neg_sample], axis=0).sample(frac=1.0, random_state=42)

train_under.to_csv(f"datasets/{file_tag}_train_under.csv", index=True)
test.to_csv(f"datasets/{file_tag}_test.csv", index=True)

figure()
eval_under = evaluate_approach(train_under.copy(), test.copy(), target=target, metric="recall")
plot_multibar_chart(["NB", "KNN"], eval_under, title=f"{file_tag} evaluation (undersampling)", percentage=True)
savefig(f"images/{file_tag}_eval_balance_under.png")

# -------------------
# SMOTE (train only)
# -------------------
from imblearn.over_sampling import SMOTE

feature_cols = [c for c in train_orig.columns if c != target]

smote = SMOTE(sampling_strategy="minority", random_state=42)
smote_X, smote_y = smote.fit_resample(
    train_orig[feature_cols].values,
    train_orig[target].values
)

train_smote: DataFrame = concat(
    [DataFrame(smote_X, columns=feature_cols), Series(smote_y, name=target)],
    axis=1
).sample(frac=1.0, random_state=42)

# recreate index for training only
n_orig = len(train_orig)
n_total = len(train_smote)
n_synth = n_total - n_orig
train_smote.index = train_orig.index.append(pd.Index([f"SMOTE_{i}" for i in range(n_synth)], name=index_col))

train_smote.to_csv(f"datasets/{file_tag}_train_smote.csv", index=True)

figure()
eval_smote = evaluate_approach(train_smote.copy(), test.copy(), target=target, metric="recall")
plot_multibar_chart(["NB", "KNN"], eval_smote, title=f"{file_tag} evaluation (smote)", percentage=True)
savefig(f"images/{file_tag}_eval_balance_smote.png")
