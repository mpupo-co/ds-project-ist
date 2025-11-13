
#----------Import Libraries----------
import pandas as pd
from dslabs_functions import get_variable_types
from sklearn.model_selection import train_test_split
from dslabs_functions import plot_evaluation_results
from numpy import array
from matplotlib.pyplot import figure, savefig, show
from typing import Literal
from ds_class_functions import naive_Bayes_study, logistic_regression_study, knn_study, trees_study, mlp_study

#---------------------------------
#----------DATA CLEANING----------
#---------------------------------

#----------Load dataset----------
filename = 'datasets/traffic_accidents.csv'
file_tag = 'traffic_accidents'

target = 'crash_type'
index = 'crash_date'
data_all = pd.read_csv(filename, index_col=index)

#---------Sampling----------
data = data_all.sample(frac=0.05, replace=False)

#---------Data Cleaning----------
# Drop columns that are completely empty
data = data.dropna(axis=1, how='all')

# Drop rows that contain any missing values
data = data.dropna(axis=0, how='any')

# Drop non-numeric columns
variable_types = get_variable_types(data)
non_numeric_cols = variable_types['binary'] + variable_types['symbolic'] + variable_types['date']
cols_to_drop = [col for col in non_numeric_cols if col != target]
data = data.drop(columns=cols_to_drop)

#---------Encoding Target----------
txt_labels = data[target].unique() # original label names
encoding_target = {label: i for i, label in enumerate(txt_labels)}
data = data.replace({target: encoding_target}, inplace=False)

labels = data[target].unique() # encoded numeric labels

y = data.pop(target).values    
X = data.values

metrics = ['accuracy', 'precision', 'recall']
#---------------------------------
#-----------NAIVE BAYES-----------
#---------------------------------

# ---------- Train-test split ----------

trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y, random_state=42)

# ---------- Models' Comparision ----------

for eval_metric in metrics:
    figure()
    best_model, params = naive_Bayes_study(trnX, trnY, tstX, tstY, eval_metric)
    savefig(f"images/{file_tag}_nb_{eval_metric}_study.png")
    show()
# ---------- Best model performance ----------
    prd_trn: array = best_model.predict(trnX)
    prd_tst: array = best_model.predict(tstX)
    figure()
    plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels)
    savefig(f'images/{file_tag}_{params["name"]}_best_{params["metric"]}_eval.png')
    show()

#---------------------------------
#-------LOGISTIC REGRESSION-------
#---------------------------------

# ---------- Train-test split ----------
trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y, random_state=42)

# ---------- Models' Comparision ----------
for eval_metric in metrics:
    figure()
    best_model, params = logistic_regression_study(
        trnX,
        trnY,
        tstX,
        tstY,
        nr_max_iterations=1000,
        lag=250,
        metric=eval_metric,
    )
    savefig(f"images/{file_tag}_lr_{eval_metric}_study.png")
    show()
# ---------- Best model performance ----------
    prd_trn: array = best_model.predict(trnX)
    prd_tst: array = best_model.predict(tstX)
    figure()
    plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels)
    savefig(f'images/{file_tag}_lr_{params["name"]}_best_{params["metric"]}_eval.png')
    show()

#---------------------------------
#---------------KNN---------------
#---------------------------------

# ---------- Trai-test split ----------
trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y, random_state=42)

# ---------- Models' Comparision ----------
for eval_metric in metrics:
    figure()
    best_model, params = knn_study(trnX, trnY, tstX, tstY, k_max=20, metric=eval_metric)
    savefig(f'images/{file_tag}_knn_{eval_metric}_study.png')
    show()
    # ---------- Best model performance ----------
    prd_trn: array = best_model.predict(trnX)
    prd_tst: array = best_model.predict(tstX)
    figure()
    plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels)
    savefig(f'images/{file_tag}_knn_{params["name"]}_best_{params["metric"]}_eval.png')
    show()

#---------------------------------
#----------DECISION TREES---------
#---------------------------------

# ---------- Train-test split ----------
trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y, random_state=42)

# ---------- Models' Comparision ----------
for eval_metric in metrics:
    figure()
    best_model, params = trees_study(trnX, trnY, tstX, tstY, d_max=20, metric=eval_metric)
    savefig(f'images/{file_tag}_dt_{eval_metric}_study.png')
    show()
# ---------- Best model performance ----------
    prd_trn: array = best_model.predict(trnX)
    prd_tst: array = best_model.predict(tstX)
    figure()
    plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels)
    savefig(f'images/{file_tag}_dt_{params["name"]}_best_{params["metric"]}_eval.png')
    show()


#---------------------------------
#----------ML PERCEPTRON----------
#---------------------------------

# ---------- Train-test split ----------
trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y, random_state=42)

# ---------- Models' Comparision ----------
for eval_metric in metrics:
    figure()
    best_model, params = mlp_study(
        trnX,
        trnY,
        tstX,
        tstY,
        nr_max_iterations=1000,
        lag=250,
        metric=eval_metric,
    )
    savefig(f"images/{file_tag}_mlp_{eval_metric}_study.png")
    show()
# ---------- Best model performance ----------
    prd_trn: array = best_model.predict(trnX)
    prd_tst: array = best_model.predict(tstX)
    figure()
    plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels)
    savefig(f'images/{file_tag}_mlp_{params["name"]}_best_{params["metric"]}_eval.png')
    show()