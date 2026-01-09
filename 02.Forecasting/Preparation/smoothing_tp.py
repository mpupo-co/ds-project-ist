import pandas as pd
from matplotlib.pyplot import figure, tight_layout, show
from dslabs_functions import *
from ft_dslabs_functions import *
from sklearn.linear_model import LinearRegression
from numpy import arange  # needed for arange

file_tag = "TrafficTwoMonth"
filename = "datasets/TrafficTwoMonth_clean.csv"
target = "Total"
timecol = "datetime"


data: DataFrame = read_csv(
    filename,
    index_col=timecol,
    parse_dates=True,
    infer_datetime_format=True,
)

series: Series = data[target]

# 15-min data → 4 per hour, 96 per day, 672 per week
sizes: list[int] = [
    1,
    int(60/15),          # 4  (hourly)
    int(60*24/15),       # 96 (daily)
    int(60*24*7/15),     # 672 (weekly)
]

# Train/test split on the target series
train_full, test = series_train_test_split(data, trn_pct=0.90)

# Loop over smoothing windows
for i, window in enumerate(sizes):

    print(f"\n[+] Using smoothing window = {window} observations\n")

    # Smoothing only on training set
    train: Series = train_full.copy()

    # Rolling mean smoothing
    smoothed_train: Series = train.rolling(window=window).mean()

    # Drop initial NaNs due to rolling window
    smoothed_train = smoothed_train.dropna()

    # Persistence Model
    fr_mod = PersistenceRealistRegressor()
    fr_mod.fit(smoothed_train)

    prd_trn_pers: Series = fr_mod.predict(smoothed_train)
    prd_tst_pers: Series = fr_mod.predict(test)

    print(f'[+] Plotting Persistence Model results for Smoothing = {window}...\n')

    plot_forecasting_eval(
        smoothed_train,
        test,
        prd_trn_pers,
        prd_tst_pers,
        title=f"{file_tag} - Persistence Realist - Smoothing={window}"
    )
    savefig(f"images/{file_tag}_persistence_real_eval_smoo{window}.png")

    plot_forecasting_series(
        smoothed_train,
        test,
        prd_tst_pers,
        title=f"{file_tag} - Persistence Realist Smoothing={window}",
        xlabel=timecol,
        ylabel=target,
    )
    savefig(f"images/{file_tag}_persistence_real_forecast_smoo{window}.png")

    # Linear Regression
    trnY = smoothed_train.to_numpy()
    trnX = arange(len(trnY)).reshape(-1, 1)

    # X for test continues after train indices
    tstX = arange(len(trnY), len(trnY) + len(test)).reshape(-1, 1)
    tstY = test.to_numpy()

    model = LinearRegression()
    model.fit(trnX, trnY)

    prd_trn_lr: Series = Series(model.predict(trnX), index=smoothed_train.index)
    prd_tst_lr: Series = Series(model.predict(tstX), index=test.index)

    print(f'[+] Plotting Linear Regression results for Smoothing = {window}...\n')

    plot_forecasting_eval(
        smoothed_train,
        test,
        prd_trn_lr,
        prd_tst_lr,
        title=f"{file_tag} - Linear Regression - Smoothing={window}"
    )
    savefig(f"images/{file_tag}_linear_regression_eval_smoo{window}.png")

    plot_forecasting_series(
        smoothed_train,
        test,
        prd_tst_lr,
        title=f"{file_tag} - Linear Regression - Smoothing={window}",
        xlabel=timecol,
        ylabel=target,
    )
    savefig(f"images/{file_tag}_linear_regression_forecast_smoo{window}.png")
