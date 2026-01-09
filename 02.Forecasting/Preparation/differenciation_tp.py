import pandas as pd
from matplotlib.pyplot import figure, tight_layout, show
from dslabs_functions import *
from ft_dslabs_functions import *
from sklearn.linear_model import LinearRegression

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

# 15-min frequency 
HOUR_STEPS = 60/15
DAY_STEPS = 60*24/15
WEEK_STEPS = 60*24*7/15

diff_settings = [
    ("lag1", 1),                  # 15-min difference
    ("hourly", HOUR_STEPS),       # 1-hour difference
    ("daily", DAY_STEPS),         # 1-day difference
    ("weekly", WEEK_STEPS),       # 1-week difference
]

for name, periods in diff_settings:
    diff_df = data.copy()
    diff_df[target] = data[target].diff(periods=periods)

    train, test = series_train_test_split(diff_df, trn_pct=0.90)

    # Persistence Model
    fr_mod = PersistenceRealistRegressor()
    fr_mod.fit(train)
    prd_trn: Series = fr_mod.predict(train)
    prd_tst: Series = fr_mod.predict(test)

    print(f'[+] Plotting Persistent Models results from for Diff = {name}...\n')
    plot_forecasting_eval(train, test, prd_trn, prd_tst, title=f"{file_tag} - Persistence Realist - Diff={name}")
    savefig(f"images/{file_tag}_persistence_real_eval_diff{name}.png")
    plot_forecasting_series(
        train,
        test,
        prd_tst,
        title=f"{file_tag} - Persistence Realist Diff={name}",
        xlabel=timecol,
        ylabel=target,
    )
    savefig(f"images/{file_tag}_persistence_real_forecast_diff{name}.png")

    # Linear Regression
    trnX = arange(len(train)).reshape(-1, 1)
    trnY = train.to_numpy()
    tstX = arange(len(train), len(diff_df)).reshape(-1, 1)
    tstY = test.to_numpy()

    model = LinearRegression()
    model.fit(trnX, trnY)

    prd_trn: Series = Series(model.predict(trnX), index=train.index)
    prd_tst: Series = Series(model.predict(tstX), index=test.index)

    print(f'[+] Plotting Linear Regression Models results from for Diff = {name}...\n')

    plot_forecasting_eval(train, test, prd_trn, prd_tst, title=f"{file_tag} - Linear Regression - Diff={name}")
    savefig(f"images/{file_tag}_linear_regression_eval_diff{name}.png")

    plot_forecasting_series(
        train,
        test,
        prd_tst,
        title=f"{file_tag} - Linear Regression - Diff={name}",
        xlabel=timecol,
        ylabel=target,
    )
    savefig(f"images/{file_tag}_linear_regression_forecast_diff{name}.png")


