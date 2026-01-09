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
    parse_dates=["datetime"],
)

series: Series = data[target]

granularities = ["15min", "H", "D", "W"]

for g in granularities:

    # Aggregate dataset
    agg_df: DataFrame = ts_aggregation_by(data, gran_level=g, agg_func="sum")
    
    train, test = series_train_test_split(agg_df, trn_pct=0.90)

    # Persistence Model
    fr_mod = PersistenceRealistRegressor()
    fr_mod.fit(train)
    prd_trn: Series = fr_mod.predict(train)
    prd_tst: Series = fr_mod.predict(test)

    print(f'[+] Plotting Persistent Models results from for Granularity = {g}...\n')
    plot_forecasting_eval(train, test, prd_trn, prd_tst, title=f"{file_tag} - Persistence Realist - Granularity={g}")
    savefig(f"images/{file_tag}_persistence_real_eval_gran{g}.png")
    plot_forecasting_series(
        train,
        test,
        prd_tst,
        title=f"{file_tag} - Persistence Realist Granularity={g}",
        xlabel=timecol,
        ylabel=target,
    )
    savefig(f"images/{file_tag}_persistence_real_forecast_gran{g}.png")

    # Linear Regression
    trnX = arange(len(train)).reshape(-1, 1)
    trnY = train.to_numpy()
    tstX = arange(len(train), len(agg_df)).reshape(-1, 1)
    tstY = test.to_numpy()

    model = LinearRegression()
    model.fit(trnX, trnY)

    prd_trn: Series = Series(model.predict(trnX), index=train.index)
    prd_tst: Series = Series(model.predict(tstX), index=test.index)

    print(f'[+] Plotting Linear Regression Models results from for Granularity = {g}...\n')

    plot_forecasting_eval(train, test, prd_trn, prd_tst, title=f"{file_tag} - Linear Regression - Granularity={g}")
    savefig(f"images/{file_tag}_linear_regression_eval_gran{g}.png")

    plot_forecasting_series(
        train,
        test,
        prd_tst,
        title=f"{file_tag} - Linear Regression - Granularity={g}",
        xlabel=timecol,
        ylabel=target,
    )
    savefig(f"images/{file_tag}_linear_regression_forecast_gran{g}.png")


