from pandas import read_csv, DataFrame, Series
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from dslabs_functions import *

# -----------------------------------------------
# Settings
# -----------------------------------------------
file_tag: str = "TrafficTwoMonth"
target: str = "Total"
timecol: str = "datetime"
measure: str = "R2"

train_file = "datasets/TrafficTwoMonth_train_prepared.csv"
test_file  = "datasets/TrafficTwoMonth_test_prepared.csv"

# -----------------------------------------------
# Load prepared datasets 
# -----------------------------------------------
train_df: DataFrame = read_csv(train_file, index_col=timecol, parse_dates=True)
test_df: DataFrame  = read_csv(test_file,  index_col=timecol, parse_dates=True)

train: Series = train_df[target]
test: Series  = test_df[target]

# -----------------------------------------------
# Study function 
# -----------------------------------------------
def exponential_smoothing_study(train: Series, test: Series, measure: str = "R2"):
    alpha_values = [i / 10 for i in range(1, 10)]
    flag = measure == "R2" or measure == "MAPE"

    best_model = None
    best_params: dict = {"name": "Exponential Smoothing", "metric": measure, "params": ()}
    best_performance: float = -1e12

    yvalues = []
    for alpha in alpha_values:
        tool = SimpleExpSmoothing(train)
        model = tool.fit(smoothing_level=alpha, optimized=False)
        prd_tst = model.forecast(steps=len(test))

        eval_val: float = FORECAST_MEASURES[measure](test, prd_tst)
        if eval_val > best_performance and abs(eval_val - best_performance) > DELTA_IMPROVE:
            best_performance = eval_val
            best_params["params"] = (alpha,)
            best_model = model

        yvalues.append(eval_val)

    print(f"Exponential Smoothing best with alpha={best_params['params'][0]} -> {measure}={best_performance:.4f}")
    plot_line_chart(
        alpha_values,
        yvalues,
        title=f"Exponential Smoothing ({measure})",
        xlabel="alpha",
        ylabel=measure,
        percentage=flag,
    )
    return best_model, best_params

best_model, best_params = exponential_smoothing_study(train, test, measure=measure)
savefig(f"images/{file_tag}_exponential_smoothing_{measure}_study.png")

alpha = best_params["params"][0]
prd_trn = best_model.predict(start=0, end=len(train) - 1)
prd_tst = best_model.forecast(steps=len(test))

plot_forecasting_eval(train, test, prd_trn, prd_tst, title=f"{file_tag} - Exponential Smoothing alpha={alpha}")
savefig(f"images/{file_tag}_exponential_smoothing_{measure}_eval.png")

plot_forecasting_series(train, test, prd_tst, title=f"{file_tag} - Exponential Smoothing", xlabel=timecol, ylabel=target)
savefig(f"images/{file_tag}_exponential_smoothing_{measure}_forecast.png")
