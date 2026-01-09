from pandas import read_csv, DataFrame, Series
from statsmodels.tsa.arima.model import ARIMA
from dslabs_functions import *

# ----------------------------------------------------
# Settings
# ----------------------------------------------------
train_file: str = "datasets/TrafficTwoMonth_train_prepared_mv.csv"
test_file: str  = "datasets/TrafficTwoMonth_test_prepared_mv.csv"

file_tag: str = "TrafficTwoMonth"
target: str = "Total"
timecol: str = "datetime"
measure: str = "R2"

# ----------------------------------------------------
# Load prepared train/test
# ----------------------------------------------------
train_df: DataFrame = read_csv(train_file, index_col=timecol, parse_dates=True)
test_df:  DataFrame = read_csv(test_file,  index_col=timecol, parse_dates=True)

# ---- Safe exogenous variables (NO leakage) ----
safe_exog = [
    "dow_sin",
    "dow_cos"
]

X_train: DataFrame = train_df[safe_exog]
X_test:  DataFrame = test_df[safe_exog]

y_train: Series = train_df[target]
y_test:  Series = test_df[target]

# ----------------------------------------------------
# ARIMAX study function (same structure as ARIMA)
# ----------------------------------------------------
def arimax_study(
    train: Series,
    test: Series,
    X_train: DataFrame,
    X_test: DataFrame,
    measure: str = "R2",
):

    d_values = (0, 1, 2)
    p_params = (1, 2, 3, 5, 7, 10)
    q_params = (1, 3, 5, 7)

    flag = measure in ("R2", "MAPE")
    best_model = None
    best_params = {"name": "ARIMAX", "metric": measure, "params": ()}
    best_performance: float = -1e9

    fig, axs = subplots(1, len(d_values), figsize=(len(d_values) * HEIGHT, HEIGHT))

    for i, d in enumerate(d_values):
        values = {}

        for q in q_params:
            yvalues = []
            for p in p_params:
                try:
                    model = ARIMA(
                        train,
                        exog=X_train,
                        order=(p, d, q),
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit()

                    prd_tst = model.forecast(
                        steps=len(test),
                        exog=X_test,
                    )

                    eval_val = FORECAST_MEASURES[measure](test, prd_tst)

                    if eval_val > best_performance + DELTA_IMPROVE:
                        best_performance = eval_val
                        best_params["params"] = (p, d, q)
                        best_model = model

                    yvalues.append(eval_val)

                except Exception:
                    yvalues.append(float("nan"))

            values[q] = yvalues

        plot_multiline_chart(
            p_params,
            values,
            ax=axs[i],
            title=f"MV ARIMA d={d} ({measure})",
            xlabel="p",
            ylabel=measure,
            percentage=flag,
        )

    print(
        f"MV ARIMA best results achieved with (p,d,q)=("
        f"{best_params['params'][0]}, "
        f"{best_params['params'][1]}, "
        f"{best_params['params'][2]}) "
        f"==> {measure}={best_performance:.2f}"
    )

    return best_model, best_params


# ----------------------------------------------------
# Run study + save plots
# ----------------------------------------------------
best_model, best_params = arimax_study(
    y_train, y_test, X_train, X_test, measure=measure
)
savefig(f"images/{file_tag}_arimax_{measure}_study.png")

params = best_params["params"]

# ----------------------------------------------------
# In-sample prediction & test forecast
# ----------------------------------------------------
prd_trn = best_model.predict(
    start=0,
    end=len(y_train) - 1,
    exog=X_train,
)

prd_tst = best_model.forecast(
    steps=len(y_test),
    exog=X_test,
)

# ----------------------------------------------------
# Evaluation plots (identical to ARIMA)
# ----------------------------------------------------
plot_forecasting_eval(
    y_train,
    y_test,
    prd_trn,
    prd_tst,
    title=f"{file_tag} - MV ARIMA (p={params[0]}, d={params[1]}, q={params[2]})",
)
savefig(f"images/{file_tag}_arimax_{measure}_eval.png")

plot_forecasting_series(
    y_train,
    y_test,
    prd_tst,
    title=f"{file_tag} - MV ARIMA Forecast (p={params[0]}, d={params[1]}, q={params[2]})",
    xlabel=timecol,
    ylabel=target,
)
savefig(f"images/{file_tag}_arimax_{measure}_forecast.png")
