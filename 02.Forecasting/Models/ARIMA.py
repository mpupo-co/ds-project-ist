from pandas import read_csv, DataFrame, Series
from statsmodels.tsa.arima.model import ARIMA
from dslabs_functions import *

#  ----------------------------------------------------
# Settings
#  ----------------------------------------------------
train_file: str = "datasets/TrafficTwoMonth_train_prepared.csv"
test_file: str  = "datasets/TrafficTwoMonth_test_prepared.csv"

file_tag: str = "TrafficTwoMonth"
target: str = "Total"
timecol: str = "datetime"
measure: str = "R2"

#  ----------------------------------------------------
# Load already-prepared train/test 
#  ----------------------------------------------------
train_df: DataFrame = read_csv(train_file, index_col=timecol, parse_dates=True)
test_df: DataFrame = read_csv(test_file, index_col=timecol, parse_dates=True)

train: Series = train_df[target]
test: Series = test_df[target]

#  ----------------------------------------------------
# Example model + diagnostics
#  ----------------------------------------------------
# This block is optional; it is useful to show one fitted model summary/diagnostics.

predictor = ARIMA(train, order=(3, 1, 2))
model = predictor.fit()
print(model.summary())
model.plot_diagnostics(figsize=(2 * HEIGHT, 1.5 * HEIGHT))
savefig(f"images/{file_tag}_arima_diagnostics.png")

# ----------------------------------------------------
# Study function 
# ----------------------------------------------------
def arima_study(train: Series, test: Series, measure: str = "R2"):
   
    d_values = (0, 1, 2)
    p_params = (1, 2, 3, 5, 7, 10)
    q_params = (1, 3, 5, 7)

    flag = measure == "R2" or measure == "MAPE"
    best_model = None
    best_params: dict = {"name": "ARIMA", "metric": measure, "params": ()}
    best_performance: float = -100000

    fig, axs = subplots(1, len(d_values), figsize=(len(d_values) * HEIGHT, HEIGHT))
    for i in range(len(d_values)):
        d: int = d_values[i]
        values = {}

        for q in q_params:
            yvalues = []
            for p in p_params:
                # Fit ARIMA(p,d,q)
                arima = ARIMA(train, order=(p, d, q))
                model = arima.fit()

                prd_tst = model.forecast(steps=len(test), signal_only=False)

                # Evaluate
                eval_val: float = FORECAST_MEASURES[measure](test, prd_tst)

                # Track best
                if eval_val > best_performance and abs(eval_val - best_performance) > DELTA_IMPROVE:
                    best_performance = eval_val
                    best_params["params"] = (p, d, q)
                    best_model = model

                yvalues.append(eval_val)

            values[q] = yvalues

        # Plot multiline per d
        plot_multiline_chart(
            p_params,
            values,
            ax=axs[i],
            title=f"ARIMA d={d} ({measure})",
            xlabel="p",
            ylabel=measure,
            percentage=flag,
        )

    print(
        f"ARIMA best results achieved with (p,d,q)=("
        f"{best_params['params'][0]:.0f}, {best_params['params'][1]:.0f}, {best_params['params'][2]:.0f}"
        f") ==> measure={best_performance:.2f}"
    )

    return best_model, best_params


#  ----------------------------------------------------
# Run study + save plots 
#  ----------------------------------------------------
best_model, best_params = arima_study(train, test, measure=measure)
savefig(f"images/{file_tag}_arima_{measure}_study.png")

params = best_params["params"]

# In-sample prediction & test forecast
prd_trn = best_model.predict(start=0, end=len(train) - 1)
prd_tst = best_model.forecast(steps=len(test))

# Evaluation plots
plot_forecasting_eval(
    train,
    test,
    prd_trn,
    prd_tst,
    title=f"{file_tag} - ARIMA (p={params[0]}, d={params[1]}, q={params[2]})",
)
savefig(f"images/{file_tag}_arima_{measure}_eval.png")

plot_forecasting_series(
    train,
    test,
    prd_tst,
    title=f"{file_tag} - ARIMA",
    xlabel=timecol,
    ylabel=target,
)
savefig(f"images/{file_tag}_arima_{measure}_forecast.png")
