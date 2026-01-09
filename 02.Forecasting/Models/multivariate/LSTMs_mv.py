from copy import deepcopy
from pandas import read_csv, Series, DataFrame
from torch import tensor, no_grad
from torch.nn import Module, LSTM, Linear, MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from dslabs_functions import (
    HEIGHT,
    subplots,
    plot_multiline_chart,
    plot_forecasting_eval,
    plot_forecasting_series,
    savefig,
    FORECAST_MEASURES,
    DELTA_IMPROVE,
)

# -------------------------
# Settings
# -------------------------
file_tag = "TrafficTwoMonth"
target = "Total"
timecol = "datetime"
measure = "R2"

train_file = "datasets/TrafficTwoMonth_train_prepared_mv.csv"
test_file  = "datasets/TrafficTwoMonth_test_prepared_mv.csv"

exog_vars = ["dow_sin", "dow_cos"]

# -------------------------
# Load data
# -------------------------
train_df = read_csv(train_file, index_col=timecol, parse_dates=True)
test_df  = read_csv(test_file, index_col=timecol, parse_dates=True)

train_series = train_df[target]
test_series  = test_df[target]

# -------------------------
# Dataset preparation
# -------------------------
def prepare_multivariate_dataset_for_lstm(df: DataFrame, target_col: str, exog_cols: list, seq_length: int = 4):
    values = df[[target_col] + exog_cols].to_numpy()
    target_values = df[target_col].to_numpy()
    
    X_list, Y_list = [], []
    for i in range(len(df) - seq_length):
        X_list.append(values[i : i + seq_length, :])
        Y_list.append(target_values[i + 1 : i + seq_length + 1])

    X = tensor(X_list).float()              # (N, seq_len, n_features)
    Y = tensor(Y_list).float().unsqueeze(-1)  # (N, seq_len, 1)
    return X, Y

# -------------------------
# Multivariate LSTM model
# -------------------------
class DS_MV_LSTM(Module):
    def __init__(self, df: DataFrame, target_col: str, exog_cols: list,
                 hidden_size: int = 50, num_layers: int = 1, seq_length: int = 4,
                 lr: float = 1e-3, batch_div: int = 10):
        super().__init__()
        self.seq_length = seq_length
        self.target_col = target_col
        self.exog_cols = exog_cols
        self.input_size = 1 + len(exog_cols)

        self.lstm = LSTM(input_size=self.input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = Linear(hidden_size, 1)

        self.optimizer = Adam(self.parameters(), lr=lr)
        self.loss_fn = MSELoss()

        trnX, trnY = prepare_multivariate_dataset_for_lstm(df, target_col, exog_cols, seq_length)
        batch_size = max(1, len(df) // batch_div)
        self.loader = DataLoader(TensorDataset(trnX, trnY), shuffle=True, batch_size=batch_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

    def fit_one_epoch(self):
        self.train()
        last_loss = None
        for batchX, batchY in self.loader:
            y_pred = self(batchX)
            loss = self.loss_fn(y_pred, batchY)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            last_loss = loss
        return last_loss

    def predict(self, X):
        self.eval()
        with no_grad():
            y_pred = self(X)
        return y_pred[:, -1, :]

# -------------------------
# LSTM study (sequence length & hidden units)
# -------------------------
def mv_lstm_study(train_df: DataFrame, test_df: DataFrame, exog_cols: list,
                  target_col: str, nr_episodes: int = 1000, measure: str = "R2"):

    sequence_sizes = [2, 4, 8]
    hidden_units_list = [25, 50, 100]
    step = nr_episodes // 10
    episodes = [1] + list(range(0, nr_episodes + 1, step))[1:]
    flag = measure in ["R2", "MAPE"]

    best_model = None
    best_params = {"name": "LSTM_MV", "metric": measure, "params": ()}
    best_perf = -1e10

    _, axs = subplots(1, len(sequence_sizes), figsize=(len(sequence_sizes) * HEIGHT, HEIGHT))

    for i, seq_len in enumerate(sequence_sizes):
        tstX, _ = prepare_multivariate_dataset_for_lstm(test_df, target_col, exog_cols, seq_len)
        values = {}

        for hidden in hidden_units_list:
            yvalues = []
            model = DS_MV_LSTM(train_df, target_col, exog_cols, hidden_size=hidden, seq_length=seq_len)

            for n in range(nr_episodes + 1):
                model.fit_one_epoch()

                if n % step == 0:
                    prd_tst = model.predict(tstX).numpy().ravel()
                    y_true = test_df[target_col][seq_len:]
                    eval_val = FORECAST_MEASURES[measure](y_true, prd_tst)
                    print(f"seq_len={seq_len} hidden={hidden} episode={n}: {eval_val:.4f}")

                    if eval_val > best_perf and abs(eval_val - best_perf) > DELTA_IMPROVE:
                        best_perf = eval_val
                        best_params["params"] = (seq_len, hidden, n)
                        best_model = deepcopy(model)

                    yvalues.append(eval_val)
            values[hidden] = yvalues

        plot_multiline_chart(
            episodes, values, ax=axs[i],
            title=f"MV LSTM seq_len={seq_len} ({measure})",
            xlabel="Episodes", ylabel=measure, percentage=flag
        )

    print(f"Best MV LSTM: seq_len={best_params['params'][0]}, hidden={best_params['params'][1]}, "
          f"episodes={best_params['params'][2]} => {measure}={best_perf:.4f}")
    return best_model, best_params

# -------------------------
# Run study
# -------------------------
best_model, best_params = mv_lstm_study(train_df, test_df, exog_vars, target, nr_episodes=3000, measure=measure)
savefig(f"images/{file_tag}_lstm_mv_{measure}_study.png")

seq_len, hidden, episodes = best_params["params"]

# Prepare final sequences
trnX, _ = prepare_multivariate_dataset_for_lstm(train_df, target, exog_vars, seq_len)
tstX, _ = prepare_multivariate_dataset_for_lstm(test_df, target, exog_vars, seq_len)

prd_trn = best_model.predict(trnX).numpy().ravel()
prd_tst = best_model.predict(tstX).numpy().ravel()

# Evaluation plot
plot_forecasting_eval(
    train_series[seq_len:], test_series[seq_len:],
    prd_trn, prd_tst,
    title=f"{file_tag} - MV LSTM (seq_len={seq_len}, hidden={hidden}, episodes={episodes})"
)
savefig(f"images/{file_tag}_lstm_mv_{measure}_eval.png")

# Forecasting series plot
pred_series = Series(prd_tst, index=test_df.index[seq_len:])
plot_forecasting_series(
    train_series[seq_len:], test_series[seq_len:], pred_series,
    title=f"{file_tag} - MV LSTM Forecast (seq_len={seq_len}, hidden={hidden}, episodes={episodes})",
    xlabel=timecol, ylabel=target
)
savefig(f"images/{file_tag}_lstm_mv_{measure}_forecast.png")
