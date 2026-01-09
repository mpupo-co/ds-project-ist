from copy import deepcopy
from pandas import read_csv, DataFrame, Series
from torch import no_grad, tensor
from torch.nn import LSTM, Linear, Module, MSELoss
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

#  -------------------------------------------------
# Settings
#  -------------------------------------------------
file_tag: str = "TrafficTwoMonth"
target: str = "Total"
timecol: str = "datetime"
measure: str = "R2"

train_file: str = "datasets/TrafficTwoMonth_train_prepared.csv"
test_file: str  = "datasets/TrafficTwoMonth_test_prepared.csv"

#  -------------------------------------------------
# Load already-prepared data 
#  -------------------------------------------------
train_df: DataFrame = read_csv(train_file, index_col=timecol, parse_dates=True)
test_df: DataFrame  = read_csv(test_file,  index_col=timecol, parse_dates=True)

train: Series = train_df[target]
test: Series  = test_df[target]


#  -------------------------------------------------
# Dataset preparation for LSTM 
#  -------------------------------------------------
def prepare_dataset_for_lstm(series, seq_length: int = 4):
    """
    Teacher-style: build sequences where:
      X = past window of length seq_length
      Y = next window shifted by 1 (sequence-to-sequence)
    The model then returns the last step prediction (one-step ahead at the end of the window).
    """
    values = series.to_numpy() if hasattr(series, "to_numpy") else series
    setX: list = []
    setY: list = []
    for i in range(len(values) - seq_length):
        past = values[i : i + seq_length]
        future = values[i + 1 : i + seq_length + 1]
        setX.append(past)
        setY.append(future)

    # Ensure float tensors (important for torch training stability)
    X = tensor(setX).float().unsqueeze(-1)  # (N, seq_len, 1)
    Y = tensor(setY).float().unsqueeze(-1)  # (N, seq_len, 1)
    return X, Y


#  -------------------------------------------------
# LSTM model class 
#  -------------------------------------------------
class DS_LSTM(Module):
    def __init__(
        self,
        train_series: Series,
        input_size: int = 1,
        hidden_size: int = 50,
        num_layers: int = 1,
        length: int = 4,
        lr: float = 1e-3,
        batch_div: int = 10,
    ):
        super().__init__()
        self.length = length
        self.lstm = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = Linear(hidden_size, 1)

        self.optimizer = Adam(self.parameters(), lr=lr)
        self.loss_fn = MSELoss()

        trnX, trnY = prepare_dataset_for_lstm(train_series, seq_length=length)

        # Teacher-style batch size: len(train)//10 (guard against too small)
        batch_size = max(1, len(train_series) // batch_div)
        self.loader = DataLoader(TensorDataset(trnX, trnY), shuffle=True, batch_size=batch_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

    def fit_one_epoch(self):
        """
        One training epoch over the loader.
        Teacher code calls fit() repeatedly and treats that as 'episodes'.
        """
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
        """
        Returns the prediction of the last time step in the output sequence,
        consistent with teacher code: y_pred[:, -1, :].
        """
        self.eval()
        with no_grad():
            y_pred = self(X)
        return y_pred[:, -1, :]  # (N, 1)


# Quick sanity run (optional like teacher)
model = DS_LSTM(train, input_size=1, hidden_size=50, num_layers=1, length=4)
loss = model.fit_one_epoch()
print("Sanity loss:", loss)


#  -------------------------------------------------
# Study function 
#  -------------------------------------------------
def lstm_study(train: Series, test: Series, nr_episodes: int = 1000, measure: str = "R2"):
    sequence_size = [2, 4, 8]
    nr_hidden_units = [25, 50, 100]

    step: int = nr_episodes // 10
    episodes = [1] + list(range(0, nr_episodes + 1, step))[1:]
    flag = measure == "R2" or measure == "MAPE"

    best_model = None
    best_params: dict = {"name": "LSTM", "metric": measure, "params": ()}
    best_performance: float = -100000

    _, axs = subplots(1, len(sequence_size), figsize=(len(sequence_size) * HEIGHT, HEIGHT))

    for i in range(len(sequence_size)):
        length = sequence_size[i]

        # Prepare test windows once per length
        tstX, _ = prepare_dataset_for_lstm(test, seq_length=length)

        values = {}
        for hidden in nr_hidden_units:
            yvalues = []

            # pass length=length so the model matches the tested sequence size
            model = DS_LSTM(train, hidden_size=hidden, length=length)

            for n in range(0, nr_episodes + 1):
                model.fit_one_epoch()

                if n % step == 0:
                    prd_tst = model.predict(tstX)  # tensor shape (N, 1)

                    # Align ground truth with predictions:
                    # predictions correspond to test[length:]
                    y_true = test[length:]
                    y_hat = prd_tst.numpy().ravel()

                    eval_val: float = FORECAST_MEASURES[measure](y_true, y_hat)
                    print(f"seq length={length} hidden_units={hidden} nr_episodes={n}", eval_val)

                    if eval_val > best_performance and abs(eval_val - best_performance) > DELTA_IMPROVE:
                        best_performance = eval_val
                        best_params["params"] = (length, hidden, n)
                        best_model = deepcopy(model)

                    yvalues.append(eval_val)

            values[hidden] = yvalues

        plot_multiline_chart(
            episodes,
            values,
            ax=axs[i],
            title=f"LSTM seq length={length} ({measure})",
            xlabel="nr episodes",
            ylabel=measure,
            percentage=flag,
        )

    print(
        f"LSTM best results achieved with length={best_params['params'][0]} "
        f"hidden_units={best_params['params'][1]} and nr_episodes={best_params['params'][2]} "
        f"==> measure={best_performance:.2f}"
    )
    return best_model, best_params


#  -------------------------------------------------
# Run study + final evaluation/plots 
#  -------------------------------------------------
best_model, best_params = lstm_study(train, test, nr_episodes=3000, measure=measure)
savefig(f"images/{file_tag}_lstms_{measure}_study.png")

params = best_params["params"]
best_length = params[0]

# Prepare full train/test windows for final plots
trnX, _ = prepare_dataset_for_lstm(train, seq_length=best_length)
tstX, _ = prepare_dataset_for_lstm(test, seq_length=best_length)

prd_trn = best_model.predict(trnX).numpy().ravel()
prd_tst = best_model.predict(tstX).numpy().ravel()

# Evaluation plot: use aligned slices train[best_length:], test[best_length:]
plot_forecasting_eval(
    train[best_length:],
    test[best_length:],
    prd_trn,
    prd_tst,
    title=f"{file_tag} - LSTM (length={best_length}, hidden={params[1]}, epochs={params[2]})",
)
savefig(f"images/{file_tag}_lstms_{measure}_eval.png")

# Forecasting series plot: build Series for predicted test values with correct index
pred_series: Series = Series(prd_tst, index=test.index[best_length:])

plot_forecasting_series(
    train[best_length:],
    test[best_length:],
    pred_series,
    title=f"{file_tag} - LSTMs",
    xlabel=timecol,
    ylabel=target,
)
savefig(f"images/{file_tag}_lstms_{measure}_forecast.png")
