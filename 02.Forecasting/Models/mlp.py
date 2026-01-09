# ============================================================
# Forecasting with MLP + Hyperparameter Study (clean script)
# ============================================================
from pandas import read_csv, DataFrame, Series
import numpy as np

from torch import tensor, no_grad
from torch.nn import Linear, Module, MSELoss, Sequential, ReLU
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from matplotlib.pyplot import figure
from dslabs_functions import *

#  -------------------------------------------------
# Settings
#  -------------------------------------------------
file_tag = "TrafficTwoMonth"
target = "Total"
timecol = "datetime"

train_file = "datasets/TrafficTwoMonth_train_prepared.csv"
test_file  = "datasets/TrafficTwoMonth_test_prepared.csv"

# --- default (used if you skip study) ---
SEQ_LENGTH = 8
HIDDEN_UNITS = 50
NR_EPOCHS = 300
LR = 1e-3

# --- study settings ---
DO_STUDY = True
STUDY_METRIC = "rmse"     # "rmse" or "mae"
VAL_RATIO = 0.2
LAG = 50
NR_MAX_EPOCHS = 300

SEQ_LENGTHS = [4, 8, 12, 16]
HIDDEN_UNITS_LIST = [20, 50, 100]
LEARNING_RATES = [1e-2, 1e-3, 5e-4]

BATCH_FRAC = 0.1          # batch_size = max(1, int(len(trnX)*BATCH_FRAC))

#  -------------------------------------------------
# Load prepared data
#  -------------------------------------------------
train_df: DataFrame = read_csv(train_file, index_col=timecol, parse_dates=True)
test_df: DataFrame  = read_csv(test_file,  index_col=timecol, parse_dates=True)

train: Series = train_df[target]
test: Series  = test_df[target]

#  -------------------------------------------------
# Dataset preparation
#  -------------------------------------------------
def prepare_dataset(series: Series, seq_length: int):
    X, y = [], []
    values = series.to_numpy(dtype=np.float32)

    for i in range(len(values) - seq_length):
        X.append(values[i : i + seq_length])
        y.append(values[i + seq_length])

    X = tensor(np.array(X)).float()                 # (N, seq_length)
    y = tensor(np.array(y)).float().unsqueeze(1)    # (N, 1)
    return X, y

def temporal_train_val_split(series: Series, val_ratio: float = 0.2):
    n = len(series)
    cut = int(n * (1 - val_ratio))
    return series.iloc[:cut], series.iloc[cut:]

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(np.mean(np.abs(y_true - y_pred)))

FORECAST_METRICS = {"rmse": rmse, "mae": mae}

#  -------------------------------------------------
# MLP model
#  -------------------------------------------------
class DS_MLP(Module):
    def __init__(self, input_size: int, hidden_size: int, lr: float):
        super().__init__()
        self.net = Sequential(
            Linear(input_size, hidden_size),
            ReLU(),
            Linear(hidden_size, 1),
        )
        self.loss_fn = MSELoss()
        self.optimizer = Adam(self.parameters(), lr=lr)

    def forward(self, x):
        return self.net(x)

    def fit_epoch(self, loader):
        self.train()
        last_loss = None
        for X, y in loader:
            y_pred = self(X)
            loss = self.loss_fn(y_pred, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            last_loss = loss
        return float(last_loss.item()) if last_loss is not None else float("nan")

    def predict(self, X):
        self.eval()
        with no_grad():
            return self(X)

#  -------------------------------------------------
# Study function (like your classification study, but forecasting)
#  -------------------------------------------------
def mlp_forecasting_study(
    train_series: Series,
    val_ratio: float = 0.2,
    metric: str = "rmse",
    seq_lengths=None,
    hidden_units=None,
    learning_rates=None,
    nr_max_epochs: int = 300,
    lag: int = 50,
    batch_frac: float = 0.1,
):
    if seq_lengths is None:
        seq_lengths = [8]
    if hidden_units is None:
        hidden_units = [50]
    if learning_rates is None:
        learning_rates = [1e-3]

    assert metric in FORECAST_METRICS, f"metric must be one of {list(FORECAST_METRICS.keys())}"
    eval_fn = FORECAST_METRICS[metric]

    epoch_steps = [lag] + [i for i in range(2 * lag, nr_max_epochs + 1, lag)]
    trn_s, val_s = temporal_train_val_split(train_series, val_ratio=val_ratio)

    best_model = None
    best_params = {"name": "MLP", "metric": metric, "params": None}
    best_score = float("inf")  # lower is better

    fig = figure()
    _, axs = __import__("matplotlib.pyplot").pyplot.subplots(
        1, len(seq_lengths), figsize=(len(seq_lengths) * 5, 4), squeeze=False
    )

    for ax_i, seq in enumerate(seq_lengths):
        values_for_plot = {}

        trnX, trnY = prepare_dataset(trn_s, seq)
        valX, valY = prepare_dataset(val_s, seq)

        bs = max(1, int(len(trnX) * batch_frac))
        loader = DataLoader(TensorDataset(trnX, trnY), batch_size=bs, shuffle=True)

        for h in hidden_units:
            for lr in learning_rates:
                label = f"h={h}, lr={lr:g}"
                y_vals = []

                model = DS_MLP(input_size=seq, hidden_size=h, lr=lr)

                for e in range(1, nr_max_epochs + 1):
                    model.fit_epoch(loader)

                    if e in epoch_steps:
                        prd_val = model.predict(valX).numpy().ravel()
                        score = eval_fn(valY.numpy().ravel(), prd_val)
                        y_vals.append(score)

                        if score < best_score:
                            best_score = score
                            best_params["params"] = (seq, h, lr, e)
                            best_model = model

                values_for_plot[label] = y_vals

        plot_multiline_chart(
            epoch_steps,
            values_for_plot,
            ax=axs[0, ax_i],
            title=f"SEQ_LENGTH={seq}",
            xlabel="epochs",
            ylabel=metric,
            percentage=False,
        )

    seq, h, lr, e = best_params["params"]
    print(f"Best MLP ({metric}) = {best_score:.6f} with (seq={seq}, hidden={h}, lr={lr:g}, epochs={e})")
    return best_params

#  -------------------------------------------------
# Run study 
#  -------------------------------------------------
if DO_STUDY:
    params = mlp_forecasting_study(
        train_series=train,
        val_ratio=VAL_RATIO,
        metric=STUDY_METRIC,
        seq_lengths=SEQ_LENGTHS,
        hidden_units=HIDDEN_UNITS_LIST,
        learning_rates=LEARNING_RATES,
        nr_max_epochs=NR_MAX_EPOCHS,
        lag=LAG,
        batch_frac=BATCH_FRAC,
    )
    savefig(f"images/{file_tag}_mlp_{STUDY_METRIC}_study.png")

    SEQ_LENGTH, HIDDEN_UNITS, LR, NR_EPOCHS = params["params"]

#  -------------------------------------------------
# Train final model on FULL TRAIN (with best or default params)
#  -------------------------------------------------
trnX, trnY = prepare_dataset(train, SEQ_LENGTH)
tstX, tstY = prepare_dataset(test, SEQ_LENGTH)

train_loader = DataLoader(
    TensorDataset(trnX, trnY),
    batch_size=max(1, len(trnX) // 10),
    shuffle=True,
)

model = DS_MLP(input_size=SEQ_LENGTH, hidden_size=HIDDEN_UNITS, lr=LR)

for epoch in range(NR_EPOCHS):
    loss = model.fit_epoch(train_loader)
    if epoch % 50 == 0:
        print(f"Epoch {epoch} | Loss = {loss:.6f}")

#  -------------------------------------------------
# Predictions
#  -------------------------------------------------
prd_trn = model.predict(trnX).numpy().ravel()
prd_tst = model.predict(tstX).numpy().ravel()

train_eval = train[SEQ_LENGTH:]
test_eval  = test[SEQ_LENGTH:]

#  -------------------------------------------------
# Evaluation plots
#  -------------------------------------------------
plot_forecasting_eval(
    train_eval,
    test_eval,
    prd_trn,
    prd_tst,
    title=f"{file_tag} - MLP (seq={SEQ_LENGTH}, hidden={HIDDEN_UNITS}, lr={LR:g}, epochs={NR_EPOCHS})",
)
savefig(f"images/{file_tag}_mlp_eval.png")

pred_series = Series(prd_tst, index=test_eval.index)
plot_forecasting_series(
    train_eval,
    test_eval,
    pred_series,
    title=f"{file_tag} - MLP Forecast",
    xlabel=timecol,
    ylabel=target,
)
savefig(f"images/{file_tag}_mlp_forecast.png")
