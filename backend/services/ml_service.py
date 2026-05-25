"""
ML service — all PyTorch/sklearn computation lives here, called by Celery workers.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

FEATURES = ["interest_rate", "oil_price", "gdp_growth",
            "unemployment_rate", "food_price_index", "money_supply_m2"]


def _normalize(X):
    mn, mx = X.min(0), X.max(0)
    return (X - mn) / (mx - mn + 1e-8), mn, mx


def _build_nn(in_dim: int):
    class _Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(in_dim, 64), torch.nn.ReLU(), torch.nn.Dropout(0.1),
                torch.nn.Linear(64, 32),     torch.nn.ReLU(),
                torch.nn.Linear(32, 1),
            )
        def forward(self, x): return self.net(x)
    return _Net()


def train_model(records: list[dict], epochs: int = 40) -> dict:
    """Train NN or Ridge on records. Returns metrics + predictions."""
    df    = pd.DataFrame(records)
    avail = [f for f in FEATURES if f in df.columns]
    sub   = df[avail + ["inflation_rate"]].dropna()
    if len(sub) < 20:
        return {"error": "Not enough data"}

    X = sub[avail].values.astype(float)
    y = sub["inflation_rate"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    X_norm, mn, mx = _normalize(X_tr)
    X_te_n = (X_te - mn) / (mx - mn + 1e-8)

    if TORCH_OK:
        X_t = torch.tensor(X_norm, dtype=torch.float32)
        y_t = torch.tensor(y_tr.reshape(-1,1), dtype=torch.float32)
        model   = _build_nn(len(avail))
        opt     = optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()
        losses  = []
        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            loss = loss_fn(model(X_t), y_t)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(X_te_n, dtype=torch.float32)).numpy().flatten()
        engine = "pytorch"
    else:
        model  = Ridge(alpha=1.0).fit(X_norm, y_tr)
        preds  = model.predict(X_te_n)
        losses = [float(mean_squared_error(y_tr, model.predict(X_norm)))]
        engine = "sklearn_ridge"

    mse = float(mean_squared_error(y_te, preds))
    r2  = float(r2_score(y_te, preds))
    return {
        "engine":   engine,
        "mse":      mse,
        "r2":       r2,
        "losses":   losses,
        "features": avail,
        "actual":   y_te.tolist(),
        "preds":    preds.tolist(),
    }


def run_lstm_forecast(records: list[dict], country: str,
                      forecast_years: int = 5, epochs: int = 60) -> dict:
    """LSTM forecast for a single country."""
    df     = pd.DataFrame(records)
    sub    = df[df["country"] == country].sort_values("year")
    series = sub["inflation_rate"].dropna().values.astype(np.float32)
    if len(series) < 5:
        return {"error": "Not enough data"}

    seq_len = 3
    mn, mx  = series.min(), series.max()
    norm    = (series - mn) / (mx - mn + 1e-8)

    def make_seq(s, sl):
        X, y = [], []
        for i in range(len(s) - sl):
            X.append(s[i:i+sl]); y.append(s[i+sl])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    if TORCH_OK and len(series) >= seq_len + 2:
        X, y = make_seq(norm, seq_len)
        X_t  = torch.from_numpy(X[:,None,:].transpose(0,2,1))  # (N,1,seq) → wrong
        # correct shape: (N, seq_len, 1)
        X_t  = torch.from_numpy(X[:,:,None])
        y_t  = torch.from_numpy(y[:,None])

        class _LSTM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = torch.nn.LSTM(1, 64, 2, batch_first=True, dropout=0.2)
                self.fc   = torch.nn.Linear(64, 1)
            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:,-1,:])

        model   = _LSTM()
        opt     = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = torch.nn.MSELoss()
        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            loss = loss_fn(model(X_t), y_t)
            loss.backward()
            opt.step()

        window = list(norm[-seq_len:].astype(float))
        future_norm = []
        model.eval()
        with torch.no_grad():
            for _ in range(forecast_years):
                seq  = np.array(window[-seq_len:], dtype=np.float32)
                inp  = torch.from_numpy(seq[None,:,None])
                pred = float(model(inp).item())
                future_norm.append(pred)
                window.append(pred)
        future = (np.array(future_norm) * (mx - mn) + mn).tolist()
        engine = "lstm"
    else:
        trend  = np.polyfit(range(len(series)), series, 1)
        future = [float(np.polyval(trend, len(series)+i)) for i in range(forecast_years)]
        engine = "linear"

    last_year = int(sub["year"].max())
    return {
        "engine":       engine,
        "country":      country,
        "historical":   series.tolist(),
        "hist_years":   sub["year"].values.tolist(),
        "forecast":     future,
        "future_years": list(range(last_year+1, last_year+forecast_years+1)),
        "std":          float(np.std(series)),
    }


def run_monte_carlo(base_return: float, volatility: float,
                    horizon: int = 10, simulations: int = 300) -> dict:
    """Monte Carlo simulation — returns paths as list of lists."""
    rng   = np.random.default_rng(42)
    paths = []
    for _ in range(simulations):
        r      = rng.normal(base_return/100, volatility/100, horizon)
        wealth = np.cumprod(1 + r).tolist()
        paths.append(wealth)
    arr    = np.array(paths)
    median = np.median(arr, axis=0).tolist()
    p5     = np.percentile(arr[:,-1], 5)
    p95    = np.percentile(arr[:,-1], 95)
    return {"paths": paths, "median": median,
            "p5": float(p5), "p95": float(p95),
            "median_final": float(np.median(arr[:,-1]))}
