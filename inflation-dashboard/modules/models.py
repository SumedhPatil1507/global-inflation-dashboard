"""
Neural network — inflation predictor.
Train/test split, model persistence, confidence scoring.
Works with or without torch (sklearn Ridge fallback).
"""
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
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
    nn = None  # prevents NameError in type hints

FEATURES   = ["interest_rate", "oil_price", "gdp_growth",
               "unemployment_rate", "food_price_index", "supply_chain_index"]
MODEL_PATH = os.path.join(os.getcwd(), "saved_model.pt")


def _normalize(X):
    mn, mx = X.min(0), X.max(0)
    return (X - mn) / (mx - mn + 1e-8), mn, mx


def _build_nn(in_dim):
    """Only called when TORCH_OK is True."""
    class _Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(in_dim, 64), torch.nn.ReLU(), torch.nn.Dropout(0.1),
                torch.nn.Linear(64, 32),     torch.nn.ReLU(),
                torch.nn.Linear(32, 1),
            )
        def forward(self, x):
            return self.net(x)
    return _Net()


def train_and_evaluate(df: pd.DataFrame, epochs: int = 40):
    avail = [f for f in FEATURES if f in df.columns]
    sub   = df[avail + ["inflation_rate"]].dropna()
    if len(sub) < 20:
        return None, None, None, None, None, None, None, avail

    X = sub[avail].values.astype(float)
    y = sub["inflation_rate"].values.reshape(-1, 1)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    X_norm, mn, mx = _normalize(X_tr)
    X_te_norm = (X_te - mn) / (mx - mn + 1e-8)

    if not TORCH_OK:
        # Sklearn Ridge fallback — no torch needed
        model = Ridge(alpha=1.0)
        model.fit(X_norm, y_tr.ravel())
        preds = model.predict(X_te_norm)
        losses = [float(mean_squared_error(y_tr.ravel(), model.predict(X_norm)))]
        mse = mean_squared_error(y_te.ravel(), preds)
        r2  = r2_score(y_te.ravel(), preds)
        return preds, y_te.flatten(), losses, mse, r2, model, X_te_norm, avail

    X_t  = torch.tensor(X_norm,    dtype=torch.float32)
    y_t  = torch.tensor(y_tr,      dtype=torch.float32)
    X_tv = torch.tensor(X_te_norm, dtype=torch.float32)

    model   = _build_nn(len(avail))
    opt     = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()
    losses  = []

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(X_t), y_t)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        preds = model(X_tv).numpy().flatten()

    mse = mean_squared_error(y_te.flatten(), preds)
    r2  = r2_score(y_te.flatten(), preds)

    try:
        torch.save({"state_dict": model.state_dict(),
                    "mn": mn, "mx": mx, "features": avail}, MODEL_PATH)
    except Exception:
        pass

    return preds, y_te.flatten(), losses, mse, r2, model, X_tv, avail


def load_saved_model():
    if not TORCH_OK:
        return None
    try:
        if os.path.exists(MODEL_PATH):
            return torch.load(MODEL_PATH, weights_only=False)
    except Exception:
        pass
    return None


def actual_vs_predicted_plot(actual, preds) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual, y=preds, mode="markers",
                             marker=dict(color="steelblue", opacity=0.5, size=5),
                             name="Test Set Predictions"))
    mn, mx = float(actual.min()), float(actual.max())
    fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines",
                             line=dict(color="red", dash="dash"), name="Perfect Fit"))
    fig.update_layout(title="Actual vs Predicted — Test Set (held-out 20%)",
                      xaxis_title="Actual Inflation (%)",
                      yaxis_title="Predicted Inflation (%)", height=420)
    return fig


def loss_curve_plot(losses: list) -> go.Figure:
    fig = px.line(y=losses, labels={"index": "Epoch", "y": "MSE Loss"},
                  title="Training Loss Curve")
    fig.update_layout(height=350)
    return fig


def permutation_importance(model, X_t, y_t, feature_names: list) -> go.Figure:
    if TORCH_OK and hasattr(model, "parameters"):
        loss_fn = torch.nn.MSELoss()
        model.eval()
        with torch.no_grad():
            baseline = loss_fn(model(X_t), torch.tensor(y_t.reshape(-1,1), dtype=torch.float32) if not isinstance(y_t, torch.Tensor) else y_t).item()
            importance = {}
            for i, col in enumerate(feature_names):
                X_perm = X_t.clone()
                X_perm[:, i] = X_perm[:, i][torch.randperm(len(X_perm))]
                importance[col] = loss_fn(model(X_perm), torch.tensor(y_t.reshape(-1,1), dtype=torch.float32) if not isinstance(y_t, torch.Tensor) else y_t).item() - baseline
    else:
        # sklearn fallback
        from sklearn.inspection import permutation_importance as sk_pi
        import numpy as np
        result = sk_pi(model, X_t, y_t, n_repeats=5, random_state=42)
        importance = dict(zip(feature_names, result.importances_mean))

    imp_df = (pd.DataFrame(list(importance.items()), columns=["Feature", "Importance"])
                .sort_values("Importance", ascending=True))
    fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                 color="Importance", color_continuous_scale="Viridis",
                 title="Feature Importance (Permutation)")
    fig.update_layout(height=380)
    return fig
