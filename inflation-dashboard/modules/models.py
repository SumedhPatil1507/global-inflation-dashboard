"""Neural network training and evaluation with Plotly output."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score

FEATURES = ["interest_rate", "oil_price", "gdp_growth",
            "unemployment_rate", "food_price_index", "supply_chain_index"]


def _normalize(X: np.ndarray):
    mn, mx = X.min(0), X.max(0)
    return (X - mn) / (mx - mn + 1e-8), mn, mx


class _SimpleNN(nn.Module):
    def __init__(self, in_dim: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_and_evaluate(df: pd.DataFrame, epochs: int = 40):
    avail = [f for f in FEATURES if f in df.columns]
    sub = df[avail + ["inflation_rate"]].dropna()
    X = sub[avail].values.astype(float)
    y = sub["inflation_rate"].values.reshape(-1, 1)

    X_norm, _, _ = _normalize(X)
    X_t = torch.tensor(X_norm, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    model = _SimpleNN(len(avail))
    opt = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    losses = []

    for _ in range(epochs):
        opt.zero_grad()
        out = model(X_t)
        loss = loss_fn(out, y_t)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    preds = model(X_t).detach().numpy().flatten()
    actual = y.flatten()
    mse = mean_squared_error(actual, preds)
    r2 = r2_score(actual, preds)

    return preds, actual, losses, mse, r2, model, X_t, avail


def actual_vs_predicted_plot(actual, preds) -> go.Figure:
    sample = min(5000, len(actual))
    idx = np.random.choice(len(actual), sample, replace=False)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual[idx], y=preds[idx], mode="markers",
                             marker=dict(color="steelblue", opacity=0.5, size=4),
                             name="Predictions"))
    mn, mx = actual.min(), actual.max()
    fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines",
                             line=dict(color="red", dash="dash"), name="Perfect Fit"))
    fig.update_layout(title="Actual vs Predicted Inflation Rate",
                      xaxis_title="Actual", yaxis_title="Predicted", height=420)
    return fig


def loss_curve_plot(losses) -> go.Figure:
    fig = px.line(y=losses, labels={"index": "Epoch", "y": "MSE Loss"},
                  title="Training Loss Curve")
    fig.update_layout(height=350)
    return fig


def permutation_importance(model, X_t, y_t, feature_names) -> go.Figure:
    loss_fn = nn.MSELoss()
    baseline = loss_fn(model(X_t), y_t).item()
    importance = {}
    for i, col in enumerate(feature_names):
        X_perm = X_t.clone()
        X_perm[:, i] = X_perm[:, i][torch.randperm(len(X_perm))]
        importance[col] = loss_fn(model(X_perm), y_t).item() - baseline

    imp_df = pd.DataFrame(list(importance.items()), columns=["Feature", "Importance"])
    imp_df = imp_df.sort_values("Importance", ascending=True)
    fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                 color="Importance", color_continuous_scale="Viridis",
                 title="Feature Importance (Permutation Method)")
    fig.update_layout(height=380)
    return fig
