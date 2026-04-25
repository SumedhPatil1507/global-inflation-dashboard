"""
Neural network — inflation predictor.
Includes train/test split, model persistence, confidence scoring.
"""
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

FEATURES = ["interest_rate", "oil_price", "gdp_growth",
            "unemployment_rate", "food_price_index", "supply_chain_index"]

MODEL_PATH = os.path.join(os.getcwd(), "saved_model.pt")


def _normalize(X: np.ndarray):
    mn, mx = X.min(0), X.max(0)
    return (X - mn) / (mx - mn + 1e-8), mn, mx


class _SimpleNN(nn.Module):
    def __init__(self, in_dim: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32),     nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_and_evaluate(df: pd.DataFrame, epochs: int = 40):
    avail = [f for f in FEATURES if f in df.columns]
    sub   = df[avail + ["inflation_rate"]].dropna()

    if len(sub) < 20:
        return None, None, None, None, None, None, None, avail

    X = sub[avail].values.astype(float)
    y = sub["inflation_rate"].values.reshape(-1, 1)

    # Train / test split (80/20)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    X_norm, mn, mx = _normalize(X_tr)
    X_te_norm       = (X_te - mn) / (mx - mn + 1e-8)

    X_t  = torch.tensor(X_norm,    dtype=torch.float32)
    y_t  = torch.tensor(y_tr,      dtype=torch.float32)
    X_tv = torch.tensor(X_te_norm, dtype=torch.float32)
    y_tv = torch.tensor(y_te,      dtype=torch.float32)

    model   = _SimpleNN(len(avail))
    opt     = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    losses  = []

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(X_t), y_t)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    # Evaluate on held-out test set
    model.eval()
    with torch.no_grad():
        preds_test = model(X_tv).numpy().flatten()

    actual_test = y_te.flatten()
    mse = mean_squared_error(actual_test, preds_test)
    r2  = r2_score(actual_test, preds_test)

    # Save model
    try:
        torch.save({"state_dict": model.state_dict(),
                    "mn": mn, "mx": mx, "features": avail}, MODEL_PATH)
    except Exception:
        pass

    # Return test-set predictions for plotting
    return preds_test, actual_test, losses, mse, r2, model, X_tv, avail


def load_saved_model() -> dict | None:
    """Load previously trained model. Returns None if not found."""
    try:
        if os.path.exists(MODEL_PATH):
            return torch.load(MODEL_PATH, weights_only=False)
    except Exception:
        pass
    return None


def actual_vs_predicted_plot(actual: np.ndarray, preds: np.ndarray) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=actual, y=preds, mode="markers",
        marker=dict(color="steelblue", opacity=0.5, size=5),
        name="Test Set Predictions",
    ))
    mn, mx = float(actual.min()), float(actual.max())
    fig.add_trace(go.Scatter(
        x=[mn, mx], y=[mn, mx], mode="lines",
        line=dict(color="red", dash="dash"), name="Perfect Fit",
    ))
    fig.update_layout(
        title="Actual vs Predicted — Test Set (held-out 20%)",
        xaxis_title="Actual Inflation (%)",
        yaxis_title="Predicted Inflation (%)", height=420,
    )
    return fig


def loss_curve_plot(losses: list) -> go.Figure:
    fig = px.line(y=losses, labels={"index": "Epoch", "y": "MSE Loss"},
                  title="Training Loss Curve")
    fig.update_layout(height=350)
    return fig


def permutation_importance(model: nn.Module, X_t: torch.Tensor,
                            y_t: torch.Tensor, feature_names: list) -> go.Figure:
    loss_fn = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        baseline   = loss_fn(model(X_t), y_t).item()
        importance = {}
        for i, col in enumerate(feature_names):
            X_perm       = X_t.clone()
            X_perm[:, i] = X_perm[:, i][torch.randperm(len(X_perm))]
            importance[col] = loss_fn(model(X_perm), y_t).item() - baseline

    imp_df = (pd.DataFrame(list(importance.items()), columns=["Feature", "Importance"])
                .sort_values("Importance", ascending=True))
    fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                 color="Importance", color_continuous_scale="Viridis",
                 title="Feature Importance (Permutation — Test Set)")
    fig.update_layout(height=380)
    return fig
