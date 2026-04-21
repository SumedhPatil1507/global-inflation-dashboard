# 🌍 Global Inflation Insights Dashboard

A premium, end-to-end Streamlit analytics product for post-COVID global inflation analysis.

## Features

| Module | What it does |
|---|---|
| **Live Data** | Fetches real CPI, GDP, unemployment data from World Bank API |
| **EDA** | Histograms, line trends, boxplots, violin, heatmap, scatter matrix |
| **ML Models** | PyTorch neural network with loss curve + permutation feature importance |
| **Anomaly Detection** | Z-score statistical + Autoencoder deep learning |
| **Clustering** | Hierarchical dendrogram + K-Means elbow + scatter |
| **Forecasting** | LSTM per-country inflation forecast with configurable horizon |
| **Advanced Plots** | 3D scatter, contour density, hexbin, facet grid |

## Quickstart

```bash
# 1. Clone
git clone https://github.com/SumedhPatil1507/global-inflation-dashboard.git
cd global-inflation-dashboard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```

## Project Structure

```
inflation-dashboard/
├── app.py                  # Main Streamlit app
├── modules/
│   ├── data_loader.py      # World Bank API + synthetic fallback
│   ├── eda.py              # Interactive EDA (Plotly)
│   ├── models.py           # Neural network training & evaluation
│   ├── anomaly.py          # Z-score + Autoencoder anomaly detection
│   ├── clustering.py       # Hierarchical + K-Means clustering
│   ├── forecasting.py      # LSTM forecasting
│   └── advanced_plots.py   # 3D, contour, hexbin, facet
├── requirements.txt
└── README.md
```

## Data Sources

- **World Bank API** (`wbgapi`) — CPI inflation, GDP growth, unemployment, lending rates
- Synthetic columns added for oil price, food price index, money supply M2, supply chain index

## Deployment

### Streamlit Community Cloud (free)
1. Push to GitHub (see commands below)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → set `app.py` as entry point → Deploy

### Local with custom port
```bash
streamlit run app.py --server.port 8502
```

## Push to GitHub

```bash
cd inflation-dashboard
git init
git add .
git commit -m "feat: global inflation insights dashboard"
git branch -M main
git remote add origin https://github.com/SumedhPatil1507/global-inflation-dashboard.git
git push -u origin main
```
