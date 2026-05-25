"""
Real data fetcher — FRED (M2, CPI, rates), yfinance (Oil, Gold, Food ETF).
No synthetic proxies for these series.
"""
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from backend.core.config import get_settings

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

FRED_SERIES = {
    "inflation_rate":    "CPIAUCSL",
    "interest_rate":     "FEDFUNDS",
    "unemployment_rate": "UNRATE",
    "gdp_growth":        "A191RL1Q225SBEA",
    "money_supply_m2":   "M2SL",          # Real M2 from FRED
}

YFINANCE_TICKERS = {
    "oil_price":        "CL=F",    # WTI Crude futures
    "gold_price":       "GC=F",    # Gold futures
    "food_price_index": "DBA",     # Invesco DB Agriculture ETF (proxy)
}


def _fred_fetch(series_id: str, start: str, end: str, freq: str = "a") -> pd.Series:
    cfg = get_settings()
    if not cfg.fred_api_key:
        return pd.Series(dtype=float)
    try:
        r = requests.get(FRED_BASE, params={
            "series_id": series_id, "api_key": cfg.fred_api_key,
            "file_type": "json", "observation_start": start,
            "observation_end": end, "frequency": freq,
        }, timeout=15)
        r.raise_for_status()
        obs = r.json().get("observations", [])
        s   = pd.DataFrame(obs).set_index("date")["value"]
        s   = pd.to_numeric(s, errors="coerce").dropna()
        s.index = pd.to_datetime(s.index).year
        return s.groupby(level=0).mean()
    except Exception:
        return pd.Series(dtype=float)


def _yfinance_fetch(ticker: str, start: str, end: str) -> pd.Series:
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            return pd.Series(dtype=float)
        s = df["Close"].resample("YE").mean()
        s.index = s.index.year
        return s
    except Exception:
        return pd.Series(dtype=float)


def fetch_real_data(start_year: int = 2020, end_year: int = 2024) -> pd.DataFrame:
    """
    Fetch all real data. Returns tidy DataFrame with country/year index.
    US data from FRED + yfinance. World Bank for other countries.
    """
    start = f"{start_year}-01-01"
    end   = f"{end_year}-12-31"

    # ── FRED series ───────────────────────────────────────────────────────────
    fred_data = {}
    for col, sid in FRED_SERIES.items():
        s = _fred_fetch(sid, start, end)
        if not s.empty:
            fred_data[col] = s

    # ── yfinance commodity prices ─────────────────────────────────────────────
    for col, ticker in YFINANCE_TICKERS.items():
        s = _yfinance_fetch(ticker, start, end)
        if not s.empty:
            fred_data[col] = s

    if not fred_data:
        return pd.DataFrame()

    # Align all series to year index
    years = list(range(start_year, end_year + 1))
    rows  = []
    for year in years:
        row = {"country": "USA", "year": year, "data_source": "FRED+yfinance"}
        for col, s in fred_data.items():
            row[col] = float(s.get(year, np.nan))
        rows.append(row)

    df = pd.DataFrame(rows)
    df["last_updated"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    return df


def fetch_world_bank_data(start_year: int = 2020, end_year: int = 2024) -> pd.DataFrame:
    """World Bank for non-US countries."""
    try:
        import wbgapi as wb
        INDICATORS = {
            "inflation_rate":    "FP.CPI.TOTL.ZG",
            "gdp_growth":        "NY.GDP.MKTP.KD.ZG",
            "unemployment_rate": "SL.UEM.TOTL.ZS",
            "interest_rate":     "FR.INR.LEND",
        }
        COUNTRIES = ["IN","CN","JP","GB","BR","CA","AU","KR","VN",
                     "TH","ID","PH","AR","RU","SA","MX","TR","ZA"]
        LABELS    = {"IN":"IND","CN":"CHN","JP":"JPN","GB":"GBR","BR":"BRA",
                     "CA":"CAN","AU":"AUS","KR":"KOR","VN":"VNM","TH":"THA",
                     "ID":"IDN","PH":"PHL","AR":"ARG","RU":"RUS","SA":"SAU",
                     "MX":"MEX","TR":"TUR","ZA":"ZAF"}
        frames = []
        for col, ind in INDICATORS.items():
            try:
                raw = wb.data.DataFrame(ind, economy=COUNTRIES,
                                        time=range(start_year, end_year+1),
                                        labels=False, numericTimeKeys=True)
                raw = raw.reset_index().rename(columns={"economy":"country"})
                yc  = [c for c in raw.columns if isinstance(c, (int, float))]
                raw = raw.melt(id_vars="country", value_vars=yc,
                               var_name="year", value_name=col)
                raw["year"] = raw["year"].astype(int)
                frames.append(raw[["country","year",col]])
            except Exception:
                pass
        if not frames:
            return pd.DataFrame()
        df = frames[0]
        for f in frames[1:]:
            df = df.merge(f, on=["country","year"], how="outer")
        df["country"]     = df["country"].map(LABELS).fillna(df["country"])
        df["data_source"] = "World Bank"
        return df.dropna(subset=["inflation_rate"])
    except Exception:
        return pd.DataFrame()
