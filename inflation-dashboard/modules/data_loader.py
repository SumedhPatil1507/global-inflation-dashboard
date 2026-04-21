"""
Live data loader using World Bank API (wbgapi).
Falls back to synthetic data if API is unavailable.
"""
import pandas as pd
import numpy as np
import streamlit as st

INDICATORS = {
    "inflation_rate":   "FP.CPI.TOTL.ZG",
    "gdp_growth":       "NY.GDP.MKTP.KD.ZG",
    "unemployment_rate":"SL.UEM.TOTL.ZS",
    "interest_rate":    "FR.INR.LEND",
}

COUNTRIES = ["US", "IN", "CN", "JP", "GB", "BR", "CA", "AU", "KR", "VN",
             "TH", "ID", "PH", "AR", "RU", "SA", "MX", "TR", "ZA"]

COUNTRY_LABELS = {
    "US": "USA", "IN": "IND", "CN": "CHN", "JP": "JPN", "GB": "GBR",
    "BR": "BRA", "CA": "CAN", "AU": "AUS", "KR": "KOR", "VN": "VNM",
    "TH": "THA", "ID": "IDN", "PH": "PHL", "AR": "ARG", "RU": "RUS",
    "SA": "SAU", "MX": "MEX", "TR": "TUR", "ZA": "ZAF",
}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_live_data(start_year: int = 2020, end_year: int = 2024) -> pd.DataFrame:
    """Fetch annual data from World Bank and return a tidy DataFrame."""
    try:
        import wbgapi as wb
    except ImportError:
        return _synthetic_data(start_year, end_year)

    frames = []
    for short_name, indicator in INDICATORS.items():
        try:
            # Returns DataFrame with MultiIndex (economy, time) or columns as years
            raw = wb.data.DataFrame(
                indicator, economy=COUNTRIES,
                time=range(start_year, end_year + 1),
                labels=False, numericTimeKeys=True,
            )
            # raw index = economy codes, columns = years (int)
            raw = raw.reset_index()                          # economy becomes a column
            raw = raw.rename(columns={"economy": "country"})
            # Melt years into rows
            year_cols = [c for c in raw.columns if isinstance(c, (int, float))]
            raw = raw.melt(id_vars="country", value_vars=year_cols,
                           var_name="year", value_name=short_name)
            raw["year"] = raw["year"].astype(int)
            frames.append(raw[["country", "year", short_name]])
        except Exception:
            pass

    if not frames:
        return _synthetic_data(start_year, end_year)

    df = frames[0]
    for f in frames[1:]:
        df = df.merge(f, on=["country", "year"], how="outer")

    df["country"] = df["country"].map(COUNTRY_LABELS).fillna(df["country"])
    df = df.dropna(subset=["inflation_rate"])
    df = _add_synthetic_columns(df)
    return df.reset_index(drop=True)


def _add_synthetic_columns(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = len(df)
    df = df.copy()
    df["oil_price"]         = rng.uniform(40, 120, n)
    df["food_price_index"]  = rng.uniform(90, 160, n)
    df["money_supply_m2"]   = rng.uniform(5,  25,  n)
    df["supply_chain_index"]= rng.uniform(-2, 4,   n)
    df["exchange_rate_usd"] = rng.uniform(0.5,150,  n)
    return df


def _synthetic_data(start_year: int, end_year: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    countries = list(COUNTRY_LABELS.values())
    rows = []
    for country in countries:
        base_inf = rng.uniform(1, 15)
        for year in range(start_year, end_year + 1):
            rows.append({
                "country":           country,
                "year":              year,
                "inflation_rate":    float(max(0, base_inf + rng.normal(0, 2))),
                "interest_rate":     float(max(0, rng.uniform(0.1, 18))),
                "gdp_growth":        float(rng.uniform(-5, 8)),
                "unemployment_rate": float(rng.uniform(2, 20)),
                "oil_price":         float(rng.uniform(40, 120)),
                "food_price_index":  float(rng.uniform(90, 160)),
                "money_supply_m2":   float(rng.uniform(5, 25)),
                "supply_chain_index":float(rng.uniform(-2, 4)),
                "exchange_rate_usd": float(rng.uniform(0.5, 150)),
            })
    return pd.DataFrame(rows)


def get_data(source: str = "live", start_year: int = 2020, end_year: int = 2024) -> pd.DataFrame:
    if source == "live":
        return fetch_live_data(start_year, end_year)
    return _synthetic_data(start_year, end_year)
