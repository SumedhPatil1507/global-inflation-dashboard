"""
Live data loader using World Bank API (wbgapi).
Falls back to synthetic data if API is unavailable.
"""
import pandas as pd
import numpy as np
import wbgapi as wb
import streamlit as st

# World Bank indicator codes
INDICATORS = {
    "inflation_rate": "FP.CPI.TOTL.ZG",
    "gdp_growth": "NY.GDP.MKTP.KD.ZG",
    "unemployment_rate": "SL.UEM.TOTL.ZS",
    "interest_rate": "FR.INR.LEND",
}

COUNTRIES = ["US", "IN", "CN", "JP", "GB", "BR", "CA", "AU", "KR", "VN",
             "EU", "TH", "ID", "PH", "AR", "RU", "SA", "MX", "TR", "ZA"]

COUNTRY_LABELS = {
    "US": "USA", "IN": "IND", "CN": "CHN", "JP": "JPN", "GB": "GBR",
    "BR": "BRA", "CA": "CAN", "AU": "AUS", "KR": "KOR", "VN": "VNM",
    "EU": "EU",  "TH": "THA", "ID": "IDN", "PH": "PHL", "AR": "ARG",
    "RU": "RUS", "SA": "SAU", "MX": "MEX", "TR": "TUR", "ZA": "ZAF",
}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_live_data(start_year: int = 2020, end_year: int = 2024) -> pd.DataFrame:
    """Fetch annual data from World Bank and return a tidy DataFrame."""
    frames = []
    for short_name, indicator in INDICATORS.items():
        try:
            raw = wb.data.DataFrame(indicator, economy=COUNTRIES,
                                    time=range(start_year, end_year + 1),
                                    labels=False)
            raw = raw.reset_index()
            raw = raw.melt(id_vars=["economy", "time"] if "economy" in raw.columns
                           else [raw.columns[0], raw.columns[1]],
                           var_name="drop", value_name=short_name)
            raw = raw.rename(columns={raw.columns[0]: "country", raw.columns[1]: "year"})
            raw = raw[["country", "year", short_name]]
            frames.append(raw)
        except Exception:
            pass  # handled in fallback

    if not frames:
        return _synthetic_data(start_year, end_year)

    df = frames[0]
    for f in frames[1:]:
        df = df.merge(f, on=["country", "year"], how="outer")

    df["country"] = df["country"].map(COUNTRY_LABELS).fillna(df["country"])
    df["year"] = df["year"].astype(int)
    df = df.dropna(subset=["inflation_rate"])

    # Add synthetic monthly columns not available from WB
    df = _add_synthetic_columns(df)
    return df.reset_index(drop=True)


def _add_synthetic_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add oil_price, food_price_index, money_supply_m2, supply_chain_index, exchange_rate_usd."""
    rng = np.random.default_rng(42)
    n = len(df)
    df = df.copy()
    df["oil_price"] = rng.uniform(40, 120, n)
    df["food_price_index"] = rng.uniform(90, 160, n)
    df["money_supply_m2"] = rng.uniform(5, 25, n)
    df["supply_chain_index"] = rng.uniform(-2, 4, n)
    df["exchange_rate_usd"] = rng.uniform(0.5, 150, n)
    return df


def _synthetic_data(start_year: int, end_year: int) -> pd.DataFrame:
    """Fully synthetic fallback dataset matching the original CSV schema."""
    rng = np.random.default_rng(0)
    countries = list(COUNTRY_LABELS.values())
    years = list(range(start_year, end_year + 1))
    rows = []
    for country in countries:
        base_inf = rng.uniform(1, 15)
        for year in years:
            rows.append({
                "country": country,
                "year": year,
                "inflation_rate": max(0, base_inf + rng.normal(0, 2)),
                "interest_rate": max(0, rng.uniform(0.1, 18)),
                "gdp_growth": rng.uniform(-5, 8),
                "unemployment_rate": rng.uniform(2, 20),
                "oil_price": rng.uniform(40, 120),
                "food_price_index": rng.uniform(90, 160),
                "money_supply_m2": rng.uniform(5, 25),
                "supply_chain_index": rng.uniform(-2, 4),
                "exchange_rate_usd": rng.uniform(0.5, 150),
            })
    return pd.DataFrame(rows)


def get_data(source: str = "live", start_year: int = 2020, end_year: int = 2024) -> pd.DataFrame:
    if source == "live":
        return fetch_live_data(start_year, end_year)
    return _synthetic_data(start_year, end_year)
