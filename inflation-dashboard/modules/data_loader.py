"""
Data loader — World Bank (annual, 19 countries) + FRED (monthly, US).
Falls back to synthetic data if all APIs are unavailable.
"""
import pandas as pd
import numpy as np
import requests
import streamlit as st

# ── World Bank ────────────────────────────────────────────────────────────────
WB_INDICATORS = {
    "inflation_rate":    "FP.CPI.TOTL.ZG",
    "gdp_growth":        "NY.GDP.MKTP.KD.ZG",
    "unemployment_rate": "SL.UEM.TOTL.ZS",
    "interest_rate":     "FR.INR.LEND",
}

WB_COUNTRIES = ["US","IN","CN","JP","GB","BR","CA","AU","KR","VN",
                "TH","ID","PH","AR","RU","SA","MX","TR","ZA"]

COUNTRY_LABELS = {
    "US":"USA","IN":"IND","CN":"CHN","JP":"JPN","GB":"GBR","BR":"BRA",
    "CA":"CAN","AU":"AUS","KR":"KOR","VN":"VNM","TH":"THA","ID":"IDN",
    "PH":"PHL","AR":"ARG","RU":"RUS","SA":"SAU","MX":"MEX","TR":"TUR","ZA":"ZAF",
}

# ── FRED series IDs ───────────────────────────────────────────────────────────
FRED_SERIES = {
    "inflation_rate":    "CPIAUCSL",   # CPI All Urban Consumers
    "interest_rate":     "FEDFUNDS",   # Federal Funds Rate
    "unemployment_rate": "UNRATE",     # Unemployment Rate
    "gdp_growth":        "A191RL1Q225SBEA",  # Real GDP growth (quarterly)
}
FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"


def _fred_key() -> str:
    try:
        return st.secrets.get("fred", {}).get("api_key", "")
    except Exception:
        return ""


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fred_monthly(start_year: int = 2020, end_year: int = 2024) -> pd.DataFrame:
    """Fetch monthly US data from FRED. Returns empty df if no API key."""
    key = _fred_key()
    if not key:
        return pd.DataFrame()

    frames = []
    for col, series_id in FRED_SERIES.items():
        try:
            r = requests.get(FRED_BASE, params={
                "series_id":        series_id,
                "api_key":          key,
                "file_type":        "json",
                "observation_start": f"{start_year}-01-01",
                "observation_end":   f"{end_year}-12-31",
                "frequency":        "a",   # annual aggregation
            }, timeout=10)
            r.raise_for_status()
            obs = r.json().get("observations", [])
            tmp = pd.DataFrame(obs)[["date", "value"]]
            tmp["value"] = pd.to_numeric(tmp["value"], errors="coerce")
            tmp["year"]  = pd.to_datetime(tmp["date"]).dt.year
            tmp = tmp.groupby("year")["value"].mean().reset_index()
            tmp.columns = ["year", col]
            frames.append(tmp)
        except Exception:
            pass

    if not frames:
        return pd.DataFrame()

    df = frames[0]
    for f in frames[1:]:
        df = df.merge(f, on="year", how="outer")
    df["country"] = "USA"
    df["data_source"] = "FRED"
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_world_bank(start_year: int = 2020, end_year: int = 2024) -> pd.DataFrame:
    """Fetch annual data from World Bank via wbgapi."""
    try:
        import wbgapi as wb
    except ImportError:
        return pd.DataFrame()

    frames = []
    for short_name, indicator in WB_INDICATORS.items():
        try:
            raw = wb.data.DataFrame(
                indicator, economy=WB_COUNTRIES,
                time=range(start_year, end_year + 1),
                labels=False, numericTimeKeys=True,
            )
            raw = raw.reset_index().rename(columns={"economy": "country"})
            year_cols = [c for c in raw.columns if isinstance(c, (int, float))]
            raw = raw.melt(id_vars="country", value_vars=year_cols,
                           var_name="year", value_name=short_name)
            raw["year"] = raw["year"].astype(int)
            frames.append(raw[["country", "year", short_name]])
        except Exception:
            pass

    if not frames:
        return pd.DataFrame()

    df = frames[0]
    for f in frames[1:]:
        df = df.merge(f, on=["country", "year"], how="outer")

    df["country"]     = df["country"].map(COUNTRY_LABELS).fillna(df["country"])
    df["data_source"] = "World Bank"
    return df.dropna(subset=["inflation_rate"])


def _add_proxy_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add proxy columns for variables not in World Bank / FRED."""
    rng = np.random.default_rng(42)
    n   = len(df)
    for col, lo, hi in [
        ("oil_price",          40,  120),
        ("food_price_index",   90,  160),
        ("money_supply_m2",     5,   25),
        ("supply_chain_index", -2,    4),
        ("exchange_rate_usd",  0.5, 150),
    ]:
        if col not in df.columns:
            df[col] = rng.uniform(lo, hi, n)
    return df


def _synthetic_data(start_year: int, end_year: int) -> pd.DataFrame:
    rng  = np.random.default_rng(0)
    rows = []
    for country in COUNTRY_LABELS.values():
        base = rng.uniform(1, 15)
        for year in range(start_year, end_year + 1):
            rows.append({
                "country":            country,
                "year":               year,
                "inflation_rate":     float(max(0, base + rng.normal(0, 2))),
                "interest_rate":      float(max(0, rng.uniform(0.1, 18))),
                "gdp_growth":         float(rng.uniform(-5, 8)),
                "unemployment_rate":  float(rng.uniform(2, 20)),
                "oil_price":          float(rng.uniform(40, 120)),
                "food_price_index":   float(rng.uniform(90, 160)),
                "money_supply_m2":    float(rng.uniform(5, 25)),
                "supply_chain_index": float(rng.uniform(-2, 4)),
                "exchange_rate_usd":  float(rng.uniform(0.5, 150)),
                "data_source":        "Synthetic",
            })
    return pd.DataFrame(rows)


@st.cache_data(ttl=3600, show_spinner=False)
def get_data(source: str = "live", start_year: int = 2020, end_year: int = 2024) -> pd.DataFrame:
    if source != "live":
        return _synthetic_data(start_year, end_year)

    wb_df   = fetch_world_bank(start_year, end_year)
    fred_df = fetch_fred_monthly(start_year, end_year)

    if wb_df.empty and fred_df.empty:
        return _synthetic_data(start_year, end_year)

    # Merge FRED into WB — FRED overrides USA rows if available
    if not wb_df.empty and not fred_df.empty:
        wb_no_usa = wb_df[wb_df["country"] != "USA"]
        df = pd.concat([wb_no_usa, fred_df], ignore_index=True)
    elif not wb_df.empty:
        df = wb_df
    else:
        df = fred_df

    df = _add_proxy_columns(df)
    df["last_updated"] = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    return df.reset_index(drop=True)
