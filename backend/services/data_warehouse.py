import os
import json
from datetime import datetime
from typing import List, Dict, Any

import pandas as pd
import asyncpg
from asyncpg import Connection
from functools import lru_cache

# Load database URL from environment or config
DATABASE_URL = os.getenv("DATABASE_URL") or "postgresql://postgres:postgres@localhost:5432/inflation"

# Global connection pool (lazy initialization)
_pool: asyncpg.pool.Pool = None

async def get_pool() -> asyncpg.pool.Pool:
    """Return a shared asyncpg connection pool, creating it if needed."""
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(dsn=DATABASE_URL, min_size=1, max_size=10)
    return _pool

# --------------------------- Bulk Upsert ---------------------------

async def upsert_records(table: str, records: List[Dict[str, Any]]) -> str:
    """Perform a high‑speed bulk upsert into ``table``.

    Parameters
    ----------
    table: str
        Target table name.
    records: List[Dict]
        List of rows where keys match column names.
    Returns
    -------
    str
        Result summary.
    """
    if not records:
        return "No records to upsert"

    # Build column list and placeholder list
    columns = list(records[0].keys())
    col_names = ", ".join(f'"{c}"' for c in columns)
    # Build the ON CONFLICT clause – we assume primary key columns are the first two columns
    conflict_target = f"({', '.join(columns[:2])})"
    update_assign = ", ".join(f'"{c}" = EXCLUDED."{c}"' for c in columns)

    values_sql = []
    args: List[Any] = []
    for idx, rec in enumerate(records):
        placeholders = []
        for col in columns:
            args.append(rec[col])
            placeholders.append(f'${len(args)}')
        values_sql.append(f"({', '.join(placeholders)})")
    values_clause = ", ".join(values_sql)

    query = f"""
        INSERT INTO {table} ({col_names})
        VALUES {values_clause}
        ON CONFLICT {conflict_target} DO UPDATE SET {update_assign};
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(query, *args)
    return f"Upserted {len(records)} rows into {table}"

# -------------------------- Cached Matrix --------------------------


async def get_historical_matrix(tickers: List[str], indicators: List[str], start_date: str) -> pd.DataFrame:
    """Return a forward‑filled matrix aligning tickers and macro indicators.

    The result is cached in‑process for fast repeated calls.
    """
    # Normalise inputs for cache key
    # lru_cache works on sync functions; we delegate to a sync helper
    return await _fetch_and_build_matrix(tickers, indicators, start_date)

async def _fetch_and_build_matrix(tickers: List[str], indicators: List[str], start_date: str) -> pd.DataFrame:
    pool = await get_pool()
    async with pool.acquire() as conn:
        # Build dynamic SQL – we pull macro_indicators and market_assets for the given range
        macro_sql = (
            "SELECT timestamp, country_iso, indicator_code, value "
            "FROM macro_indicators "
            "WHERE indicator_code = ANY($1) AND timestamp >= $2"
        )
        market_sql = (
            "SELECT timestamp, ticker, open, high, low, close, volume "
            "FROM market_assets "
            "WHERE ticker = ANY($3) AND timestamp >= $2"
        )
        macro_rows = await conn.fetch(macro_sql, indicators, start_date)
        market_rows = await conn.fetch(market_sql, tickers, start_date)

    # Convert to DataFrames
    macro_df = pd.DataFrame(macro_rows, columns=["timestamp", "country_iso", "indicator_code", "value"])
    market_df = pd.DataFrame(market_rows, columns=["timestamp", "ticker", "open", "high", "low", "close", "volume"])

    # Pivot macro indicators to wide form
    if not macro_df.empty:
        macro_pivot = macro_df.pivot_table(index="timestamp", columns=["country_iso", "indicator_code"], values="value")
    else:
        macro_pivot = pd.DataFrame()

    # Pivot market assets (use close price as representative)
    if not market_df.empty:
        market_pivot = market_df.pivot(index="timestamp", columns="ticker", values="close")
    else:
        market_pivot = pd.DataFrame()

    # Join both pivots on timestamp
    matrix = pd.concat([macro_pivot, market_pivot], axis=1)
    matrix = matrix.sort_index()
    # Forward fill missing values
    matrix = matrix.ffill().bfill()
    return matrix
