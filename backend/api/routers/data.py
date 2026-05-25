"""Data endpoints — real FRED + yfinance + World Bank."""
from fastapi import APIRouter, Depends, Query
from backend.core.security import get_current_user
from backend.services.data_service import fetch_real_data, fetch_world_bank_data
import pandas as pd

router = APIRouter(prefix="/api/data", tags=["data"])


@router.get("/combined")
async def get_combined(
    start_year: int = Query(2020, ge=2000, le=2024),
    end_year:   int = Query(2024, ge=2000, le=2024),
    user: dict = Depends(get_current_user),
):
    us_df  = fetch_real_data(start_year, end_year)
    wb_df  = fetch_world_bank_data(start_year, end_year)

    if us_df.empty and wb_df.empty:
        return {"data": [], "source": "unavailable"}

    if not us_df.empty and not wb_df.empty:
        wb_no_us = wb_df[wb_df["country"] != "USA"]
        df = pd.concat([wb_no_us, us_df], ignore_index=True)
    elif not us_df.empty:
        df = us_df
    else:
        df = wb_df

    return {"data": df.where(pd.notna(df), None).to_dict(orient="records"),
            "source": "FRED+yfinance+WorldBank",
            "rows": len(df)}


@router.get("/countries")
async def get_countries(user: dict = Depends(get_current_user)):
    wb_df = fetch_world_bank_data(2020, 2024)
    countries = sorted(wb_df["country"].unique().tolist()) if not wb_df.empty else []
    if "USA" not in countries:
        countries = ["USA"] + countries
    return {"countries": countries}
