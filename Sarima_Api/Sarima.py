from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict

import pandas as pd
import numpy as np
import uvicorn
import warnings

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from pmdarima import auto_arima   # <-- IMPORTANT: install pmdarima

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# =========================================================
# INITIALIZE APP
# =========================================================
app = FastAPI(
    title="DCPO Crime Intelligence API",
    description="Provides crime forecast, hotspots, and heatmap analytics",
    version="1.0"
)


# =========================================================
# LOAD & CLEAN DATA (RUN ON IMPORT)
# =========================================================

# NOTE: palitan mo kung iba talaga file name mo
DATA_PATH = "../data/DCPO_Data.csv"     # <-- Your dataset here (dirty/raw)

df_raw = pd.read_csv(DATA_PATH)

# ---------- basic cleaning ----------
df = df_raw.copy()
df.columns = df.columns.str.upper().str.strip()

required = ["BARANGAY", "CRIME_TYPE", "YEAR", "MONTH"]
for col in required:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# convert YEAR & MONTH
df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")
df["MONTH"] = pd.to_numeric(df["MONTH"], errors="coerce")
df = df.dropna(subset=["YEAR", "MONTH"])

df["YEAR"] = df["YEAR"].astype(int)
df["MONTH"] = df["MONTH"].astype(int)

# keep valid months
df = df[(df["MONTH"] >= 1) & (df["MONTH"] <= 12)]

# text fields
df["CRIME_TYPE"] = df["CRIME_TYPE"].astype(str).str.upper().str.strip()
df["BARANGAY"] = df["BARANGAY"].astype(str).str.strip()

# date fields
df["DATE"] = pd.to_datetime(dict(year=df["YEAR"], month=df["MONTH"], day=1))
df = df.drop_duplicates().sort_values("DATE").reset_index(drop=True)

# ---------- monthly aggregation ----------
monthly = (
    df.groupby(["DATE", "CRIME_TYPE"])
      .size()
      .reset_index(name="COUNT")
)

monthly_pivot = (
    monthly.pivot(index="DATE", columns="CRIME_TYPE", values="COUNT")
           .fillna(0)
           .sort_index()
)

# ensure walang skip na buwan
full_idx = pd.date_range(
    start=monthly_pivot.index.min(),
    end=monthly_pivot.index.max(),
    freq="MS"
)
monthly_pivot = monthly_pivot.reindex(full_idx).fillna(0)
monthly_pivot.index.name = "DATE"

# total crimes per month
total_monthly = monthly_pivot.sum(axis=1).to_frame(name="TOTAL_CRIMES")


# =========================================================
# RESPONSE MODELS
# =========================================================

class ForecastItem(BaseModel):
    date: str          # YYYY-MM
    crime_type: str
    forecast: float
    lower_ci: float
    upper_ci: float

class ForecastResponse(BaseModel):
    status: str
    horizon: int
    results: List[ForecastItem]

class HotspotItem(BaseModel):
    barangay: str
    total_crime: int

class HotspotResponse(BaseModel):
    status: str
    top_n: int
    results: List[HotspotItem]

class HeatmapResponse(BaseModel):
    status: str
    matrix: Dict[str, Dict[str, int]]  # {barangay: {crime: count}}


# =========================================================
# SARIMA HELPER (same style as sa Colab mo)
# =========================================================

def run_sarima(series: pd.Series, horizon: int = 12) -> List[Dict]:
    """
    SARIMA forecast with:
    - monthly frequency (MS)
    - auto_arima to select (p,d,q)(P,D,Q,12) with D=1
    - linear trend "t"
    """
    s = series.asfreq("MS").fillna(0)

    # kung sobrang konti data, huwag na i-forecast
    if len(s) < 10 or s.sum() < 5:
        return []

    try:
        auto = auto_arima(
            s,
            start_p=0, start_q=0,
            max_p=3, max_q=3,
            start_P=0, start_Q=0,
            max_P=2, max_Q=2,
            m=12,               # monthly seasonality
            seasonal=True,
            d=None,
            D=1,                # force seasonal differencing
            trace=False,
            error_action="ignore",
            suppress_warnings=True,
            stepwise=True,
        )
        order = auto.order
        seasonal_order = auto.seasonal_order
    except Exception:
        # fallback simple model kung mag-fail auto_arima
        order = (1, 0, 0)
        seasonal_order = (1, 1, 0, 12)

    try:
        model = SARIMAX(
            s,
            order=order,
            seasonal_order=seasonal_order,
            trend="t",
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        result = model.fit(disp=False)
    except Exception:
        return []

    forecast = result.get_forecast(steps=horizon)
    mean = forecast.predicted_mean.clip(lower=0)
    ci = forecast.conf_int().clip(lower=0)

    output = []
    for i in range(len(mean)):
        output.append({
            "date": mean.index[i].strftime("%Y-%m"),
            "forecast": float(mean.iloc[i]),
            "lower_ci": float(ci.iloc[i, 0]),
            "upper_ci": float(ci.iloc[i, 1]),
        })

    return output


# =========================================================
# ====================== API ROUTES =======================
# =========================================================

# -----------------------------------------
# 🔥 1. GET HOTSPOTS (Top barangays)
# -----------------------------------------
@app.get("/hotspots", response_model=HotspotResponse)
def get_hotspots(top_n: int = 20):

    totals = (
        df.groupby("BARANGAY")
          .size()
          .reset_index(name="TOTAL")
          .sort_values("TOTAL", ascending=False)
    )

    results = [
        HotspotItem(barangay=row["BARANGAY"], total_crime=int(row["TOTAL"]))
        for _, row in totals.head(top_n).iterrows()
    ]

    return HotspotResponse(
        status="success",
        top_n=top_n,
        results=results
    )


# -----------------------------------------
# 🔥 2. FORECAST PER CRIME TYPE
# -----------------------------------------
@app.get("/forecast/{crime_type}", response_model=ForecastResponse)
def get_forecast(crime_type: str, horizon: int = 12):

    crime_type = crime_type.upper()
    if crime_type not in monthly_pivot.columns:
        raise HTTPException(404, detail="Crime type not found.")

    series = monthly_pivot[crime_type]
    sarima_output = run_sarima(series, horizon)

    if not sarima_output:
        raise HTTPException(500, detail="SARIMA model failed or too little data.")

    results = [
        ForecastItem(
            date=item["date"],
            crime_type=crime_type,
            forecast=item["forecast"],
            lower_ci=item["lower_ci"],
            upper_ci=item["upper_ci"],
        )
        for item in sarima_output
    ]

    return ForecastResponse(
        status="success",
        horizon=horizon,
        results=results
    )


# -----------------------------------------
# 🔥 2B. FORECAST TOTAL CRIMES (All types)
# -----------------------------------------
@app.get("/forecast-total", response_model=ForecastResponse)
def get_forecast_total(horizon: int = 12):

    series = total_monthly["TOTAL_CRIMES"]
    sarima_output = run_sarima(series, horizon)

    if not sarima_output:
        raise HTTPException(500, detail="SARIMA model failed.")

    results = [
        ForecastItem(
            date=item["date"],
            crime_type="TOTAL_CRIMES",
            forecast=item["forecast"],
            lower_ci=item["lower_ci"],
            upper_ci=item["upper_ci"],
        )
        for item in sarima_output
    ]

    return ForecastResponse(
        status="success",
        horizon=horizon,
        results=results
    )


# -----------------------------------------
# 🔥 3. TOP CRIME PER MONTH (Historical)
# -----------------------------------------
@app.get("/top-crimes")
def get_top_crimes():

    top = (
        monthly.sort_values(["DATE", "COUNT"], ascending=[True, False])
               .groupby("DATE")
               .head(1)
               .reset_index(drop=True)
    )

    output = []
    for _, row in top.iterrows():
        output.append({
            "date": row["DATE"].strftime("%Y-%m"),
            "crime_type": row["CRIME_TYPE"],
            "count": int(row["COUNT"])
        })

    return {"status": "success", "results": output}


# -----------------------------------------
# 🔥 4. HEATMAP DATA (Barangay × Crime Type)
# -----------------------------------------
@app.get("/heatmap", response_model=HeatmapResponse)
def get_heatmap():

    matrix = (
        df.groupby(["BARANGAY", "CRIME_TYPE"])
          .size()
          .reset_index(name="COUNT")
    )

    pivot = (
        matrix.pivot(index="BARANGAY", columns="CRIME_TYPE", values="COUNT")
              .fillna(0)
              .astype(int)
    )

    # {barangay: {crime: count}}
    heatmap_dict = pivot.to_dict(orient="index")

    return HeatmapResponse(
        status="success",
        matrix=heatmap_dict
    )


# -----------------------------------------
# ROOT
# -----------------------------------------
@app.get("/")
def root():
    return {
        "status": "running",
        "message": "DCPO Crime Intelligence API is active."
    }


# =========================================================
# RUN SERVER (for local dev)
# =========================================================
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
