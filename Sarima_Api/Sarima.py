from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import logging
import asyncio

# Suppress noisy convergence warnings from statsmodels by default;
# safe_fit will still detect them when needed.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.ERROR)

# ============================================================
# Pydantic models (API response formats)
# ============================================================

class TotalForecastItem(BaseModel):
    month: str          # e.g. "2026-01-01"
    forecast_total: float
    lower_ci: float
    upper_ci: float

class TotalForecastResponse(BaseModel):
    status: str
    horizon: int
    data: List[TotalForecastItem]

class TopCrimeItem(BaseModel):
    month: str
    top_offense: str
    top_value: float

class TopCrimeResponse(BaseModel):
    status: str
    horizon: int
    data: List[TopCrimeItem]

# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="DCPO SARIMA Crime Forecast API",
    description=(
        "Uses optimized SARIMA model on DCPO_5years_monthly.csv to forecast:\n"
        "- total monthly crimes\n"
        "- top crime type per future month\n"
        "Dataset includes both crime and cybercrime complaints."
    ),
    version="2.0.0",
)

# ============================================================
# GLOBALS (loaded at startup)
# ============================================================

df_dcpo: pd.DataFrame | None = None         # cleaned raw data
ts_total: pd.Series | None = None           # total crimes per month
monthly_offense: pd.DataFrame | None = None # crimes per offense per month
model_total_full: SARIMAX | None = None     # fitted SARIMA for total
offense_models: dict[str, SARIMAX] = {}     # fitted SARIMA per offense

# Chosen best SARIMA order from your optimization
BEST_ORDER = (1, 1, 1)
BEST_SEASONAL_ORDER = (0, 1, 1, 12)


# ============================================================
# Helper: load CSV, clean, build series, train models
# ============================================================

def load_and_train() -> None:
    """Load DCPO data, clean it, build monthly series, train SARIMA models."""
    global df_dcpo, ts_total, monthly_offense, model_total_full, offense_models

    # ---------- 1. Load CSV ----------
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "..", "data", "DCPO_5years_monthly.csv")
    csv_path = os.path.abspath(csv_path)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"DCPO_5years_monthly.csv not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    # Expect columns: gu, Date, offense, Count
    required_cols = {"Date", "offense", "Count"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # ---------- 2. Basic Cleaning ----------
    # Parse dates
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date")

    # Normalize offense labels (upper-case, trim spaces)
    df["offense"] = df["offense"].astype(str).str.upper().str.strip()

    # Ensure Count is numeric
    df["Count"] = pd.to_numeric(df["Count"], errors="coerce").fillna(0)

    # Keep a clean copy
    df_dcpo = df.copy()

    # Month index (period → timestamp at month start)
    df["month"] = df["Date"].dt.to_period("M").dt.to_timestamp()

    # ---------- 3. Build monthly TOTAL series ----------
    monthly_total = (
        df.groupby("month")["Count"]
          .sum()
          .rename("count")
          .to_frame()
    )

    ts = monthly_total["count"].astype(float)
    ts = ts.asfreq("MS").fillna(0)   # ensure monthly frequency

    ts_total = ts

    # ---------- 4. Build monthly OFFENSE matrix ----------
    monthly_off = (
        df.groupby(["month", "offense"])["Count"]
          .sum()
          .unstack(fill_value=0)
    )
    monthly_off = monthly_off.asfreq("MS").fillna(0)
    monthly_offense = monthly_off

    model = SARIMAX(
        ts_total,
        order=BEST_ORDER,
        seasonal_order=BEST_SEASONAL_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    # Use a robust fit wrapper that retries with alternative optimizers
    def safe_fit(smodel):
        """Fit `smodel`, detect ConvergenceWarning without raising it, and retry.

        Strategy:
        - Run a normal fit under a local warnings capture that records any
          ConvergenceWarning instances (so global suppression doesn't hide them).
        - If a ConvergenceWarning was recorded, retry with alternative optimizers
          (`powell`, then `nm`). Log only INFO-level messages so the terminal
          isn't flooded with repeated WARNING lines.

        Returns the fitted results object.
        """
        # Attempt 1: run normally but capture any ConvergenceWarning instances
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", ConvergenceWarning)
                res = smodel.fit(disp=False, method="lbfgs", maxiter=1000)

                # Check if any recorded warnings are ConvergenceWarning
                conv_warns = [x for x in w if issubclass(x.category, ConvergenceWarning)]

            if not conv_warns:
                return res
            # Otherwise, log and fall through to retries
            logger.debug("lbfgs produced ConvergenceWarning; retrying with 'powell' (more robust).")
        except Exception as e:
            # If the fit raised an exception (not just a convergence warning), log and retry
            logger.info("Fit attempt (lbfgs) raised an exception; retrying: %s", e)

        # Retry 1: powell
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                res = smodel.fit(disp=False, method="powell", maxiter=2000)
            return res
        except Exception as e:
            logger.info("Retry (powell) failed: %s", e)

        # Retry 2: Nelder-Mead
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                res = smodel.fit(disp=False, method="nm", maxiter=2000)
            return res
        except Exception as e:
            logger.info("Retry (nm) failed: %s", e)

        # Final fallback: default fit (suppress convergence warnings)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            res = smodel.fit(disp=False)
        return res

    model_total_full = safe_fit(model)
    logger.debug("DCPO TOTAL SARIMA model trained.")
    offense_models.clear()
    for off in monthly_offense.columns:
        series_off = monthly_offense[off].astype(float)

        # skip if masyadong konti ang data
        if series_off.sum() == 0 or series_off.notna().sum() < 24:
            continue

        off_model = SARIMAX(
            series_off,
            order=BEST_ORDER,
            seasonal_order=BEST_SEASONAL_ORDER,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        try:
            off_fit = safe_fit(off_model)
            offense_models[off] = off_fit
        except Exception as e:
            logger.error("Failed to fit offense model '%s': %s", off, e)

    logger.debug("DCPO SARIMA models trained.")
    logger.debug("  • Months in series: %s", len(ts_total))
    logger.debug("  • Offense models : %s", len(offense_models))


# ============================================================
# Startup event: train once when server starts
# ============================================================

@app.on_event("startup")
async def startup_event():
    """Schedule training in a background thread so startup doesn't block the event loop.

    This prevents long-running synchronous fits from causing CancelledError tracebacks
    when the server (re)loader shuts down or restarts.
    """
    async def _run_training():
        try:
            await asyncio.to_thread(load_and_train)
        except asyncio.CancelledError:
            logger.error("Startup training task was cancelled.")
        except Exception as exc:
            logger.error("Error during startup training: %s", exc)

    # Launch background training task and don't await it here.
    asyncio.create_task(_run_training())


# ============================================================
# ROUTES
# ============================================================

@app.get("/", tags=["health"])
def health_check():
    return {"status": "ok", "message": "DCPO SARIMA API is running."}


@app.get("/total-forecast", response_model=TotalForecastResponse, tags=["forecast"])
def total_forecast(horizon: int = 12):
    """
    Forecast TOTAL crime (all offenses, all barangays) for the next N months.

    - horizon: 1–60 months (default 12)
    """
    global ts_total, model_total_full

    if ts_total is None or model_total_full is None:
        raise HTTPException(status_code=500, detail="Model not trained.")

    if horizon <= 0 or horizon > 60:
        raise HTTPException(status_code=400, detail="horizon must be between 1 and 60")

    fc = model_total_full.get_forecast(steps=horizon)
    mean = fc.predicted_mean
    ci = fc.conf_int()

    future_months = pd.date_range(
        start=ts_total.index[-1] + pd.offsets.MonthBegin(1),
        periods=horizon,
        freq="MS",
    )

    items: List[TotalForecastItem] = []
    for i in range(horizon):
        items.append(
            TotalForecastItem(
                month=str(future_months[i].date()),
                forecast_total=float(mean.iloc[i]),
                lower_ci=float(ci.iloc[i, 0]),
                upper_ci=float(ci.iloc[i, 1]),
            )
        )

    return TotalForecastResponse(
        status="success",
        horizon=horizon,
        data=items,
    )


@app.get("/offense-list", tags=["offense"])
def offense_list():
    """Listahan ng offenses na may SARIMA model."""
    if not offense_models:
        raise HTTPException(status_code=500, detail="Offense models not trained.")
    return {"status": "success", "offenses": sorted(offense_models.keys())}


@app.get("/top-crime", response_model=TopCrimeResponse, tags=["forecast"])
def top_crime(horizon: int = 12):
    """
    For each future month, predict which offense will have the HIGHEST count.

    - horizon: 1–60 months (default 12)
    """
    global ts_total, offense_models

    if ts_total is None or not offense_models:
        raise HTTPException(status_code=500, detail="Models not trained.")

    if horizon <= 0 or horizon > 60:
        raise HTTPException(status_code=400, detail="horizon must be between 1 and 60")

    future_months = pd.date_range(
        start=ts_total.index[-1] + pd.offsets.MonthBegin(1),
        periods=horizon,
        freq="MS",
    )

    # Forecast per offense
    offense_fc: Dict[str, np.ndarray] = {}
    for off, model in offense_models.items():
        fc = model.get_forecast(steps=horizon)
        offense_fc[off] = fc.predicted_mean.values

    rows: List[TopCrimeItem] = []
    for i in range(horizon):
        # scores for this step
        step_scores = {off: float(vals[i]) for off, vals in offense_fc.items()}

        # pick offense with highest forecast
        top_off = max(step_scores, key=step_scores.get)
        top_val = step_scores[top_off]

        rows.append(
            TopCrimeItem(
                month=str(future_months[i].date()),
                top_offense=top_off,
                top_value=top_val,
            )
        )

    return TopCrimeResponse(
        status="success",
        horizon=horizon,
        data=rows,
    )
